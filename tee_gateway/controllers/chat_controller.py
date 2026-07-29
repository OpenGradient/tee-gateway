import json
import time
import uuid
import logging

import connexion
from flask import Response
from dataclasses import dataclass, field
from typing import Any

from tee_gateway.models.create_chat_completion_request import (
    CreateChatCompletionRequest,
)
from tee_gateway.models.create_chat_completion_response import (
    CreateChatCompletionResponse,
)
from tee_gateway.models import (
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestFunctionMessage,
)

from langchain_core.messages import AIMessage, SystemMessage

from tee_gateway.tee_manager import get_tee_keys, compute_tee_msg_hash
from tee_gateway.llm_backend import (
    get_provider_from_model,
    get_chat_model_cached,
    convert_messages,
    extract_usage,
    validate_attachments,
    AttachmentValidationError,
    canonical_user_content,
)
from tee_gateway.image_generation import (
    create_image_generation_response,
    create_image_generation_streaming_response,
)
from tee_gateway.model_registry import get_model_config, model_supports_web_search
from tee_gateway.pricing import compute_session_cost
from tee_gateway.web_search import (
    WEB_SEARCH_TOOL_NAME,
    get_web_search_tool,
    web_search_available,
)
from tee_gateway.search_loop import (
    MAX_SEARCH_ROUNDS,
    SearchLoopState,
    execute_search_calls,
    run_search_loop,
    split_tool_calls,
    strip_search_tool_calls,
)

logger = logging.getLogger(__name__)


def _split_text_and_images(content: Any) -> tuple[str, list[str]]:
    """Split a LangChain message content into plain text and image data URIs.

    Image-generation models (e.g. Gemini "nano banana") return generated images
    as content blocks alongside any text. We collapse the text parts into a
    single string and collect each image as a ``data:`` URI. The image data is
    intentionally NOT folded into the text used for the TEE output hash — the
    OHTTP channel already guarantees confidentiality and integrity end-to-end,
    so images ride inside the signed envelope without being signed themselves.
    """
    if not isinstance(content, list):
        return (content or "", [])

    text_parts: list[str] = []
    images: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            text_parts.append(str(item))
            continue
        item_type = item.get("type")
        if item_type == "image_url":
            url = (item.get("image_url") or {}).get("url")
            if url:
                images.append(url)
        elif item_type in {"image", "media"}:
            # Standardized content block: raw base64 data + mime type.
            data = item.get("data")
            if data:
                mime = item.get("mime_type") or "image/png"
                images.append(f"data:{mime};base64,{data}")
        else:
            text_parts.append(item.get("text", "") or "")
    return ("".join(text_parts), images)


@dataclass
class _SearchEvent:
    """Streaming-only marker yielded between rounds of the search loop.

    The chunk handler in ``_create_streaming_response`` buffers tool-call
    fragments without knowing whose tool they belong to; this tells it. Carried
    in-band on the chunk iterator so that handler needs no other knowledge of the
    loop.
    """

    # Queries to announce to the client before the (blocking) searches run.
    queries: list[str] = field(default_factory=list)
    # The buffered calls were all ours: forget them and keep streaming.
    clear_buffer: bool = False
    # Terminal turn mixing our tool with the caller's: strip ours, forward theirs.
    drop_search_calls: bool = False


def _search_status_frame(model: str, query: str) -> dict[str, Any]:
    """An SSE frame telling the client which query is being searched.

    Shaped as an ordinary empty-delta chunk so a client that doesn't know about
    `web_search` ignores it harmlessly, with the status hung off a top-level key
    (the same convention `images` and `citations` use on the final frame).
    """
    return {
        "choices": [{"delta": {}, "index": 0, "finish_reason": None}],
        "model": model,
        "web_search": {"status": "searching", "query": query},
    }


def _accumulate_round(
    chunk: Any, round_text: list[str], round_calls: dict[int, dict[str, Any]]
) -> None:
    """Collect one round's text and tool-call fragments inside the search loop.

    Separate from the outer chunk handler's buffering because the two answer
    different questions: this one decides whether to search again, that one
    decides what to forward to the client.
    """
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        round_text.append(content)
    elif isinstance(content, list):
        round_text.extend(
            item.get("text", "") for item in content if isinstance(item, dict)
        )

    for fragment in getattr(chunk, "tool_call_chunks", None) or []:
        index = fragment.get("index", 0)
        entry = round_calls.setdefault(index, {"id": "", "name": "", "args": ""})
        if fragment.get("id"):
            entry["id"] = fragment["id"]
        if fragment.get("name"):
            entry["name"] = fragment["name"]
        args = fragment.get("args")
        if args:
            entry["args"] += args if isinstance(args, str) else json.dumps(args)


def _round_tool_calls(round_calls: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn buffered fragments into LangChain-shaped tool calls.

    Arguments arrive as a concatenated JSON string; a call whose arguments never
    parse is still returned with empty args so it is classified (and, if it is a
    web_search, answered with a "query is required" error the model can recover
    from) rather than silently vanishing.
    """
    calls: list[dict[str, Any]] = []
    for index in sorted(round_calls):
        entry = round_calls[index]
        if not entry["name"]:
            continue
        try:
            args = json.loads(entry["args"]) if entry["args"].strip() else {}
        except ValueError:
            logger.warning(
                "Could not parse streamed arguments for tool %r", entry["name"]
            )
            args = {}
        calls.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "args": args if isinstance(args, dict) else {},
            }
        )
    return calls


def create_chat_completion(body):
    """Create a chat completion (streaming or non-streaming)."""
    if not connexion.request.is_json:
        return {
            "error": "Unsupported Media Type",
            "message": "Request must be application/json",
        }, 415

    chat_request: CreateChatCompletionRequest = _parse_chat_request(
        connexion.request.get_json()
    )

    # Reject attachments the target model can't handle before doing provider work.
    try:
        validate_attachments(chat_request.messages, chat_request.model)
    except AttachmentValidationError as e:
        return {"error": "Invalid attachment", "message": str(e)}, 400

    if chat_request.stream:
        return _create_streaming_response(chat_request)
    else:
        return _create_non_streaming_response(chat_request)


def _build_user_tools_list(chat_request: CreateChatCompletionRequest) -> list:
    """Normalize the caller's own function tools into bind_tools() form."""
    tools_list: list = []
    if chat_request.tools:
        for tool in chat_request.tools:
            if isinstance(tool, dict):
                func = tool.get("function", {})
                tools_list.append(
                    {"type": tool.get("type", "function"), "function": func}
                )
            else:
                tools_list.append(tool)
    return tools_list


def _search_enabled(
    chat_request: CreateChatCompletionRequest,
    provider: str,
    anthropic_structured: bool,
) -> bool:
    """Whether to bind and run the gateway's web_search tool for this request.

    Off when the caller didn't ask, when no Exa key was injected (rather than
    advertising a tool that always fails), for image models, and for Anthropic's
    structured-output path — ``with_structured_output`` occupies the tool slot
    with a forced schema tool, so a search tool bound beside it would never be
    callable. Every other model gets search, regardless of provider.
    """
    if not getattr(chat_request, "web_search", False):
        return False
    if anthropic_structured and provider == "anthropic":
        logger.info(
            "web_search requested with Anthropic structured output; skipping "
            "search (structured output uses a forced tool)"
        )
        return False
    if not model_supports_web_search(chat_request.model):
        return False
    if not web_search_available():
        logger.warning(
            "web_search requested but no Exa API key is configured; answering "
            "without search"
        )
        return False
    return True


def _needs_responses_api_for_tools(provider: str, cfg, tools_list: list) -> bool:
    """Whether this request must use OpenAI's Responses API because of tools.

    The gpt-5.6 family (``responses_api_for_tools=True``) rejects function tools
    combined with its default ``reasoning_effort`` on the Chat Completions
    endpoint; the Responses API supports both together. Only relevant when
    function tools are actually bound and the provider is OpenAI.
    """
    return (
        provider == "openai"
        and getattr(cfg, "responses_api_for_tools", False)
        and bool(tools_list)
    )


def _normalize_response_format(rf) -> dict:
    """Coerce response_format to a plain dict, preserving all fields including json_schema."""
    if isinstance(rf, dict):
        return rf
    if hasattr(rf, "model_dump"):
        return rf.model_dump()
    return vars(rf)


def _invoke_anthropic_structured(
    model, rf: dict, langchain_messages: list
) -> AIMessage:
    """
    Use LangChain's with_structured_output() for Anthropic structured output.

    Anthropic does not support response_format via bind(). For json_schema, we use
    with_structured_output(schema, method="json_schema") which calls Anthropic's
    native structured output API. The parsed dict result is re-wrapped as an AIMessage
    so all downstream signing/response-building code stays unchanged.

    json_object (no schema) has no Anthropic native equivalent — callers should use
    json_schema with an explicit schema instead.
    """
    rf_type = rf.get("type", "text")
    if rf_type != "json_schema":
        raise ValueError(
            f"response_format type '{rf_type}' is not natively supported by Anthropic. "
            "Use json_schema with an explicit schema instead."
        )

    schema_obj = rf.get("json_schema", {})
    schema_def = schema_obj.get("schema", {})
    name = schema_obj.get("name", "output")
    strict = schema_obj.get("strict", False)

    # LangChain-Anthropic derives the tool function name from the schema's "title" key.
    # The OpenAI-compatible json_schema wrapper puts this as "name" one level up, so
    # we inject it into the schema dict if it isn't already there.
    if "title" not in schema_def:
        schema_def = {**schema_def, "title": name}

    structured = model.with_structured_output(
        schema_def, method="json_schema", strict=strict, include_raw=True
    )
    result = structured.invoke(langchain_messages)

    # include_raw=True returns {"raw": AIMessage, "parsed": dict, "parsing_error": ...}
    if isinstance(result, dict) and result.get("parsing_error"):
        raise ValueError(f"Structured output parsing failed: {result['parsing_error']}")
    raw_msg = result.get("raw") if isinstance(result, dict) else None
    parsed = result.get("parsed") if isinstance(result, dict) else result
    content_str = json.dumps(parsed) if isinstance(parsed, dict) else str(parsed)

    # Preserve usage_metadata from the raw Anthropic response so the x402
    # cost calculator can extract token counts from the final response body.
    msg = AIMessage(content=content_str)
    if (
        raw_msg is not None
        and hasattr(raw_msg, "usage_metadata")
        and raw_msg.usage_metadata
    ):
        msg.usage_metadata = raw_msg.usage_metadata
    return msg


def _messages_contain_json_word(messages: list) -> bool:
    """Return True if any message content contains the word 'json' (case-insensitive).

    Handles both plain-string content and list-of-parts content (multimodal messages).
    """
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, str):
            if "json" in content.lower():
                return True
        elif isinstance(content, list):
            for part in content:
                text = part.get("text", "") if isinstance(part, dict) else str(part)
                if "json" in text.lower():
                    return True
    return False


def _create_non_streaming_response(chat_request: CreateChatCompletionRequest):
    """Handle non-streaming chat completion via direct LangChain call."""
    try:
        logger.info("=" * 80)
        logger.info(f"Chat request for model: {chat_request.model}")
        logger.info(f"Number of messages: {len(chat_request.messages)}")

        provider = get_provider_from_model(chat_request.model)
        cfg = get_model_config(chat_request.model)

        # Serialize request for hashing (canonical, deterministic)
        request_dict = _chat_request_to_dict(chat_request)
        request_bytes = json.dumps(request_dict, sort_keys=True).encode("utf-8")

        # Image-generation models (xAI Grok, ByteDance Seedream) are served via a
        # dedicated /images/generations endpoint — not the chat path — but are
        # surfaced through this same endpoint, returning images out-of-band just
        # like Gemini's inline-image models.
        if cfg.image_generation:
            return create_image_generation_response(chat_request, request_bytes)

        # response_format is resolved before the model is built: whether Anthropic
        # takes the structured-output path decides whether web search can run at
        # all (see _search_enabled).
        rf_dict: dict | None = None
        if chat_request.response_format:
            rf = _normalize_response_format(chat_request.response_format)
            if rf.get("type", "text") != "text":
                rf_dict = rf

        # Build the tools list first: some OpenAI models (gpt-5.6 family) must be
        # constructed against the Responses API when function tools are bound.
        user_tools = _build_user_tools_list(chat_request)
        search_enabled = _search_enabled(chat_request, provider, rf_dict is not None)
        tools_list = (
            user_tools + [get_web_search_tool()] if search_enabled else user_tools
        )

        base_model = get_chat_model_cached(
            model=chat_request.model,
            temperature=float(chat_request.temperature)
            if chat_request.temperature is not None
            else 0.0,
            max_tokens=chat_request.max_tokens or 4096,
            force_responses_api=_needs_responses_api_for_tools(
                provider, cfg, tools_list
            ),
        )

        model = base_model.bind_tools(tools_list) if tools_list else base_model
        # The search loop's final round runs without the search tool bound, which
        # is what turns the round cap into "answer now" instead of "hand back an
        # unanswerable search request".
        model_without_search = (
            (base_model.bind_tools(user_tools) if user_tools else base_model)
            if search_enabled
            else model
        )

        # Bind response_format if provided (json_object or json_schema).
        # Anthropic does not support response_format via bind(); use
        # with_structured_output() for json_schema instead (json_object has no
        # Anthropic native equivalent and raises a clear error).
        if rf_dict is not None and provider != "anthropic":
            model = model.bind(response_format=rf_dict)
            model_without_search = model_without_search.bind(response_format=rf_dict)

        langchain_messages = convert_messages(chat_request.messages)

        # OpenAI (and compatible providers) require the word "json" to appear
        # somewhere in the messages when response_format.type == "json_object".
        # Inject a brief system instruction if none of the messages satisfy this.
        if rf_dict and rf_dict.get("type") == "json_object":
            if not _messages_contain_json_word(langchain_messages):
                langchain_messages = [
                    SystemMessage(content="Respond in JSON format.")
                ] + langchain_messages

        search_state = SearchLoopState()
        if search_enabled:
            # Run searches to completion inside the enclave, then answer. The
            # caller sent one request and gets one answer; the rounds in between
            # are invisible except in the token usage they add.
            response = run_search_loop(
                model,
                model_without_search,
                langchain_messages,
                search_state,
            )
        elif rf_dict and provider == "anthropic":
            response = _invoke_anthropic_structured(model, rf_dict, langchain_messages)
        else:
            response = model.invoke(langchain_messages)

        # Normalize content (Gemini may return a list of content parts, and
        # image-generation models return image blocks alongside any text).
        content_str, generated_images = _split_text_and_images(response.content)

        message_dict: dict[str, Any] = {
            "role": "assistant",
            "content": content_str,
        }
        # Surface generated images out-of-band on the message. They are not part
        # of the signed output hash (see _split_text_and_images).
        if generated_images:
            message_dict["images"] = generated_images
        # Sources the model searched, surfaced out-of-band alongside images and
        # on the same terms: the answer text is signed, this metadata about what
        # informed it rides inside the OHTTP envelope unsigned.
        if search_state.citations:
            message_dict["citations"] = search_state.citations

        finish_reason = "stop"
        # Any web_search calls left on a terminal turn belong to a turn that also
        # called one of the caller's tools; they are dropped rather than handed to
        # a client that has no way to run them.
        client_tool_calls = (
            strip_search_tool_calls(response)
            if search_enabled
            else (getattr(response, "tool_calls", None) or [])
        )
        if client_tool_calls:
            finish_reason = "tool_calls"
            message_dict["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("args", {})),
                    },
                }
                for tc in client_tool_calls
            ]

        # For tool-call responses, hash the serialized tool calls so the
        # signature covers which tools were invoked and with what arguments.
        if finish_reason == "tool_calls" and message_dict.get("tool_calls"):
            response_content = json.dumps(message_dict["tool_calls"], sort_keys=True)
        else:
            response_content = content_str

        timestamp = int(time.time())
        msg_hash, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
            request_bytes, response_content, timestamp
        )
        tee_keys = get_tee_keys()
        signature = tee_keys.sign_data(msg_hash)

        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": timestamp,
            "model": chat_request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason,
                }
            ],
            "tee_signature": signature,
            "tee_request_hash": input_hash_hex,
            "tee_output_hash": output_hash_hex,
            "tee_timestamp": timestamp,
            "tee_id": f"0x{tee_keys.get_tee_id()}",
        }

        logger.debug(
            f"Response Final\n\tTEE Signature: {signature}\n\tTEE request hash: {input_hash_hex}\n\tTEE output hash: {output_hash_hex}\n\tTEE timestamp: {timestamp}\n\tTEE ID: 0x{tee_keys.get_tee_id()}"
        )

        # TODO: If no usage is returned, we should compute it here.
        # With search on, `usage` is the sum over every round of the loop — each
        # round re-sent the conversation plus the accumulated search results, and
        # the caller is charged for all of those tokens, not just the last round's.
        usage = search_state.usage if search_enabled else extract_usage(response)
        if usage:
            # Surface the standard OpenAI usage triple on the response; the
            # reasoning split rides along to the cost calculator via `usage`.
            openai_response["usage"] = {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            }
            cost = compute_session_cost(
                chat_request.model,
                usage,
                web_search_count=search_state.search_count,
            )
            if cost is not None:
                openai_response["opengradient"] = cost.model_dump(mode="json")

        # Validate schema (the extra tee_* fields are preserved by returning dict directly)
        CreateChatCompletionResponse.from_dict(openai_response)
        return openai_response

    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}", exc_info=True)
        return {
            "error": str(e) or "Request processing failed",
            "exception_type": type(e).__name__,
        }, 500


def _create_streaming_response(chat_request: CreateChatCompletionRequest):
    """Handle streaming chat completion via direct LangChain call."""
    try:
        provider = get_provider_from_model(chat_request.model)
        cfg = get_model_config(chat_request.model)
        # Gemini inline-image models return a single image rather than a token
        # stream — invoke once and emit the result inside the SSE envelope.
        image_output_model = cfg.image_output

        request_dict = _chat_request_to_dict(chat_request)
        request_bytes = json.dumps(request_dict, sort_keys=True).encode("utf-8")

        # Image-generation models (xAI Grok, ByteDance Seedream) use a dedicated
        # images endpoint; handle them without building a chat model.
        if get_model_config(chat_request.model).image_generation:
            return create_image_generation_streaming_response(
                chat_request, request_bytes
            )

        # response_format is resolved before the model is built: whether Anthropic
        # takes the structured-output path decides whether web search can run at
        # all (see _search_enabled).
        rf_dict: dict | None = None
        if chat_request.response_format:
            rf = _normalize_response_format(chat_request.response_format)
            if rf.get("type", "text") != "text":
                rf_dict = rf
        anthropic_structured_rf: dict | None = (
            rf_dict if rf_dict is not None and provider == "anthropic" else None
        )

        # Build the tools list first: some OpenAI models (gpt-5.6 family) must be
        # constructed against the Responses API when function tools are bound.
        user_tools = _build_user_tools_list(chat_request)
        search_enabled = _search_enabled(chat_request, provider, rf_dict is not None)
        tools_list = (
            user_tools + [get_web_search_tool()] if search_enabled else user_tools
        )

        # OpenAI and Anthropic stream tool calls as fragments that must be
        # buffered and flushed once complete. Gemini emits complete tool calls.
        # With search on, ALWAYS buffer: a fragment can't be forwarded to the
        # client until the round ends and we know whether the call was a
        # web_search this gateway will answer itself.
        buffer_tool_calls = provider in ["openai", "anthropic"] or search_enabled

        base_model = get_chat_model_cached(
            model=chat_request.model,
            temperature=float(chat_request.temperature)
            if chat_request.temperature is not None
            else 0.0,
            max_tokens=chat_request.max_tokens or 4096,
            force_responses_api=_needs_responses_api_for_tools(
                provider, cfg, tools_list
            ),
        )

        model = base_model.bind_tools(tools_list) if tools_list else base_model
        # The search loop's final round runs without the search tool bound, so a
        # model that would keep searching has to answer with what it has.
        model_without_search = (
            (base_model.bind_tools(user_tools) if user_tools else base_model)
            if search_enabled
            else model
        )

        # Bind response_format if provided (json_object or json_schema).
        # Anthropic does not support response_format via bind(); use
        # with_structured_output() for json_schema instead (json_object has no
        # Anthropic native equivalent and raises a clear error).
        if rf_dict is not None and anthropic_structured_rf is None:
            model = model.bind(response_format=rf_dict)
            model_without_search = model_without_search.bind(response_format=rf_dict)

        langchain_messages = convert_messages(chat_request.messages)

        # OpenAI (and compatible providers) require the word "json" to appear
        # somewhere in the messages when response_format.type == "json_object".
        # Inject a brief system instruction if none of the messages satisfy this.
        # `rf` is defined inside the `if chat_request.response_format` block above;
        # guard with the same condition before accessing it.
        if (
            chat_request.response_format
            and anthropic_structured_rf is None
            and rf.get("type") == "json_object"
        ):
            if not _messages_contain_json_word(langchain_messages):
                langchain_messages = [
                    SystemMessage(content="Respond in JSON format.")
                ] + langchain_messages

        tee_keys = get_tee_keys()

        # For Anthropic structured output, with_structured_output() invokes
        # synchronously and returns a complete dict — streaming partial JSON is
        # not meaningful for schema-validated output. We invoke once and emit the
        # full result as a single content chunk inside the SSE stream.
        if anthropic_structured_rf is not None:
            ai_msg = _invoke_anthropic_structured(
                model, anthropic_structured_rf, langchain_messages
            )
            anthropic_structured_content: str | None = (
                ai_msg.content
                if isinstance(ai_msg.content, str)
                else json.dumps(ai_msg.content)
            )
            # Capture usage now — the streaming loop never runs for this path,
            # so final_usage would otherwise stay None and x402 cannot charge.
            anthropic_structured_usage: dict[str, int] | None = None

            if hasattr(ai_msg, "usage_metadata") and ai_msg.usage_metadata:
                um = ai_msg.usage_metadata

                anthropic_structured_usage = {
                    "input_tokens": um.get("input_tokens", 0),
                    "output_tokens": um.get("output_tokens", 0),
                    "total_tokens": um.get("total_tokens", 0),
                }
        else:
            anthropic_structured_content = None
            anthropic_structured_usage = None

        search_state = SearchLoopState()

        def _streamed_search_rounds():
            """Yield chunks across every round of the in-enclave search loop.

            Written as a generator wrapping ``model.stream`` so the chunk handling
            below is identical whether or not search is on — one round or five, it
            sees one flat stream of chunks. Between rounds it yields a
            ``_SearchEvent`` telling that handler what to do with the tool-call
            fragments it just buffered, since only this generator knows whether
            they were ``web_search`` calls it answered itself.
            """
            loop_messages = list(langchain_messages)
            rounds = MAX_SEARCH_ROUNDS + 1 if search_enabled else 1

            for round_index in range(rounds):
                last_round = search_enabled and round_index == MAX_SEARCH_ROUNDS
                active_model = model_without_search if last_round else model

                round_text: list[str] = []
                round_calls: dict[int, dict[str, Any]] = {}

                for chunk in active_model.stream(loop_messages):
                    if search_enabled:
                        _accumulate_round(chunk, round_text, round_calls)
                    yield chunk

                if not search_enabled:
                    return

                ours, theirs = split_tool_calls(_round_tool_calls(round_calls))
                if theirs or not ours:
                    # Terminal turn. A turn that asked for both our search and one
                    # of the caller's tools can't be served by either side alone,
                    # so the caller's tools win and ours are dropped downstream.
                    if ours:
                        logger.info(
                            "Dropping %d streamed web_search call(s) from a turn "
                            "that also called %d client tool(s)",
                            len(ours),
                            len(theirs),
                        )
                        yield _SearchEvent(drop_search_calls=True)
                    return

                # Announce the queries before running them: the Exa call blocks
                # for a second or two and the client should say why it is waiting.
                yield _SearchEvent(
                    queries=[
                        q
                        for q in ((c.get("args") or {}).get("query") for c in ours)
                        if isinstance(q, str) and q.strip()
                    ],
                    clear_buffer=True,
                )

                loop_messages.append(
                    AIMessage(content="".join(round_text), tool_calls=ours)
                )
                loop_messages.extend(execute_search_calls(ours, search_state))

        def generate():
            full_content = ""
            final_usage = None
            buffered_tool_calls = {}
            finish_reason = "stop"
            generated_images: list[str] = []

            try:
                if image_output_model:
                    # Image generation isn't a token stream: invoke once, emit
                    # any caption text as a content delta, and carry the image
                    # out-of-band on the final frame (it is not signed).
                    response = model.invoke(langchain_messages)
                    text_content, generated_images = _split_text_and_images(
                        response.content
                    )
                    full_content = text_content
                    if text_content:
                        data = {
                            "choices": [
                                {
                                    "delta": {
                                        "content": text_content,
                                        "role": "assistant",
                                    },
                                    "index": 0,
                                    "finish_reason": None,
                                }
                            ],
                            "model": chat_request.model,
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    # BILLING (image output): Gemini bills each generated image as
                    # ~1290 output tokens reported in candidates_token_count, which
                    # langchain-google-genai folds into usage_metadata.output_tokens
                    # (-> completion_tokens -> output_price_usd). So images are charged
                    # purely via the normal token path; the image bytes themselves are
                    # never metered by size. CAVEAT: if the provider omits usage_metadata
                    # (langchain then yields None), final_usage stays None, the final
                    # frame carries no 'usage', and compute_session_cost is skipped — the
                    # client is NOT charged and gets a free image (fail-open). Acceptable
                    # for now since Gemini reliably returns usage on success; revisit with
                    # a ~1290-token-per-image fallback if that ever changes.
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        final_usage = {}
                        for k, v in response.usage_metadata.items():
                            if isinstance(v, (int, float)):
                                final_usage[k] = v
                        # Thinking tokens are billed at the cheaper text rate; they
                        # live in the nested output_token_details dict (skipped by
                        # the int/float loop above), so pull them out explicitly.
                        _otd = response.usage_metadata.get("output_token_details")
                        if isinstance(_otd, dict) and isinstance(
                            _otd.get("reasoning"), (int, float)
                        ):
                            final_usage["reasoning"] = _otd["reasoning"]
                    chunks_iter: list = []
                elif anthropic_structured_content is not None:
                    # Emit the pre-computed structured result as a single chunk.
                    # Seed final_usage from the synchronous invoke so the final
                    # SSE chunk carries a 'usage' dict for the x402 cost calculator.
                    full_content = anthropic_structured_content
                    final_usage = anthropic_structured_usage
                    data = {
                        "choices": [
                            {
                                "delta": {"content": full_content, "role": "assistant"},
                                "index": 0,
                                "finish_reason": None,
                            }
                        ],
                        "model": chat_request.model,
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    chunks_iter = []
                else:
                    chunks_iter = _streamed_search_rounds()  # type: ignore[assignment]

                for chunk in chunks_iter:
                    # --- Search-round boundary (in-enclave web search) ---
                    # Not a model chunk: an instruction about the tool-call
                    # fragments buffered so far, plus the queries to tell the
                    # client about.
                    if isinstance(chunk, _SearchEvent):
                        if chunk.clear_buffer:
                            # Those calls were web_search calls this gateway is
                            # answering itself — the client must never see them.
                            buffered_tool_calls = {}
                            finish_reason = "stop"
                        if chunk.drop_search_calls:
                            buffered_tool_calls = {
                                index: tc
                                for index, tc in buffered_tool_calls.items()
                                if tc["function"]["name"] != WEB_SEARCH_TOOL_NAME
                            }
                            if not buffered_tool_calls:
                                finish_reason = "stop"
                        for query in chunk.queries:
                            yield f"data: {json.dumps(_search_status_frame(chat_request.model, query))}\n\n"
                        continue

                    # --- Text content ---
                    if chunk.content:
                        if isinstance(chunk.content, str):
                            content_str = chunk.content
                        elif isinstance(chunk.content, list):
                            content_str = "".join(
                                item.get("text", "")
                                if isinstance(item, dict)
                                else str(item)
                                for item in chunk.content
                            )
                        else:
                            content_str = str(chunk.content)

                        if content_str:
                            full_content += content_str
                            data = {
                                "choices": [
                                    {
                                        "delta": {
                                            "content": content_str,
                                            "role": "assistant",
                                        },
                                        "index": 0,
                                        "finish_reason": None,
                                    }
                                ],
                                "model": chat_request.model,
                            }
                            yield f"data: {json.dumps(data)}\n\n"

                    # --- Tool call chunks ---
                    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                        finish_reason = "tool_calls"
                        for tc_chunk in chunk.tool_call_chunks:
                            tc_index = tc_chunk.get("index", 0)

                            if tc_index not in buffered_tool_calls:
                                buffered_tool_calls[tc_index] = {
                                    "id": tc_chunk.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": tc_chunk.get("name", ""),
                                        "arguments": "",
                                    },
                                }

                            if tc_chunk.get("id"):
                                buffered_tool_calls[tc_index]["id"] = tc_chunk["id"]
                            if tc_chunk.get("name"):
                                buffered_tool_calls[tc_index]["function"]["name"] = (
                                    tc_chunk["name"]
                                )
                            if tc_chunk.get("args"):
                                args_value = tc_chunk["args"]
                                if isinstance(args_value, dict):
                                    args_str = json.dumps(args_value)
                                elif isinstance(args_value, str):
                                    args_str = args_value
                                else:
                                    args_str = str(args_value)
                                buffered_tool_calls[tc_index]["function"][
                                    "arguments"
                                ] += args_str

                            # For providers that don't need buffering, emit each fragment immediately
                            if not buffer_tool_calls:
                                delta = {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "index": tc_index,
                                            "type": "function",
                                            "function": {},
                                        }
                                    ],
                                }
                                if tc_chunk.get("id"):
                                    delta["tool_calls"][0]["id"] = tc_chunk["id"]
                                if tc_chunk.get("name"):
                                    delta["tool_calls"][0]["function"]["name"] = (
                                        tc_chunk["name"]
                                    )
                                if tc_chunk.get("args"):
                                    args_value = tc_chunk["args"]
                                    if isinstance(args_value, dict):
                                        delta["tool_calls"][0]["function"][
                                            "arguments"
                                        ] = json.dumps(args_value)
                                    elif isinstance(args_value, str):
                                        delta["tool_calls"][0]["function"][
                                            "arguments"
                                        ] = args_value
                                    else:
                                        delta["tool_calls"][0]["function"][
                                            "arguments"
                                        ] = str(args_value)

                                    if not delta["tool_calls"][0]["function"]:
                                        del delta["tool_calls"][0]["function"]

                                data = {
                                    "choices": [
                                        {
                                            "delta": delta,
                                            "index": 0,
                                            "finish_reason": None,
                                        }
                                    ],
                                    "model": chat_request.model,
                                }
                                yield f"data: {json.dumps(data)}\n\n"

                    # --- Usage metadata ---
                    # Accumulate deltas rather than replacing: Gemini returns cumulative
                    # usageMetadata on every chunk and LangChain emits deltas via
                    # subtract_usage(), so input_tokens only appears non-zero in the
                    # *first* chunk carrying usage. Replacing on each chunk would
                    # overwrite that value with 0 from all subsequent chunks.
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        if final_usage is None:
                            final_usage = {}
                        for k, v in chunk.usage_metadata.items():
                            if isinstance(v, (int, float)):
                                final_usage[k] = final_usage.get(k, 0) + v
                        # Thinking tokens (billed at the cheaper text rate) are
                        # nested in output_token_details and emitted as deltas like
                        # the top-level counts, so accumulate them the same way.
                        _otd = chunk.usage_metadata.get("output_token_details")
                        if isinstance(_otd, dict) and isinstance(
                            _otd.get("reasoning"), (int, float)
                        ):
                            final_usage["reasoning"] = (
                                final_usage.get("reasoning", 0) + _otd["reasoning"]
                            )

                # Flush buffered tool calls for OpenAI/Anthropic
                if buffer_tool_calls and buffered_tool_calls:
                    for tc_index, tc in buffered_tool_calls.items():
                        delta = {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": tc_index,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": tc["function"]["arguments"],
                                    },
                                }
                            ],
                        }
                        data = {
                            "choices": [
                                {"delta": delta, "index": 0, "finish_reason": None}
                            ],
                            "model": chat_request.model,
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                # Sign the completed response.
                # For tool-call responses, hash the buffered tool calls so the
                # signature covers the actual invocations.
                timestamp = int(time.time())
                if finish_reason == "tool_calls" and buffered_tool_calls:
                    tool_calls_list = [
                        buffered_tool_calls[k]
                        for k in sorted(buffered_tool_calls.keys())
                    ]
                    output_content = json.dumps(tool_calls_list, sort_keys=True)
                else:
                    output_content = full_content

                msg_hash, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
                    request_bytes, output_content, timestamp
                )
                tee_signature = tee_keys.sign_data(msg_hash)

                final_data = {
                    "choices": [
                        {"delta": {}, "index": 0, "finish_reason": finish_reason}
                    ],
                    "model": chat_request.model,
                    "tee_signature": tee_signature,
                    "tee_timestamp": timestamp,
                    "tee_request_hash": input_hash_hex,
                    "tee_output_hash": output_hash_hex,
                    "tee_id": f"0x{tee_keys.get_tee_id()}",
                }
                # Generated images travel out-of-band on the final frame; they
                # are not part of the signed output hash.
                if generated_images:
                    final_data["images"] = generated_images
                # Likewise the sources the model searched: the answer text is
                # signed, this metadata about what informed it is not.
                if search_state.citations:
                    final_data["citations"] = search_state.citations

                logger.debug(
                    f"Response Final\n\tTEE Signature: {tee_signature}\n\tTEE request hash: {input_hash_hex}\n\tTEE output hash: {output_hash_hex}\n\tTEE timestamp: {timestamp}\n\tTEE ID: 0x{tee_keys.get_tee_id()}"
                )

                # TODO: If no usage is returned, we should compute it here.
                if final_usage:
                    final_data["usage"] = {
                        "prompt_tokens": final_usage.get("input_tokens", 0),
                        "completion_tokens": final_usage.get("output_tokens", 0),
                        "total_tokens": final_usage.get("total_tokens", 0),
                    }
                    # Pass thinking tokens to the cost calculator (for the image
                    # dual-rate split) without polluting the OpenAI usage triple.
                    # `final_usage` already sums every round of the search loop —
                    # each round re-sent the conversation plus the accumulated
                    # results, and the caller pays for all of those input tokens.
                    cost_usage = dict(
                        final_data["usage"],
                        reasoning_tokens=final_usage.get("reasoning", 0),
                    )
                    cost = compute_session_cost(
                        chat_request.model,
                        cost_usage,
                        web_search_count=search_state.search_count,
                    )
                    if cost is not None:
                        # final_data is hand-serialized to SSE via json.dumps below,
                        # which doesn't go through Flask's JSONEncoder — so do the
                        # serialization ourselves here.
                        final_data["opengradient"] = cost.model_dump(mode="json")
                    logger.info(
                        f"Stream completed — usage: {final_data['usage']}, "
                        f"finish: {finish_reason}, "
                        f"searches: {search_state.search_count}, "
                        f"inputHash: {input_hash_hex[:16]}..., outputHash: {output_hash_hex[:16]}..."
                    )

                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {str(e)}", exc_info=True)
                # The HTTP status is already locked to 200 (headers flushed
                # before the first chunk), so this in-band error event is the
                # only channel left to tell the client why the stream died.
                # `exception_type` is a content-free class name the client can
                # bucket on; `error` carries the human-readable detail. This
                # event IS the terminal marker for the error path — we do NOT
                # emit a trailing `[DONE]`, which conventionally signals a
                # clean completion and would mis-signal an errored stream.
                error_payload = {
                    "error": str(e) or "Stream processing failed",
                    "exception_type": type(e).__name__,
                }
                yield f"data: {json.dumps(error_payload)}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        logger.error(f"Stream setup error: {str(e)}", exc_info=True)
        return {
            "error": str(e) or "Stream setup failed",
            "exception_type": type(e).__name__,
        }, 500


# ---------------------------------------------------------------------------
# Request parsing helpers
# ---------------------------------------------------------------------------


def _chat_request_to_dict(chat_request: CreateChatCompletionRequest) -> dict:
    """Serialize a CreateChatCompletionRequest to a canonical dict for hashing."""
    messages = []
    for msg in chat_request.messages:
        if isinstance(msg, ChatCompletionRequestSystemMessage):
            messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, ChatCompletionRequestUserMessage):
            messages.append(
                {
                    "role": "user",
                    "content": canonical_user_content(msg.content),
                }
            )
        elif isinstance(msg, ChatCompletionRequestAssistantMessage):
            m = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                m["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(m)
        elif isinstance(msg, ChatCompletionRequestToolMessage):
            messages.append(
                {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                }
            )
        elif isinstance(msg, ChatCompletionRequestFunctionMessage):
            messages.append(
                {"role": "function", "content": msg.content, "name": msg.name}
            )

    d = {
        "model": chat_request.model,
        "messages": messages,
        "temperature": float(chat_request.temperature)
        if chat_request.temperature is not None
        else 0.0,
    }
    if chat_request.max_tokens is not None:
        d["max_tokens"] = chat_request.max_tokens
    if chat_request.stop:
        d["stop"] = chat_request.stop
    if chat_request.tools:
        d["tools"] = (
            chat_request.tools
            if isinstance(chat_request.tools, list)
            else list(chat_request.tools)
        )
    if chat_request.response_format:
        d["response_format"] = _normalize_response_format(chat_request.response_format)
    if chat_request.web_search:
        d["web_search"] = True
    return d


def _parse_chat_request(chat_request_dict: dict) -> CreateChatCompletionRequest:
    messages = [_parse_message(msg) for msg in chat_request_dict.get("messages", [])]
    return CreateChatCompletionRequest(
        messages=messages,
        model=chat_request_dict.get("model"),
        frequency_penalty=chat_request_dict.get("frequency_penalty"),
        logit_bias=chat_request_dict.get("logit_bias"),
        max_tokens=chat_request_dict.get("max_tokens"),
        n=chat_request_dict.get("n"),
        presence_penalty=chat_request_dict.get("presence_penalty"),
        response_format=chat_request_dict.get("response_format"),
        seed=chat_request_dict.get("seed"),
        stop=chat_request_dict.get("stop"),
        stream=chat_request_dict.get("stream"),
        temperature=chat_request_dict.get("temperature"),
        top_p=chat_request_dict.get("top_p"),
        tools=chat_request_dict.get("tools"),
        tool_choice=chat_request_dict.get("tool_choice"),
        user=chat_request_dict.get("user"),
        web_search=chat_request_dict.get("web_search", False),
    )


def _parse_message(message_dict: dict):
    role = message_dict.get("role")
    if role == "user":
        return ChatCompletionRequestUserMessage.from_dict(message_dict)
    elif role == "system":
        return ChatCompletionRequestSystemMessage.from_dict(message_dict)
    elif role == "assistant":
        return ChatCompletionRequestAssistantMessage.from_dict(message_dict)
    elif role == "tool":
        return ChatCompletionRequestToolMessage.from_dict(message_dict)
    elif role == "function":
        return ChatCompletionRequestFunctionMessage.from_dict(message_dict)
    else:
        raise ValueError(f"Unknown role: {role}")
