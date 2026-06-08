import json
import time
import uuid
import logging

import connexion
from flask import Response
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
    get_web_search_tool,
    extract_web_search_count,
    convert_messages,
    extract_usage,
    generate_images,
)
from tee_gateway.model_registry import get_model_config
from tee_gateway.pricing import compute_session_cost

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


def _extract_image_prompt(langchain_messages: list) -> str:
    """Collapse the user-turn text into a single image-generation prompt.

    Image-generation models (xAI Grok, ByteDance Seedream) take a single text
    prompt rather than a chat transcript, so we join the text of all human
    messages. System/assistant/tool turns are ignored.
    """
    from langchain_core.messages import HumanMessage

    parts: list[str] = []
    for m in langchain_messages:
        if isinstance(m, HumanMessage):
            content = m.content
            parts.append(content if isinstance(content, str) else str(content))
    return "\n".join(p for p in parts if p)


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

    if chat_request.stream:
        return _create_streaming_response(chat_request)
    else:
        return _create_non_streaming_response(chat_request)


def _build_tools_list(chat_request: CreateChatCompletionRequest, provider: str) -> list:
    """Build the combined tools list (user tools + native web search tool).

    User-supplied function tools and the provider's built-in web search tool are
    bound together in a single bind_tools() call. xAI configures search at
    construction time (no tool), and providers without native web search return
    no tool — in both cases only user tools are included.
    """
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

    if getattr(chat_request, "web_search", False):
        ws_tool = get_web_search_tool(provider)
        if ws_tool is not None:
            tools_list.append(ws_tool)

    return tools_list


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


def _create_image_generation_response(
    chat_request: CreateChatCompletionRequest, request_bytes: bytes
):
    """Non-streaming image generation via a provider's images endpoint.

    Surfaces generated images on the message under the ``images`` key exactly
    like Gemini's inline-image models. There is no text to sign, so the signature
    covers the request hash and an empty output; the image bytes ride out-of-band
    inside the OHTTP envelope. Billing is flat per generated image.
    """
    langchain_messages = convert_messages(chat_request.messages)
    prompt = _extract_image_prompt(langchain_messages)
    images, image_count = generate_images(
        chat_request.model, prompt, n=chat_request.n or 1
    )

    message_dict: dict[str, Any] = {"role": "assistant", "content": ""}
    if images:
        message_dict["images"] = images

    timestamp = int(time.time())
    msg_hash, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
        request_bytes, "", timestamp
    )
    tee_keys = get_tee_keys()
    signature = tee_keys.sign_data(msg_hash)

    openai_response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": timestamp,
        "model": chat_request.model,
        "choices": [{"index": 0, "message": message_dict, "finish_reason": "stop"}],
        "tee_signature": signature,
        "tee_request_hash": input_hash_hex,
        "tee_output_hash": output_hash_hex,
        "tee_timestamp": timestamp,
        "tee_id": f"0x{tee_keys.get_tee_id()}",
    }

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    openai_response["usage"] = usage
    cost = compute_session_cost(chat_request.model, usage, image_count=image_count)
    if cost is not None:
        openai_response["opengradient"] = cost.model_dump(mode="json")

    CreateChatCompletionResponse.from_dict(openai_response)
    return openai_response


def _create_image_generation_streaming_response(
    chat_request: CreateChatCompletionRequest, request_bytes: bytes
):
    """Streaming image generation: image gen is not a token stream, so we invoke
    once and emit the result on the final SSE frame (mirrors the Gemini path)."""
    tee_keys = get_tee_keys()

    def generate():
        try:
            langchain_messages = convert_messages(chat_request.messages)
            prompt = _extract_image_prompt(langchain_messages)
            images, image_count = generate_images(
                chat_request.model, prompt, n=chat_request.n or 1
            )

            timestamp = int(time.time())
            msg_hash, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
                request_bytes, "", timestamp
            )
            tee_signature = tee_keys.sign_data(msg_hash)

            final_data: dict[str, Any] = {
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "model": chat_request.model,
                "tee_signature": tee_signature,
                "tee_timestamp": timestamp,
                "tee_request_hash": input_hash_hex,
                "tee_output_hash": output_hash_hex,
                "tee_id": f"0x{tee_keys.get_tee_id()}",
            }
            # Images travel out-of-band on the final frame; not part of the hash.
            if images:
                final_data["images"] = images

            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            final_data["usage"] = usage
            cost = compute_session_cost(
                chat_request.model, usage, image_count=image_count
            )
            if cost is not None:
                final_data["opengradient"] = cost.model_dump(mode="json")

            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Image generation streaming error: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'error': 'Stream processing failed', 'exception_type': type(e).__name__})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


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
            return _create_image_generation_response(chat_request, request_bytes)

        model = get_chat_model_cached(
            model=chat_request.model,
            temperature=float(chat_request.temperature)
            if chat_request.temperature is not None
            else 0.0,
            max_tokens=chat_request.max_tokens or 4096,
            web_search=bool(chat_request.web_search),
        )

        # Bind user tools and/or the native web search tool if requested.
        tools_list = _build_tools_list(chat_request, provider)
        if tools_list:
            model = model.bind_tools(tools_list)

        # Bind response_format if provided (json_object or json_schema).
        # Anthropic does not support response_format via bind(); use
        # with_structured_output() for json_schema instead (json_object has no
        # Anthropic native equivalent and raises a clear error).
        rf_dict: dict | None = None
        if chat_request.response_format:
            rf = _normalize_response_format(chat_request.response_format)
            if rf.get("type", "text") != "text":
                rf_dict = rf
                if provider != "anthropic":
                    model = model.bind(response_format=rf_dict)

        langchain_messages = convert_messages(chat_request.messages)

        # OpenAI (and compatible providers) require the word "json" to appear
        # somewhere in the messages when response_format.type == "json_object".
        # Inject a brief system instruction if none of the messages satisfy this.
        if rf_dict and rf_dict.get("type") == "json_object":
            if not _messages_contain_json_word(langchain_messages):
                langchain_messages = [
                    SystemMessage(content="Respond in JSON format.")
                ] + langchain_messages

        if rf_dict and provider == "anthropic":
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

        finish_reason = "stop"
        if hasattr(response, "tool_calls") and response.tool_calls:
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
                for tc in response.tool_calls
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
        usage = extract_usage(response)
        if usage:
            # Surface the standard OpenAI usage triple on the response; the
            # reasoning split rides along to the cost calculator via `usage`.
            openai_response["usage"] = {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            }
            web_search_count = (
                extract_web_search_count(response) if chat_request.web_search else 0
            )
            cost = compute_session_cost(
                chat_request.model, usage, web_search_count=web_search_count
            )
            if cost is not None:
                openai_response["opengradient"] = cost.model_dump(mode="json")

        # Validate schema (the extra tee_* fields are preserved by returning dict directly)
        CreateChatCompletionResponse.from_dict(openai_response)
        return openai_response

    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}", exc_info=True)
        return {
            "error": "Request processing failed",
            "exception_type": type(e).__name__,
        }, 500


def _create_streaming_response(chat_request: CreateChatCompletionRequest):
    """Handle streaming chat completion via direct LangChain call."""
    try:
        provider = get_provider_from_model(chat_request.model)
        # OpenAI and Anthropic stream tool calls as fragments that must be
        # buffered and flushed once complete. Gemini emits complete tool calls.
        buffer_tool_calls = provider in ["openai", "anthropic"]
        # Gemini inline-image models return a single image rather than a token
        # stream — invoke once and emit the result inside the SSE envelope.
        image_output_model = get_model_config(chat_request.model).image_output

        request_dict = _chat_request_to_dict(chat_request)
        request_bytes = json.dumps(request_dict, sort_keys=True).encode("utf-8")

        # Image-generation models (xAI Grok, ByteDance Seedream) use a dedicated
        # images endpoint; handle them without building a chat model.
        if get_model_config(chat_request.model).image_generation:
            return _create_image_generation_streaming_response(
                chat_request, request_bytes
            )

        model = get_chat_model_cached(
            model=chat_request.model,
            temperature=float(chat_request.temperature)
            if chat_request.temperature is not None
            else 0.0,
            max_tokens=chat_request.max_tokens or 4096,
            web_search=bool(chat_request.web_search),
        )

        # Bind user tools and/or the native web search tool if requested.
        tools_list = _build_tools_list(chat_request, provider)
        if tools_list:
            model = model.bind_tools(tools_list)

        # Bind response_format if provided (json_object or json_schema).
        # Anthropic does not support response_format via bind(); use
        # with_structured_output() for json_schema instead (json_object has no
        # Anthropic native equivalent and raises a clear error).
        anthropic_structured_rf: dict | None = None
        if chat_request.response_format:
            rf = _normalize_response_format(chat_request.response_format)
            if rf.get("type", "text") != "text":
                if provider == "anthropic":
                    anthropic_structured_rf = rf
                else:
                    model = model.bind(response_format=rf)

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

        def generate():
            full_content = ""
            final_usage = None
            buffered_tool_calls = {}
            finish_reason = "stop"
            generated_images: list[str] = []
            # Accumulate streamed chunks so native web-search activity (content
            # blocks, citations, grounding metadata) can be counted for billing
            # once the stream completes.
            merged_chunk = None

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
                    chunks_iter = model.stream(langchain_messages)  # type: ignore[assignment]

                for chunk in chunks_iter:
                    # Accumulate for post-stream web-search billing (cheap: merges
                    # content/metadata deltas into a single AIMessageChunk).
                    if chat_request.web_search:
                        merged_chunk = (
                            chunk if merged_chunk is None else merged_chunk + chunk
                        )

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
                    web_search_count = (
                        extract_web_search_count(merged_chunk)
                        if chat_request.web_search
                        else 0
                    )
                    # Pass thinking tokens to the cost calculator (for the image
                    # dual-rate split) without polluting the OpenAI usage triple.
                    cost_usage = dict(
                        final_data["usage"],
                        reasoning_tokens=final_usage.get("reasoning", 0),
                    )
                    cost = compute_session_cost(
                        chat_request.model,
                        cost_usage,
                        web_search_count=web_search_count,
                    )
                    if cost is not None:
                        # final_data is hand-serialized to SSE via json.dumps below,
                        # which doesn't go through Flask's JSONEncoder — so do the
                        # serialization ourselves here.
                        final_data["opengradient"] = cost.model_dump(mode="json")
                    logger.info(
                        f"Stream completed — usage: {final_data['usage']}, "
                        f"finish: {finish_reason}, "
                        f"inputHash: {input_hash_hex[:16]}..., outputHash: {output_hash_hex[:16]}..."
                    )

                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'error': 'Stream processing failed', 'exception_type': type(e).__name__})}\n\n"

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
            "error": "Stream setup failed",
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
                    "content": msg.content,
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
