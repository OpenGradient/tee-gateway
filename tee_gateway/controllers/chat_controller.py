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

from langchain_core.messages import AIMessage

from tee_gateway.tee_manager import get_tee_keys, compute_tee_msg_hash
from tee_gateway.llm_backend import (
    get_provider_from_model,
    get_chat_model_cached,
    convert_messages,
    extract_usage,
)

logger = logging.getLogger(__name__)


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
        schema_def, method="json_schema", strict=strict
    )
    result = structured.invoke(langchain_messages)

    content_str = json.dumps(result) if isinstance(result, dict) else str(result)
    return AIMessage(content=content_str)


def _create_non_streaming_response(chat_request: CreateChatCompletionRequest):
    """Handle non-streaming chat completion via direct LangChain call."""
    try:
        logger.info("=" * 80)
        logger.info(f"Chat request for model: {chat_request.model}")
        logger.info(f"Number of messages: {len(chat_request.messages)}")

        # Serialize request for hashing (canonical, deterministic)
        request_dict = _chat_request_to_dict(chat_request)
        request_bytes = json.dumps(request_dict, sort_keys=True).encode("utf-8")

        model = get_chat_model_cached(
            model=chat_request.model,
            temperature=float(chat_request.temperature)
            if chat_request.temperature is not None
            else 0.0,
            max_tokens=chat_request.max_tokens or 4096,
        )

        # Bind tools if provided
        if chat_request.tools:
            tools_list = []
            for tool in chat_request.tools:
                if isinstance(tool, dict):
                    func = tool.get("function", {})
                    tools_list.append(
                        {"type": tool.get("type", "function"), "function": func}
                    )
                else:
                    tools_list.append(tool)
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
                if get_provider_from_model(chat_request.model) != "anthropic":
                    model = model.bind(response_format=rf_dict)

        langchain_messages = convert_messages(chat_request.messages)
        if rf_dict and get_provider_from_model(chat_request.model) == "anthropic":
            response = _invoke_anthropic_structured(model, rf_dict, langchain_messages)
        else:
            response = model.invoke(langchain_messages)

        # Normalize content (Gemini may return a list of content parts)
        if isinstance(response.content, list):
            content_str = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in response.content
            )
        else:
            content_str = response.content or ""

        message_dict: dict[str, Any] = {
            "role": "assistant",
            "content": content_str,
        }

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

        usage = extract_usage(response)

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

        if usage:
            openai_response["usage"] = usage

        # Validate schema (the extra tee_* fields are preserved by returning dict directly)
        CreateChatCompletionResponse.from_dict(openai_response)
        return openai_response

    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}", exc_info=True)
        return {"error": "Request processing failed"}, 500


def _create_streaming_response(chat_request: CreateChatCompletionRequest):
    """Handle streaming chat completion via direct LangChain call."""
    try:
        provider = get_provider_from_model(chat_request.model)
        # OpenAI and Anthropic stream tool calls as fragments that must be
        # buffered and flushed once complete. Gemini emits complete tool calls.
        buffer_tool_calls = provider in ["openai", "anthropic"]

        request_dict = _chat_request_to_dict(chat_request)
        request_bytes = json.dumps(request_dict, sort_keys=True).encode("utf-8")

        model = get_chat_model_cached(
            model=chat_request.model,
            temperature=float(chat_request.temperature)
            if chat_request.temperature is not None
            else 0.0,
            max_tokens=chat_request.max_tokens or 4096,
        )

        if chat_request.tools:
            tools_list = []
            for tool in chat_request.tools:
                if isinstance(tool, dict):
                    func = tool.get("function", {})
                    tools_list.append(
                        {"type": tool.get("type", "function"), "function": func}
                    )
                else:
                    tools_list.append(tool)
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
        else:
            anthropic_structured_content = None

        def generate():
            full_content = ""
            final_usage = None
            buffered_tool_calls = {}
            finish_reason = "stop"

            try:
                if anthropic_structured_content is not None:
                    # Emit the pre-computed structured result as a single chunk
                    full_content = anthropic_structured_content
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
                    chunks_iter: list = []
                else:
                    chunks_iter = model.stream(langchain_messages)  # type: ignore[assignment]

                for chunk in chunks_iter:
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
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        final_usage = chunk.usage_metadata

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

                logger.debug(
                    f"Response Final\n\tTEE Signature: {tee_signature}\n\tTEE request hash: {input_hash_hex}\n\tTEE output hash: {output_hash_hex}\n\tTEE timestamp: {timestamp}\n\tTEE ID: 0x{tee_keys.get_tee_id()}"
                )

                if final_usage:
                    final_data["usage"] = {
                        "prompt_tokens": final_usage.get("input_tokens", 0),
                        "completion_tokens": final_usage.get("output_tokens", 0),
                        "total_tokens": final_usage.get("total_tokens", 0),
                    }
                    logger.info(
                        f"Stream completed — usage: {final_data['usage']}, "
                        f"finish: {finish_reason}, "
                        f"inputHash: {input_hash_hex[:16]}..., outputHash: {output_hash_hex[:16]}..."
                    )

                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'error': 'Stream processing failed'})}\n\n"

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
        return {"error": "Stream setup failed"}, 500


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
                    "content": msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content),
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
