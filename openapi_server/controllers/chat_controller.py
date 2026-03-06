import json
import time
import uuid
import logging

import connexion
from flask import Response

from openapi_server.models.create_chat_completion_request import CreateChatCompletionRequest
from openapi_server.models.create_chat_completion_response import CreateChatCompletionResponse
from openapi_server.models import (
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestFunctionMessage,
)

from openapi_server.tee_manager import get_tee_keys, compute_tee_msg_hash
from openapi_server.llm_backend import (
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
            'error': 'Unsupported Media Type',
            'message': 'Request must be application/json'
        }, 415

    chat_request: CreateChatCompletionRequest = _parse_chat_request(connexion.request.get_json())

    if chat_request.stream:
        return _create_streaming_response(chat_request)
    else:
        return _create_non_streaming_response(chat_request)


def _create_non_streaming_response(chat_request: CreateChatCompletionRequest):
    """Handle non-streaming chat completion via direct LangChain call."""
    try:
        logger.info("=" * 80)
        logger.info(f"Chat request for model: {chat_request.model}")
        logger.info(f"Number of messages: {len(chat_request.messages)}")

        # Serialize request for hashing (canonical, deterministic)
        request_dict = _chat_request_to_dict(chat_request)
        request_bytes = json.dumps(request_dict, sort_keys=True).encode('utf-8')

        model = get_chat_model_cached(
            model=chat_request.model,
            temperature=float(chat_request.temperature) if chat_request.temperature is not None else 0.0,
            max_tokens=chat_request.max_tokens or 4096,
        )

        # Bind tools if provided
        if chat_request.tools:
            tools_list = []
            for tool in chat_request.tools:
                if isinstance(tool, dict):
                    func = tool.get('function', {})
                    tools_list.append({"type": tool.get('type', 'function'), "function": func})
                else:
                    tools_list.append(tool)
            model = model.bind_tools(tools_list)

        langchain_messages = convert_messages(chat_request.messages)
        response = model.invoke(langchain_messages)

        # Normalize content (Gemini may return a list of content parts)
        if isinstance(response.content, list):
            content_str = ''.join(
                item.get('text', '') if isinstance(item, dict) else str(item)
                for item in response.content
            )
        else:
            content_str = response.content or ""

        message_dict = {
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
                    }
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
            "choices": [{
                "index": 0,
                "message": message_dict,
                "finish_reason": finish_reason,
            }],
            "tee_signature": signature,
            "tee_request_hash": input_hash_hex,
            "tee_output_hash": output_hash_hex,
            "tee_timestamp": timestamp,
            "tee_id": f"0x{tee_keys.get_tee_id()}",
        }

        if usage:
            openai_response["usage"] = usage

        # Validate schema (the extra tee_* fields are preserved by returning dict directly)
        CreateChatCompletionResponse.from_dict(openai_response)
        return openai_response

    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}", exc_info=True)
        return {"error": "Request processing failed", "details": str(e)}, 500


def _create_streaming_response(chat_request: CreateChatCompletionRequest):
    """Handle streaming chat completion via direct LangChain call."""
    try:
        provider = get_provider_from_model(chat_request.model)
        # OpenAI and Anthropic stream tool calls as fragments that must be
        # buffered and flushed once complete. Gemini emits complete tool calls.
        buffer_tool_calls = provider in ["openai", "anthropic"]

        request_dict = _chat_request_to_dict(chat_request)
        request_bytes = json.dumps(request_dict, sort_keys=True).encode('utf-8')

        model = get_chat_model_cached(
            model=chat_request.model,
            temperature=float(chat_request.temperature) if chat_request.temperature is not None else 0.0,
            max_tokens=chat_request.max_tokens or 4096,
        )

        if chat_request.tools:
            tools_list = []
            for tool in chat_request.tools:
                if isinstance(tool, dict):
                    func = tool.get('function', {})
                    tools_list.append({"type": tool.get('type', 'function'), "function": func})
                else:
                    tools_list.append(tool)
            model = model.bind_tools(tools_list)

        langchain_messages = convert_messages(chat_request.messages)
        tee_keys = get_tee_keys()

        def generate():
            full_content = ""
            final_usage = None
            buffered_tool_calls = {}
            finish_reason = "stop"

            try:
                for chunk in model.stream(langchain_messages):
                    # --- Text content ---
                    if chunk.content:
                        if isinstance(chunk.content, str):
                            content_str = chunk.content
                        elif isinstance(chunk.content, list):
                            content_str = ''.join(
                                item.get('text', '') if isinstance(item, dict) else str(item)
                                for item in chunk.content
                            )
                        else:
                            content_str = str(chunk.content)

                        if content_str:
                            full_content += content_str
                            data = {
                                "choices": [{
                                    "delta": {"content": content_str, "role": "assistant"},
                                    "index": 0,
                                    "finish_reason": None
                                }],
                                "model": chat_request.model
                            }
                            yield f"data: {json.dumps(data)}\n\n"

                    # --- Tool call chunks ---
                    if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                        finish_reason = "tool_calls"
                        for tc_chunk in chunk.tool_call_chunks:
                            tc_index = tc_chunk.get('index', 0)

                            if tc_index not in buffered_tool_calls:
                                buffered_tool_calls[tc_index] = {
                                    'id': tc_chunk.get('id', ''),
                                    'type': 'function',
                                    'function': {'name': tc_chunk.get('name', ''), 'arguments': ''}
                                }

                            if tc_chunk.get('id'):
                                buffered_tool_calls[tc_index]['id'] = tc_chunk['id']
                            if tc_chunk.get('name'):
                                buffered_tool_calls[tc_index]['function']['name'] = tc_chunk['name']
                            if tc_chunk.get('args'):
                                args_value = tc_chunk['args']
                                if isinstance(args_value, dict):
                                    args_str = json.dumps(args_value)
                                elif isinstance(args_value, str):
                                    args_str = args_value
                                else:
                                    args_str = str(args_value)
                                buffered_tool_calls[tc_index]['function']['arguments'] += args_str

                            # For providers that don't need buffering, emit each fragment immediately
                            if not buffer_tool_calls:
                                delta = {"role": "assistant", "tool_calls": [{"index": tc_index, "type": "function", "function": {}}]}
                                if tc_chunk.get('id'):
                                    delta["tool_calls"][0]["id"] = tc_chunk['id']
                                if tc_chunk.get('name'):
                                    delta["tool_calls"][0]["function"]["name"] = tc_chunk['name']
                                if tc_chunk.get('args'):
                                    args_value = tc_chunk['args']
                                    if isinstance(args_value, dict):
                                        delta["tool_calls"][0]["function"]["arguments"] = json.dumps(args_value)
                                    elif isinstance(args_value, str):
                                        delta["tool_calls"][0]["function"]["arguments"] = args_value
                                    else:
                                        delta["tool_calls"][0]["function"]["arguments"] = str(args_value)

                                    if not delta["tool_calls"][0]["function"]:
                                        del delta["tool_calls"][0]["function"]

                                data = {
                                    "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
                                    "model": chat_request.model
                                }
                                yield f"data: {json.dumps(data)}\n\n"

                    # --- Usage metadata ---
                    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                        final_usage = chunk.usage_metadata

                # Flush buffered tool calls for OpenAI/Anthropic
                if buffer_tool_calls and buffered_tool_calls:
                    for tc_index, tc in buffered_tool_calls.items():
                        delta = {
                            "role": "assistant",
                            "tool_calls": [{
                                "index": tc_index,
                                "id": tc['id'],
                                "type": "function",
                                "function": {
                                    "name": tc['function']['name'],
                                    "arguments": tc['function']['arguments']
                                }
                            }]
                        }
                        data = {
                            "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
                            "model": chat_request.model
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                # Sign the completed response.
                # For tool-call responses, hash the buffered tool calls so the
                # signature covers the actual invocations.
                timestamp = int(time.time())
                if finish_reason == "tool_calls" and buffered_tool_calls:
                    tool_calls_list = [buffered_tool_calls[k] for k in sorted(buffered_tool_calls.keys())]
                    output_content = json.dumps(tool_calls_list, sort_keys=True)
                else:
                    output_content = full_content

                msg_hash, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
                    request_bytes, output_content, timestamp
                )
                tee_signature = tee_keys.sign_data(msg_hash)

                final_data = {
                    "choices": [{"delta": {}, "index": 0, "finish_reason": finish_reason}],
                    "model": chat_request.model,
                    "tee_signature": tee_signature,
                    "tee_timestamp": timestamp,
                    "tee_request_hash": input_hash_hex,
                    "tee_output_hash": output_hash_hex,
                    "tee_id": f"0x{tee_keys.get_tee_id()}",
                }

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
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )

    except Exception as e:
        logger.error(f"Stream setup error: {str(e)}", exc_info=True)
        return {"error": "Stream setup failed", "details": str(e)}, 500


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
            messages.append({"role": "user", "content": msg.content if isinstance(msg.content, str) else str(msg.content)})
        elif isinstance(msg, ChatCompletionRequestAssistantMessage):
            m = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                m["tool_calls"] = [
                    {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            messages.append(m)
        elif isinstance(msg, ChatCompletionRequestToolMessage):
            messages.append({"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id})
        elif isinstance(msg, ChatCompletionRequestFunctionMessage):
            messages.append({"role": "function", "content": msg.content, "name": msg.name})

    d = {
        "model": chat_request.model,
        "messages": messages,
        "temperature": float(chat_request.temperature) if chat_request.temperature is not None else 0.0,
    }
    if chat_request.max_tokens is not None:
        d["max_tokens"] = chat_request.max_tokens
    if chat_request.stop:
        d["stop"] = chat_request.stop
    if chat_request.tools:
        d["tools"] = chat_request.tools if isinstance(chat_request.tools, list) else list(chat_request.tools)
    return d


def _parse_chat_request(chat_request_dict: dict) -> CreateChatCompletionRequest:
    messages = [_parse_message(msg) for msg in chat_request_dict.get('messages', [])]
    return CreateChatCompletionRequest(
        messages=messages,
        model=chat_request_dict.get('model'),
        frequency_penalty=chat_request_dict.get('frequency_penalty'),
        logit_bias=chat_request_dict.get('logit_bias'),
        max_tokens=chat_request_dict.get('max_tokens'),
        n=chat_request_dict.get('n'),
        presence_penalty=chat_request_dict.get('presence_penalty'),
        response_format=chat_request_dict.get('response_format'),
        seed=chat_request_dict.get('seed'),
        stop=chat_request_dict.get('stop'),
        stream=chat_request_dict.get('stream'),
        temperature=chat_request_dict.get('temperature'),
        top_p=chat_request_dict.get('top_p'),
        tools=chat_request_dict.get('tools'),
        tool_choice=chat_request_dict.get('tool_choice'),
        user=chat_request_dict.get('user'),
    )


def _parse_message(message_dict: dict):
    role = message_dict.get('role')
    if role == 'user':
        return ChatCompletionRequestUserMessage.from_dict(message_dict)
    elif role == 'system':
        return ChatCompletionRequestSystemMessage.from_dict(message_dict)
    elif role == 'assistant':
        return ChatCompletionRequestAssistantMessage.from_dict(message_dict)
    elif role == 'tool':
        return ChatCompletionRequestToolMessage.from_dict(message_dict)
    elif role == 'function':
        return ChatCompletionRequestFunctionMessage.from_dict(message_dict)
    else:
        raise ValueError(f"Unknown role: {role}")
