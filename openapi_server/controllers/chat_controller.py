import os
import time
import json

import connexion
from typing import List
from flask import Response
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import urllib3

from openapi_server.models.create_chat_completion_request import CreateChatCompletionRequest
from openapi_server.models.create_chat_completion_response import CreateChatCompletionResponse
from openapi_server.models import (
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestFunctionMessage,
)

import uuid
from openapi_server.controllers.defaults import HTTP_BACKEND_SERVER

# Create a session with retry strategy for HTTP requests
http_session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http_session.mount("http://", adapter)
http_session.mount("https://", adapter)

def create_chat_completion(body):
    """
    Create a chat completion (streaming or non-streaming based on stream parameter).
    """
    if not connexion.request.is_json:
        return {
            'error': 'Unsupported Media Type',
            'message': 'Request must be application/json'
        }, 415

    chat_request: CreateChatCompletionRequest = parse_chat_request(connexion.request.get_json())

    # Check if streaming is requested
    if chat_request.stream:
        return _create_streaming_response(chat_request)
    else:
        return _create_non_streaming_response(chat_request)


def _create_non_streaming_response(chat_request: CreateChatCompletionRequest):
    """Handle non-streaming HTTP-based chat completion"""
    
    # Convert OpenAI format to backend format
    backend_messages = []
    for msg in chat_request.messages:
        if isinstance(msg, ChatCompletionRequestSystemMessage):
            backend_messages.append({
                "role": "system",
                "content": msg.content
            })
        elif isinstance(msg, ChatCompletionRequestUserMessage):
            if isinstance(msg.content, str):
                backend_messages.append({
                    "role": "user",
                    "content": msg.content
                })
        elif isinstance(msg, ChatCompletionRequestAssistantMessage):
            msg_dict = {
                "role": "assistant",
                "content": msg.content or ""
            }
            if msg.tool_calls:
                tool_calls_list = []
                for tc in msg.tool_calls:
                    tool_calls_list.append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                msg_dict["tool_calls"] = tool_calls_list
            backend_messages.append(msg_dict)
        elif isinstance(msg, ChatCompletionRequestToolMessage):
            backend_messages.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id
            })
        elif isinstance(msg, ChatCompletionRequestFunctionMessage):
            backend_messages.append({
                "role": "function",
                "content": msg.content,
                "name": msg.name
            })

    # Convert tools to backend format
    backend_tools = None
    if chat_request.tools:
        backend_tools = []
        for tool in chat_request.tools:
            func = tool.get('function', {})
            backend_tool = {
                "type": tool.get('type', 'function'),
                "function": {
                    "name": func.get('name'),
                    "description": func.get('description'),
                    "parameters": func.get('parameters')
                }
            }
            if func.get('strict') is not None:
                backend_tool["function"]["strict"] = func.get('strict')
            backend_tools.append(backend_tool)

    # Convert tool_choice to string format
    backend_tool_choice = None
    if chat_request.tool_choice:
        if isinstance(chat_request.tool_choice, str):
            backend_tool_choice = chat_request.tool_choice
        elif hasattr(chat_request.tool_choice, 'type'):
            backend_tool_choice = chat_request.tool_choice.type

    # Prepare backend request
    backend_request = {
        "model": chat_request.model,
        "messages": backend_messages,
        "temperature": chat_request.temperature if chat_request.temperature is not None else 0.0,
    }
    
    if chat_request.max_tokens is not None:
        backend_request["max_tokens"] = chat_request.max_tokens
    
    if chat_request.stop:
        backend_request["stop"] = chat_request.stop
    
    if backend_tools:
        backend_request["tools"] = backend_tools
    
    if backend_tool_choice:
        backend_request["tool_choice"] = backend_tool_choice
    
    # Capture payment header while in request context
    payment_header = connexion.request.headers.get('X-PAYMENT')

    try:
        # Make HTTP request to TEE server
        backend_url = f"{HTTP_BACKEND_SERVER}/v1/chat/completions"
        
        headers = {'Content-Type': 'application/json'}
        if payment_header:
            headers['X-PAYMENT'] = payment_header
        
        response = http_session.post(
            backend_url,
            json=backend_request,
            timeout=120,
            verify=False,
            headers=headers
        )
        
        response.raise_for_status()
        
        # Parse response from TEE server
        tee_response = response.json()
        
        # Convert TEE response to OpenAI format
        # TEE server returns: {finish_reason, message, model, usage, timestamp, signature, request_hash}
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": tee_response.get("model", chat_request.model),
            "choices": [{
                "index": 0,
                "message": tee_response.get("message", {}),
                "finish_reason": tee_response.get("finish_reason", "stop")
            }]
        }
        
        # Add usage if available
        if "usage" in tee_response and tee_response["usage"]:
            openai_response["usage"] = tee_response["usage"]
        
        # Add TEE-specific metadata if present
        if "signature" in tee_response:
            openai_response["tee_signature"] = tee_response["signature"]
        if "request_hash" in tee_response:
            openai_response["tee_request_hash"] = tee_response["request_hash"]
        if "output_hash" in tee_response:
            openai_response["tee_output_hash"] = tee_response["output_hash"]
        if "timestamp" in tee_response:
            openai_response["tee_timestamp"] = tee_response["timestamp"]
        
        # Validate with OpenAPI model but return the dict directly to preserve TEE metadata
        CreateChatCompletionResponse.from_dict(openai_response)
        return openai_response
        
    except requests.exceptions.RequestException as e:
        return {"error": "Backend request failed", "details": str(e)}, 500
    except Exception as e:
        return {"error": "Request processing failed", "details": str(e)}, 500


def _create_streaming_response(chat_request: CreateChatCompletionRequest):
    """Handle streaming HTTP-based chat completion"""
    
    # Convert OpenAI format to backend format
    backend_messages = []
    for msg in chat_request.messages:
        if isinstance(msg, ChatCompletionRequestSystemMessage):
            backend_messages.append({
                "role": "system",
                "content": msg.content
            })
        elif isinstance(msg, ChatCompletionRequestUserMessage):
            if isinstance(msg.content, str):
                backend_messages.append({
                    "role": "user",
                    "content": msg.content
                })
        elif isinstance(msg, ChatCompletionRequestAssistantMessage):
            msg_dict = {
                "role": "assistant",
                "content": msg.content or ""
            }
            if msg.tool_calls:
                tool_calls_list = []
                for tc in msg.tool_calls:
                    tool_calls_list.append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                msg_dict["tool_calls"] = tool_calls_list
            backend_messages.append(msg_dict)
        elif isinstance(msg, ChatCompletionRequestToolMessage):
            backend_messages.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id
            })
        elif isinstance(msg, ChatCompletionRequestFunctionMessage):
            backend_messages.append({
                "role": "function",
                "content": msg.content,
                "name": msg.name
            })

    # Convert tools to backend format
    backend_tools = None
    if chat_request.tools:
        backend_tools = []
        for tool in chat_request.tools:
            func = tool.get('function', {})
            backend_tool = {
                "type": tool.get('type', 'function'),
                "function": {
                    "name": func.get('name'),
                    "description": func.get('description'),
                    "parameters": func.get('parameters')
                }
            }
            backend_tools.append(backend_tool)

    # Convert tool_choice to string format
    backend_tool_choice = None
    if chat_request.tool_choice:
        if isinstance(chat_request.tool_choice, str):
            backend_tool_choice = chat_request.tool_choice
        elif hasattr(chat_request.tool_choice, 'type'):
            backend_tool_choice = chat_request.tool_choice.type

    # Prepare backend request
    backend_request = {
        "model": chat_request.model,
        "messages": backend_messages,
        "temperature": chat_request.temperature or 0.0,
        "max_tokens": chat_request.max_tokens,
        "stop": chat_request.stop if chat_request.stop else None,
    }
    
    if backend_tools:
        backend_request["tools"] = backend_tools
    if backend_tool_choice:
        backend_request["tool_choice"] = backend_tool_choice
    
    # Capture payment header while in request context
    payment_header = connexion.request.headers.get('X-PAYMENT')

    def generate():
        """Generate SSE stream from TEE server"""
        buffer = b""
        
        try:
            backend_url = f"{HTTP_BACKEND_SERVER}/v1/chat/completions/stream"
            
            with http_session.post(
                backend_url,
                json=backend_request,
                stream=True,
                timeout=None,
                verify=False,
                headers={'X-PAYMENT': payment_header} if payment_header else {}
            ) as response:
                
                response.raise_for_status()
                
                # Process raw bytes from TEE server
                for raw_chunk in response.iter_content(chunk_size=1, decode_unicode=False):
                    if not raw_chunk:
                        continue
                    
                    buffer += raw_chunk
                    
                    # Process complete SSE messages (data: ... \n\n)
                    while b"\n\n" in buffer:
                        message, buffer = buffer.split(b"\n\n", 1)
                        message_str = message.decode('utf-8', errors='ignore').strip()
                        
                        if not message_str:
                            continue
                        
                        # Check if it's a data line
                        if message_str.startswith("data: "):
                            data_str = message_str[6:].strip()
                            
                            # Handle [DONE] signal
                            if data_str == "[DONE]":
                                yield b"data: [DONE]\n\n"
                                return
                            
                            yield f"data: {data_str}\n\n".encode('utf-8')
                
                # Send final [DONE] if not received
                yield b"data: [DONE]\n\n"
                        
        except requests.exceptions.RequestException as e:
            error_data = {"error": "Backend request failed", "details": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
        except Exception as e:
            error_data = {"error": "Streaming error", "details": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        }
    )


def parse_chat_request(chat_request):
    
    # Parse each message individually
    if 'messages' in chat_request:
        messages = [
            parse_message(msg) for msg in chat_request['messages']
        ]
    
    return CreateChatCompletionRequest(
        messages=messages,
        model=chat_request.get('model'),
        frequency_penalty=chat_request.get('frequency_penalty'),
        logit_bias=chat_request.get('logit_bias'),
        max_tokens=chat_request.get('max_tokens'),
        n=chat_request.get('n'),
        presence_penalty=chat_request.get('presence_penalty'),
        response_format=chat_request.get('response_format'),
        seed=chat_request.get('seed'),
        stop=chat_request.get('stop'),
        stream=chat_request.get('stream'),
        temperature=chat_request.get('temperature'),
        top_p=chat_request.get('top_p'),
        tools=chat_request.get('tools'),
        tool_choice=chat_request.get('tool_choice'),
        user=chat_request.get('user')
    )

def parse_message(message_dict):
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