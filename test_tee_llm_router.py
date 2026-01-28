#!/usr/bin/env python3
"""
Unit tests for TEE LLM Router Server
Tests completion, chat, tool calls, and streaming endpoints for each provider.
"""

import pytest
import asyncio
import json
import os
import sys
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Verify API keys are loaded
required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY"]
missing_keys = [key for key in required_keys if not os.getenv(key)]
if missing_keys:
    raise ValueError(f"Missing required API keys in .env: {', '.join(missing_keys)}")

from fastapi.testclient import TestClient

# Mock the TEE key manager registration before importing the app
with patch('urllib.request.urlopen'):
    from server import (
        app, 
        get_chat_model_cached,
        convert_messages,
        Message,
        CompletionRequest,
        ChatRequest,
        Tool
    )

client = TestClient(app)


class MockAIMessage:
    """Mock LangChain AIMessage for testing"""
    def __init__(self, content: str, tool_calls: List[Dict] = None, usage_metadata: Dict = None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = usage_metadata or {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30
        }


class MockStreamChunk:
    """Mock streaming chunk for testing"""
    def __init__(self, content: str = "", tool_call_chunks: List[Dict] = None, usage_metadata: Dict = None):
        self.content = content
        self.tool_call_chunks = tool_call_chunks or []
        self.usage_metadata = usage_metadata


# Test Fixtures
@pytest.fixture
def mock_openai_model():
    """Mock OpenAI chat model"""
    mock_model = MagicMock()
    mock_model.invoke = MagicMock(return_value=MockAIMessage(
        content="Hello from OpenAI!",
        usage_metadata={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}
    ))
    
    async def mock_astream(messages):
        """Mock async streaming"""
        # Yield content chunks
        yield MockStreamChunk(content="Hello ")
        yield MockStreamChunk(content="from ")
        yield MockStreamChunk(content="OpenAI!")
        # Yield final usage
        yield MockStreamChunk(usage_metadata={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15})
    
    mock_model.astream = mock_astream
    mock_model.bind_tools = MagicMock(return_value=mock_model)
    return mock_model


@pytest.fixture
def mock_anthropic_model():
    """Mock Anthropic chat model"""
    mock_model = MagicMock()
    mock_model.invoke = MagicMock(return_value=MockAIMessage(
        content="Hello from Anthropic!",
        usage_metadata={"input_tokens": 8, "output_tokens": 12, "total_tokens": 20}
    ))
    
    async def mock_astream(messages):
        yield MockStreamChunk(content="Hello ")
        yield MockStreamChunk(content="from ")
        yield MockStreamChunk(content="Anthropic!")
        yield MockStreamChunk(usage_metadata={"input_tokens": 8, "output_tokens": 12, "total_tokens": 20})
    
    mock_model.astream = mock_astream
    mock_model.bind_tools = MagicMock(return_value=mock_model)
    return mock_model


@pytest.fixture
def mock_google_model():
    """Mock Google chat model"""
    mock_model = MagicMock()
    mock_model.invoke = MagicMock(return_value=MockAIMessage(
        content="Hello from Google!",
        usage_metadata={"input_tokens": 6, "output_tokens": 11, "total_tokens": 17}
    ))
    
    async def mock_astream(messages):
        yield MockStreamChunk(content="Hello ")
        yield MockStreamChunk(content="from ")
        yield MockStreamChunk(content="Google!")
        yield MockStreamChunk(usage_metadata={"input_tokens": 6, "output_tokens": 11, "total_tokens": 17})
    
    mock_model.astream = mock_astream
    mock_model.bind_tools = MagicMock(return_value=mock_model)
    return mock_model


@pytest.fixture
def mock_xai_model():
    """Mock xAI chat model"""
    mock_model = MagicMock()
    mock_model.invoke = MagicMock(return_value=MockAIMessage(
        content="Hello from xAI!",
        usage_metadata={"input_tokens": 7, "output_tokens": 13, "total_tokens": 20}
    ))
    
    async def mock_astream(messages):
        yield MockStreamChunk(content="Hello ")
        yield MockStreamChunk(content="from ")
        yield MockStreamChunk(content="xAI!")
        yield MockStreamChunk(usage_metadata={"input_tokens": 7, "output_tokens": 13, "total_tokens": 20})
    
    mock_model.astream = mock_astream
    mock_model.bind_tools = MagicMock(return_value=mock_model)
    return mock_model


@pytest.fixture
def mock_tool_call_model():
    """Mock model that returns tool calls"""
    mock_model = MagicMock()
    mock_model.invoke = MagicMock(return_value=MockAIMessage(
        content="",
        tool_calls=[{
            "id": "call_123",
            "name": "get_weather",
            "args": {"location": "San Francisco", "unit": "celsius"},
            "type": "function"
        }],
        usage_metadata={"input_tokens": 15, "output_tokens": 25, "total_tokens": 40}
    ))
    
    async def mock_astream_tools(messages):
        # Stream tool call chunks
        yield MockStreamChunk(tool_call_chunks=[{
            "index": 0,
            "id": "call_123",
            "name": "get_weather",
            "args": '{"location": "San Francisco", "unit": "celsius"}',
            "type": "function"
        }])
        # Final usage
        yield MockStreamChunk(usage_metadata={"input_tokens": 15, "output_tokens": 25, "total_tokens": 40})
    
    mock_model.astream = mock_astream_tools
    mock_model.bind_tools = MagicMock(return_value=mock_model)
    return mock_model


# Health Check Test
def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["tee_enabled"] is True
    assert "memory_mb" in data
    assert "uptime_seconds" in data


# Attestation Test
def test_attestation():
    """Test TEE attestation endpoint"""
    response = client.get("/attestation")
    assert response.status_code == 200
    data = response.json()
    assert "public_key" in data
    assert "timestamp" in data
    assert "enclave_info" in data
    assert data["enclave_info"]["platform"] == "aws-nitro"


# Models List Test
def test_list_models():
    """Test models listing endpoint"""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0
    
    # Check for models from each provider
    model_ids = [m["id"] for m in data["data"]]
    assert any("gpt" in m for m in model_ids)  # OpenAI
    assert any("claude" in m for m in model_ids)  # Anthropic
    assert any("gemini" in m for m in model_ids)  # Google
    assert any("grok" in m for m in model_ids)  # xAI


# Completion Tests
@patch('server.get_chat_model_cached')
def test_completion_openai(mock_get_model, mock_openai_model):
    """Test OpenAI completion endpoint"""
    mock_get_model.return_value = mock_openai_model
    
    response = client.post("/v1/completions", json={
        "model": "gpt-4o",
        "prompt": "Say hello",
        "temperature": 0.7,
        "max_tokens": 100
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "completion" in data
    assert "Hello from OpenAI!" in data["completion"]
    assert data["model"] == "gpt-4o"
    assert "usage" in data
    assert data["usage"]["total_tokens"] == 15
    assert "signature" in data
    assert "request_hash" in data


@patch('server.get_chat_model_cached')
def test_completion_anthropic(mock_get_model, mock_anthropic_model):
    """Test Anthropic completion endpoint"""
    mock_get_model.return_value = mock_anthropic_model
    
    response = client.post("/v1/completions", json={
        "model": "claude-3.7-sonnet",
        "prompt": "Say hello",
        "temperature": 0.7,
        "max_tokens": 100
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "completion" in data
    assert "Hello from Anthropic!" in data["completion"]
    assert data["model"] == "claude-3.7-sonnet"
    assert "usage" in data
    assert data["usage"]["total_tokens"] == 20


# Chat Completion Tests
@patch('server.get_chat_model_cached')
def test_chat_completion_openai(mock_get_model, mock_openai_model):
    """Test OpenAI chat completion endpoint"""
    mock_get_model.return_value = mock_openai_model
    
    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"]["role"] == "assistant"
    assert "Hello from OpenAI!" in data["message"]["content"]
    assert data["finish_reason"] == "stop"
    assert data["model"] == "gpt-4o"
    assert "usage" in data
    assert data["usage"]["total_tokens"] == 15
    assert "signature" in data


@patch('server.get_chat_model_cached')
def test_chat_completion_anthropic(mock_get_model, mock_anthropic_model):
    """Test Anthropic chat completion endpoint"""
    mock_get_model.return_value = mock_anthropic_model
    
    response = client.post("/v1/chat/completions", json={
        "model": "claude-4.0-sonnet",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Say hello"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Hello from Anthropic!" in data["message"]["content"]
    assert data["finish_reason"] == "stop"
    assert data["model"] == "claude-4.0-sonnet"
    assert data["usage"]["total_tokens"] == 20


@patch('server.get_chat_model_cached')
def test_chat_completion_google(mock_get_model, mock_google_model):
    """Test Google chat completion endpoint"""
    mock_get_model.return_value = mock_google_model
    
    response = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash-preview",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Hello from Google!" in data["message"]["content"]
    assert data["finish_reason"] == "stop"
    assert data["model"] == "gemini-2.5-flash-preview"
    assert data["usage"]["total_tokens"] == 17


@patch('server.get_chat_model_cached')
def test_chat_completion_xai(mock_get_model, mock_xai_model):
    """Test xAI chat completion endpoint"""
    mock_get_model.return_value = mock_xai_model
    
    response = client.post("/v1/chat/completions", json={
        "model": "grok-3-beta",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Hello from xAI!" in data["message"]["content"]
    assert data["finish_reason"] == "stop"
    assert data["model"] == "grok-3-beta"
    assert data["usage"]["total_tokens"] == 20


# Tool Call Tests
@patch('server.get_chat_model_cached')
def test_chat_tool_calls_openai(mock_get_model, mock_tool_call_model):
    """Test OpenAI chat with tool calls"""
    mock_get_model.return_value = mock_tool_call_model
    
    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }],
        "temperature": 0.7,
        "max_tokens": 100
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["finish_reason"] == "tool_calls"
    assert "tool_calls" in data["message"]
    assert len(data["message"]["tool_calls"]) == 1
    
    tool_call = data["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    
    # Parse arguments
    args = json.loads(tool_call["function"]["arguments"])
    assert args["location"] == "San Francisco"
    assert args["unit"] == "celsius"
    assert data["usage"]["total_tokens"] == 40


@patch('server.get_chat_model_cached')
def test_chat_tool_calls_anthropic(mock_get_model, mock_tool_call_model):
    """Test Anthropic chat with tool calls"""
    mock_get_model.return_value = mock_tool_call_model
    
    response = client.post("/v1/chat/completions", json={
        "model": "claude-4.0-sonnet",
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string"}
                    }
                }
            }
        }],
        "temperature": 0.7
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["finish_reason"] == "tool_calls"
    assert "tool_calls" in data["message"]
    assert len(data["message"]["tool_calls"]) > 0


@patch('server.get_chat_model_cached')
def test_chat_tool_calls_google(mock_get_model, mock_tool_call_model):
    """Test Google chat with tool calls"""
    mock_get_model.return_value = mock_tool_call_model
    
    response = client.post("/v1/chat/completions", json={
        "model": "gemini-2.5-flash-preview",
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}}
            }
        }]
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["finish_reason"] == "tool_calls"


@patch('server.get_chat_model_cached')
def test_chat_tool_calls_xai(mock_get_model, mock_tool_call_model):
    """Test xAI chat with tool calls"""
    mock_get_model.return_value = mock_tool_call_model
    
    response = client.post("/v1/chat/completions", json={
        "model": "grok-3-beta",
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}}
            }
        }]
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["finish_reason"] == "tool_calls"


# Streaming Tests
@patch('server.get_chat_model_cached')
def test_chat_streaming_openai(mock_get_model, mock_openai_model):
    """Test OpenAI streaming chat completion"""
    mock_get_model.return_value = mock_openai_model
    
    response = client.post("/v1/chat/completions/stream", json={
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    })
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # Parse SSE stream
    content_chunks = []
    usage_data = None
    finish_reason = None
    
    for line in response.text.split('\n'):
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    if 'delta' in choice and 'content' in choice['delta']:
                        content_chunks.append(choice['delta']['content'])
                    if choice.get('finish_reason'):
                        finish_reason = choice['finish_reason']
                if 'usage' in chunk:
                    usage_data = chunk['usage']
            except json.JSONDecodeError:
                pass
    
    full_content = ''.join(content_chunks)
    assert "Hello from OpenAI!" in full_content
    assert finish_reason == "stop"
    assert usage_data is not None
    assert usage_data["total_tokens"] == 15


@patch('server.get_chat_model_cached')
def test_chat_streaming_anthropic(mock_get_model, mock_anthropic_model):
    """Test Anthropic streaming chat completion"""
    mock_get_model.return_value = mock_anthropic_model
    
    response = client.post("/v1/chat/completions/stream", json={
        "model": "claude-4.0-sonnet",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "temperature": 0.7
    })
    
    assert response.status_code == 200
    
    content_chunks = []
    for line in response.text.split('\n'):
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    if 'delta' in choice and 'content' in choice['delta']:
                        content_chunks.append(choice['delta']['content'])
            except json.JSONDecodeError:
                pass
    
    full_content = ''.join(content_chunks)
    assert "Hello from Anthropic!" in full_content


@patch('server.get_chat_model_cached')
def test_chat_streaming_google(mock_get_model, mock_google_model):
    """Test Google streaming chat completion"""
    mock_get_model.return_value = mock_google_model
    
    response = client.post("/v1/chat/completions/stream", json={
        "model": "gemini-2.5-flash-preview",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ]
    })
    
    assert response.status_code == 200
    
    content_chunks = []
    for line in response.text.split('\n'):
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    if 'delta' in choice and 'content' in choice['delta']:
                        content_chunks.append(choice['delta']['content'])
            except json.JSONDecodeError:
                pass
    
    full_content = ''.join(content_chunks)
    assert "Hello from Google!" in full_content


@patch('server.get_chat_model_cached')
def test_chat_streaming_xai(mock_get_model, mock_xai_model):
    """Test xAI streaming chat completion"""
    mock_get_model.return_value = mock_xai_model
    
    response = client.post("/v1/chat/completions/stream", json={
        "model": "grok-3-beta",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ]
    })
    
    assert response.status_code == 200
    
    content_chunks = []
    for line in response.text.split('\n'):
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    if 'delta' in choice and 'content' in choice['delta']:
                        content_chunks.append(choice['delta']['content'])
            except json.JSONDecodeError:
                pass
    
    full_content = ''.join(content_chunks)
    assert "Hello from xAI!" in full_content


# Streaming with Tool Calls Tests
@patch('server.get_chat_model_cached')
def test_chat_streaming_tool_calls_openai(mock_get_model, mock_tool_call_model):
    """Test OpenAI streaming with tool calls (buffered)"""
    mock_get_model.return_value = mock_tool_call_model
    
    response = client.post("/v1/chat/completions/stream", json={
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}}
            }
        }]
    })
    
    assert response.status_code == 200
    
    tool_calls = []
    finish_reason = None
    usage_data = None
    
    for line in response.text.split('\n'):
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    if 'delta' in choice and 'tool_calls' in choice['delta']:
                        tool_calls.extend(choice['delta']['tool_calls'])
                    if choice.get('finish_reason'):
                        finish_reason = choice['finish_reason']
                if 'usage' in chunk:
                    usage_data = chunk['usage']
            except json.JSONDecodeError:
                pass
    
    # OpenAI buffers tool calls, so we should get complete tool calls
    assert len(tool_calls) > 0
    assert finish_reason == "tool_calls"
    assert usage_data is not None
    assert usage_data["total_tokens"] == 40


@patch('server.get_chat_model_cached')
def test_chat_streaming_tool_calls_anthropic(mock_get_model, mock_tool_call_model):
    """Test Anthropic streaming with tool calls (buffered)"""
    mock_get_model.return_value = mock_tool_call_model
    
    response = client.post("/v1/chat/completions/stream", json={
        "model": "claude-4.0-sonnet",
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}}
            }
        }]
    })
    
    assert response.status_code == 200
    
    finish_reason = None
    has_tool_calls = False
    
    for line in response.text.split('\n'):
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    if 'delta' in choice and 'tool_calls' in choice['delta']:
                        has_tool_calls = True
                    if choice.get('finish_reason'):
                        finish_reason = choice['finish_reason']
            except json.JSONDecodeError:
                pass
    
    assert has_tool_calls
    assert finish_reason == "tool_calls"


@patch('server.get_chat_model_cached')
def test_chat_streaming_tool_calls_google(mock_get_model, mock_tool_call_model):
    """Test Google streaming with tool calls (not buffered)"""
    mock_get_model.return_value = mock_tool_call_model
    
    response = client.post("/v1/chat/completions/stream", json={
        "model": "gemini-2.5-flash-preview",
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}}
            }
        }]
    })
    
    assert response.status_code == 200
    
    finish_reason = None
    has_tool_calls = False
    
    for line in response.text.split('\n'):
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    if 'delta' in choice and 'tool_calls' in choice['delta']:
                        has_tool_calls = True
                    if choice.get('finish_reason'):
                        finish_reason = choice['finish_reason']
            except json.JSONDecodeError:
                pass
    
    # Google streams tool calls immediately
    assert has_tool_calls
    assert finish_reason == "tool_calls"


@patch('server.get_chat_model_cached')
def test_chat_streaming_tool_calls_xai(mock_get_model, mock_tool_call_model):
    """Test xAI streaming with tool calls (not buffered)"""
    mock_get_model.return_value = mock_tool_call_model
    
    response = client.post("/v1/chat/completions/stream", json={
        "model": "grok-3-beta",
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}}
            }
        }]
    })
    
    assert response.status_code == 200
    
    finish_reason = None
    has_tool_calls = False
    
    for line in response.text.split('\n'):
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    if 'delta' in choice and 'tool_calls' in choice['delta']:
                        has_tool_calls = True
                    if choice.get('finish_reason'):
                        finish_reason = choice['finish_reason']
            except json.JSONDecodeError:
                pass
    
    assert has_tool_calls
    assert finish_reason == "tool_calls"


# Test Summary Function
def run_all_tests():
    """Run all tests and return summary"""
    import sys
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--color=yes'
    ])
    
    return exit_code == 0


if __name__ == "__main__":
    print("=" * 80)
    print("TEE LLM Router - Comprehensive Unit Tests")
    print("=" * 80)
    print()
    
    success = run_all_tests()
    
    print()
    print("=" * 80)
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)
    
    sys.exit(0 if success else 1)
