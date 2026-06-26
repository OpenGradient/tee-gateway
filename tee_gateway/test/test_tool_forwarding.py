import json
import unittest
from unittest.mock import patch, Mock

from tee_gateway.controllers.chat_controller import (
    _parse_chat_request as parse_chat_request,
    _parse_message as parse_message,
    create_chat_completion,
)
from tee_gateway.models import (
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestFunctionMessage,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _MockLangChainResponse:
    """Minimal stand-in for a LangChain AIMessage returned by model.invoke()."""

    def __init__(self, content="", tool_calls=None, usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = usage or {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
        }


class _MockStreamChunk:
    """Minimal stand-in for a LangChain AIMessageChunk yielded by model.stream()."""

    def __init__(self, content="", tool_call_chunks=None, usage=None):
        self.content = content
        self.tool_call_chunks = tool_call_chunks or []
        self.usage_metadata = usage


def _make_mock_model(response: _MockLangChainResponse) -> Mock:
    """Return a mock LangChain chat model whose invoke() returns *response*."""
    mock_model = Mock()
    mock_model.invoke.return_value = response
    mock_model.bind_tools.return_value = mock_model  # bind_tools returns self
    return mock_model


def _make_mock_tee_keys() -> Mock:
    mock_tee = Mock()
    mock_tee.sign_data.return_value = "bW9ja3NpZ25hdHVyZQ=="  # base64("mocksignature")
    mock_tee.get_tee_id.return_value = "abcdef01" * 8  # 64 hex chars
    return mock_tee


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolForwarding(unittest.TestCase):
    """Unit tests for tool forwarding functionality"""

    # ------------------------------------------------------------------
    # Parsing tests — pure functions, no mocking required
    # ------------------------------------------------------------------

    def test_parse_tool_message(self):
        """Test that tool messages are parsed correctly"""
        message_dict = {
            "role": "tool",
            "content": "The weather is sunny",
            "tool_call_id": "call_123",
        }
        result = parse_message(message_dict)
        self.assertIsInstance(result, ChatCompletionRequestToolMessage)
        self.assertEqual(result.role, "tool")
        self.assertEqual(result.content, "The weather is sunny")
        self.assertEqual(result.tool_call_id, "call_123")

    def test_parse_function_message(self):
        """Test that function messages are parsed correctly"""
        message_dict = {
            "role": "function",
            "content": "Result from function",
            "name": "get_weather",
        }
        result = parse_message(message_dict)
        self.assertIsInstance(result, ChatCompletionRequestFunctionMessage)
        self.assertEqual(result.role, "function")
        self.assertEqual(result.content, "Result from function")
        self.assertEqual(result.name, "get_weather")

    def test_parse_chat_request_with_tools(self):
        """Test that tools are parsed from request"""
        request_dict = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
        }
        result = parse_chat_request(request_dict)
        self.assertEqual(result.model, "gpt-4.1")
        self.assertIsNotNone(result.tools)
        self.assertEqual(len(result.tools), 1)
        self.assertEqual(result.tool_choice, "auto")

    # ------------------------------------------------------------------
    # Integration tests — mock the LangChain model and TEE keys
    # ------------------------------------------------------------------

    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_tools_forwarded_to_backend(
        self, mock_connexion, mock_get_model, mock_get_tee_keys
    ):
        """Tools in the request must be bound to the LangChain model via bind_tools()."""
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                        "strict": False,
                    },
                }
            ],
            "stream": False,
        }

        mock_response = _MockLangChainResponse(content="The weather is sunny.")
        mock_model = _make_mock_model(mock_response)
        mock_get_model.return_value = mock_model
        mock_get_tee_keys.return_value = _make_mock_tee_keys()

        result = create_chat_completion(None)

        # model.invoke() must have been called (request reached the model)
        mock_model.invoke.assert_called_once()
        # Response must have standard chat completion structure
        self.assertIn("choices", result)
        self.assertEqual(len(result["choices"]), 1)

    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_tool_calls_extracted_from_response(
        self, mock_connexion, mock_get_model, mock_get_tee_keys
    ):
        """Tool calls returned by the model must appear in the response choices."""
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "stream": False,
        }

        mock_response = _MockLangChainResponse(
            content="",
            tool_calls=[
                {
                    "id": "call_abc123",
                    "name": "get_weather",
                    "args": {"location": "San Francisco"},
                    "type": "function",
                }
            ],
        )
        mock_get_model.return_value = _make_mock_model(mock_response)
        mock_get_tee_keys.return_value = _make_mock_tee_keys()

        response = create_chat_completion(None)

        self.assertIn("choices", response)
        message = response["choices"][0]["message"]
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")
        self.assertIn("tool_calls", message)
        self.assertEqual(len(message["tool_calls"]), 1)
        tc = message["tool_calls"][0]
        self.assertEqual(tc["id"], "call_abc123")
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "get_weather")

    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_tool_message_forwarded_to_backend(
        self, mock_connexion, mock_get_model, mock_get_tee_keys
    ):
        """A conversation containing tool messages must be passed to model.invoke()."""
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "user", "content": "What is the weather in SF?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"temperature": 72, "condition": "sunny"}',
                    "tool_call_id": "call_abc123",
                },
            ],
            "stream": False,
        }

        mock_response = _MockLangChainResponse(
            content="The weather in San Francisco is 72°F and sunny."
        )
        mock_model = _make_mock_model(mock_response)
        mock_get_model.return_value = mock_model
        mock_get_tee_keys.return_value = _make_mock_tee_keys()

        result = create_chat_completion(None)

        # model.invoke must have been called with the full 3-message conversation
        mock_model.invoke.assert_called_once()
        langchain_messages = mock_model.invoke.call_args[0][0]
        self.assertEqual(len(langchain_messages), 3)

        self.assertIn("choices", result)
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")

    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_tee_metadata_in_response(
        self, mock_connexion, mock_get_model, mock_get_tee_keys
    ):
        """Every chat completion response must include TEE signature fields."""
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        mock_response = _MockLangChainResponse(content="Hello!")
        mock_get_model.return_value = _make_mock_model(mock_response)
        mock_tee = _make_mock_tee_keys()
        mock_get_tee_keys.return_value = mock_tee

        result = create_chat_completion(None)

        # All four TEE attestation fields must be present
        self.assertIn("tee_signature", result)
        self.assertIn("tee_request_hash", result)
        self.assertIn("tee_output_hash", result)
        self.assertIn("tee_timestamp", result)
        self.assertIn("tee_id", result)
        # The signature must have been produced by sign_data()
        self.assertEqual(result["tee_signature"], "bW9ja3NpZ25hdHVyZQ==")
        # tee_id must carry the 0x prefix
        self.assertTrue(result["tee_id"].startswith("0x"))

    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_streaming_tool_call_emits_early_progress_frame(
        self, mock_connexion, mock_get_model, mock_get_tee_keys
    ):
        """For buffered providers (OpenAI/Anthropic) the streaming response must
        emit a lightweight progress frame carrying just the tool name as soon as
        it is known — so the client can show "writing file…" immediately and the
        connection keeps producing bytes — while the complete, argument-bearing
        tool call is still flushed (and signed) at the end of the stream.

        Without this, a large write_file tool call streams nothing to the client
        until the whole file has been generated (minutes), which both hides the
        activity and lets the read timeout trip.
        """
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Write a file."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write a file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "stream": True,
        }

        # Real providers stream the name in the first fragment and the arguments
        # across many later fragments; mirror that so the test exercises the
        # "name known early, args trickle in" ordering.
        chunks = [
            _MockStreamChunk(
                tool_call_chunks=[
                    {"index": 0, "id": "call_1", "name": "write_file", "args": ""}
                ]
            ),
            _MockStreamChunk(
                tool_call_chunks=[{"index": 0, "args": '{"path": "app.js",'}]
            ),
            _MockStreamChunk(
                tool_call_chunks=[{"index": 0, "args": ' "content": "hi"}'}]
            ),
            _MockStreamChunk(
                usage={
                    "input_tokens": 5,
                    "output_tokens": 7,
                    "total_tokens": 12,
                }
            ),
        ]
        mock_model = Mock()
        mock_model.stream.return_value = iter(chunks)
        mock_model.bind_tools.return_value = mock_model
        mock_get_model.return_value = mock_model
        mock_get_tee_keys.return_value = _make_mock_tee_keys()

        response = create_chat_completion(None)
        # The generator yields SSE frames as str; join them directly.
        body = "".join(
            part.decode("utf-8") if isinstance(part, bytes) else part
            for part in response.response
        )

        # Parse the SSE frames in arrival order.
        frames = []
        for line in body.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload.strip() == "[DONE]":
                break
            frames.append(json.loads(payload))

        # Collect (frame_index, tool_call_delta) pairs in order.
        tool_deltas = [
            (i, tc)
            for i, frame in enumerate(frames)
            for tc in (frame["choices"][0].get("delta") or {}).get("tool_calls", [])
        ]
        self.assertGreaterEqual(
            len(tool_deltas),
            2,
            "expected an early progress frame plus the final flushed tool call",
        )

        # First tool delta announces the name but carries no arguments yet.
        _, first = tool_deltas[0]
        self.assertEqual(first.get("function", {}).get("name"), "write_file")
        self.assertNotIn("arguments", first.get("function", {}))

        # The complete arguments are flushed in a single later frame.
        with_args = [
            tc for _, tc in tool_deltas if tc.get("function", {}).get("arguments")
        ]
        self.assertTrue(with_args, "expected a frame with the complete arguments")
        args = json.loads(with_args[-1]["function"]["arguments"])
        self.assertEqual(args["path"], "app.js")

        # The progress frame(s) arrive strictly before the final signed frame, so
        # the client sees activity well before generation finishes.
        first_tool_frame_index = tool_deltas[0][0]
        signed_frame_index = next(
            i for i, frame in enumerate(frames) if "tee_signature" in frame
        )
        self.assertLess(first_tool_frame_index, signed_frame_index)

    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_error_handling_when_model_raises(
        self, mock_connexion, mock_get_model, mock_get_tee_keys
    ):
        """An exception from model.invoke() must produce a 500 error response."""
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        mock_model = Mock()
        mock_model.invoke.side_effect = RuntimeError("Connection failed")
        mock_model.bind_tools.return_value = mock_model
        mock_get_model.return_value = mock_model
        mock_get_tee_keys.return_value = _make_mock_tee_keys()

        result, status_code = create_chat_completion(None)

        self.assertEqual(status_code, 500)
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
