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


class TestToolForwarding(unittest.TestCase):
    """Unit tests for tool forwarding functionality"""

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
            "model": "gpt-4",
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
        self.assertEqual(result.model, "gpt-4")
        self.assertIsNotNone(result.tools)
        self.assertEqual(len(result.tools), 1)
        self.assertEqual(result.tool_choice, "auto")

    @patch("tee_gateway.controllers.chat_controller.http_session")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_tools_forwarded_to_backend(self, mock_connexion, mock_http_session):
        """Test that tools are forwarded to HTTP backend"""
        # Setup mock request
        mock_connexion.request.is_json = True
        mock_connexion.request.headers.get.return_value = None
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4",
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
            "tool_choice": "auto",
            "stream": False,
        }

        # Setup mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "The weather is sunny."},
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "timestamp": "2025-01-26T12:00:00Z",
            "signature": "test_signature",
            "request_hash": "test_hash",
        }
        mock_http_session.post.return_value = mock_response

        # Call the function
        result = create_chat_completion(None)

        # Verify HTTP POST was called
        mock_http_session.post.assert_called_once()
        call_kwargs = mock_http_session.post.call_args[1]

        # Check that tools were included in the request
        request_json = call_kwargs["json"]
        self.assertIn("tools", request_json)
        self.assertEqual(len(request_json["tools"]), 1)
        self.assertEqual(request_json["tools"][0]["type"], "function")
        self.assertEqual(request_json["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(request_json["tool_choice"], "auto")

        # Verify response structure
        self.assertIn("choices", result)
        self.assertEqual(len(result["choices"]), 1)

    @patch("tee_gateway.controllers.chat_controller.http_session")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_tool_calls_extracted_from_response(
        self, mock_connexion, mock_http_session
    ):
        """Test that tool_calls are extracted from HTTP response"""
        # Setup mock request
        mock_connexion.request.is_json = True
        mock_connexion.request.headers.get.return_value = None
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "stream": False,
        }

        # Setup mock HTTP response with tool_calls
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
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
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "timestamp": "2025-01-26T12:00:00Z",
            "signature": "test_signature",
            "request_hash": "test_hash",
        }
        mock_http_session.post.return_value = mock_response

        # Call the function
        response = create_chat_completion(None)

        # Verify tool_calls are in the response
        self.assertIn("choices", response)
        self.assertEqual(len(response["choices"]), 1)
        message = response["choices"][0]["message"]
        self.assertIn("tool_calls", message)
        self.assertEqual(len(message["tool_calls"]), 1)
        self.assertEqual(message["tool_calls"][0]["id"], "call_abc123")
        self.assertEqual(message["tool_calls"][0]["type"], "function")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(
            message["tool_calls"][0]["function"]["arguments"],
            '{"location": "San Francisco"}',
        )

    @patch("tee_gateway.controllers.chat_controller.http_session")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_tool_message_forwarded_to_backend(self, mock_connexion, mock_http_session):
        """Test that tool messages in the conversation are forwarded to HTTP backend"""
        # Setup mock request with tool message
        mock_connexion.request.is_json = True
        mock_connexion.request.headers.get.return_value = None
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4",
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

        # Setup mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "The weather in San Francisco is 72°F and sunny.",
            },
            "model": "gpt-4",
            "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
            "timestamp": "2025-01-26T12:00:00Z",
            "signature": "test_signature",
            "request_hash": "test_hash",
        }
        mock_http_session.post.return_value = mock_response

        # Call the function
        result = create_chat_completion(None)

        # Verify HTTP POST was called with tool message
        mock_http_session.post.assert_called_once()
        call_kwargs = mock_http_session.post.call_args[1]

        request_json = call_kwargs["json"]
        messages = request_json["messages"]

        # Should have 3 messages: user, assistant with tool_calls, tool
        self.assertEqual(len(messages), 3)

        # Check tool message was converted
        tool_msg = messages[2]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(
            tool_msg["content"], '{"temperature": 72, "condition": "sunny"}'
        )
        self.assertEqual(tool_msg["tool_call_id"], "call_abc123")

        # Check assistant message has tool_calls
        assistant_msg = messages[1]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertIn("tool_calls", assistant_msg)
        self.assertEqual(len(assistant_msg["tool_calls"]), 1)
        self.assertEqual(assistant_msg["tool_calls"][0]["id"], "call_abc123")
        self.assertEqual(
            assistant_msg["tool_calls"][0]["function"]["name"], "get_weather"
        )

        # Verify response
        self.assertIn("choices", result)
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")

    @patch("tee_gateway.controllers.chat_controller.http_session")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_payment_header_forwarded(self, mock_connexion, mock_http_session):
        """Test that X-PAYMENT header is forwarded to backend"""
        # Setup mock request with payment header
        mock_connexion.request.is_json = True
        mock_connexion.request.headers.get.return_value = "payment_token_123"
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        # Setup mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
            "model": "gpt-4",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        mock_http_session.post.return_value = mock_response

        # Call the function
        create_chat_completion(None)

        # Verify X-PAYMENT header was included
        mock_http_session.post.assert_called_once()
        call_kwargs = mock_http_session.post.call_args[1]

        self.assertIn("headers", call_kwargs)
        self.assertIn("X-PAYMENT", call_kwargs["headers"])
        self.assertEqual(call_kwargs["headers"]["X-PAYMENT"], "payment_token_123")

    @patch("tee_gateway.controllers.chat_controller.http_session")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_tee_metadata_preserved(self, mock_connexion, mock_http_session):
        """Test that TEE metadata is preserved in response"""
        # Setup mock request
        mock_connexion.request.is_json = True
        mock_connexion.request.headers.get.return_value = None
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        # Setup mock HTTP response with TEE metadata
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
            "model": "gpt-4",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            "timestamp": "2025-01-26T12:00:00Z",
            "signature": "tee_signature_abc123",
            "request_hash": "hash_def456",
        }
        mock_http_session.post.return_value = mock_response

        # Call the function
        result = create_chat_completion(None)

        # Verify basic response structure
        self.assertIn("choices", result)
        self.assertIn("model", result)
        self.assertEqual(result["model"], "gpt-4")

        # Verify TEE metadata is preserved in response
        self.assertIn("tee_signature", result)
        self.assertEqual(result["tee_signature"], "tee_signature_abc123")
        self.assertIn("tee_request_hash", result)
        self.assertEqual(result["tee_request_hash"], "hash_def456")
        self.assertIn("tee_timestamp", result)
        self.assertEqual(result["tee_timestamp"], "2025-01-26T12:00:00Z")

    @patch("tee_gateway.controllers.chat_controller.http_session")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_http_error_handling(self, mock_connexion, mock_http_session):
        """Test that HTTP errors are handled properly"""
        # Setup mock request
        mock_connexion.request.is_json = True
        mock_connexion.request.headers.get.return_value = None
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        # Setup mock to raise HTTP error
        import requests

        mock_http_session.post.side_effect = requests.exceptions.RequestException(
            "Connection failed"
        )

        # Call the function
        result, status_code = create_chat_completion(None)

        # Verify error response
        self.assertEqual(status_code, 500)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Backend request failed")
        self.assertIn("details", result)


if __name__ == "__main__":
    unittest.main()
