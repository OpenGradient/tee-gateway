import json
import unittest
from unittest.mock import patch, MagicMock

from tee_gateway.controllers.chat_controller import (
    _parse_chat_request as parse_chat_request,
    _chat_request_to_dict as chat_request_to_dict,
)
from tee_gateway.models.create_chat_completion_request import (
    CreateChatCompletionRequest,
)


class TestResponseFormatParsing(unittest.TestCase):
    """Tests for response_format parsing from request dicts."""

    def _base_request(self, **overrides):
        d = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        d.update(overrides)
        return d

    def test_no_response_format(self):
        req = parse_chat_request(self._base_request())
        self.assertIsNone(req.response_format)

    def test_text_response_format(self):
        req = parse_chat_request(self._base_request(response_format={"type": "text"}))
        self.assertEqual(req.response_format, {"type": "text"})

    def test_json_object_response_format(self):
        rf = {"type": "json_object"}
        req = parse_chat_request(self._base_request(response_format=rf))
        self.assertEqual(req.response_format, {"type": "json_object"})

    def test_json_schema_response_format(self):
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "user_info",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                },
            },
        }
        req = parse_chat_request(self._base_request(response_format=rf))
        self.assertEqual(req.response_format["type"], "json_schema")
        self.assertEqual(req.response_format["json_schema"]["name"], "user_info")
        self.assertTrue(req.response_format["json_schema"]["strict"])


class TestResponseFormatInHashDict(unittest.TestCase):
    """Tests that response_format is included in the TEE hash dict."""

    def _make_request(self, response_format=None):
        return CreateChatCompletionRequest(
            model="gpt-4.1",
            messages=[],
            temperature=1.0,
            response_format=response_format,
        )

    def test_no_response_format_omitted(self):
        req = self._make_request()
        d = chat_request_to_dict(req)
        self.assertNotIn("response_format", d)

    def test_json_object_included(self):
        req = self._make_request(response_format={"type": "json_object"})
        d = chat_request_to_dict(req)
        self.assertIn("response_format", d)
        self.assertEqual(d["response_format"]["type"], "json_object")

    def test_json_schema_included(self):
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "math_answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "number"}},
                },
            },
        }
        req = self._make_request(response_format=rf)
        d = chat_request_to_dict(req)
        self.assertEqual(d["response_format"], rf)

    def test_hash_deterministic_with_response_format(self):
        rf = {"type": "json_object"}
        req = self._make_request(response_format=rf)
        d1 = json.dumps(chat_request_to_dict(req), sort_keys=True)
        d2 = json.dumps(chat_request_to_dict(req), sort_keys=True)
        self.assertEqual(d1, d2)

    def test_hash_differs_with_and_without_response_format(self):
        req_plain = self._make_request()
        req_json = self._make_request(response_format={"type": "json_object"})
        h1 = json.dumps(chat_request_to_dict(req_plain), sort_keys=True)
        h2 = json.dumps(chat_request_to_dict(req_json), sort_keys=True)
        self.assertNotEqual(h1, h2)


class TestResponseFormatModelBinding(unittest.TestCase):
    """Tests that response_format is bound to the model before invocation."""

    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_json_object_binds_to_model(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash
    ):
        from tee_gateway.controllers.chat_controller import (
            _create_non_streaming_response,
        )

        mock_model = MagicMock()
        mock_bound = MagicMock()
        mock_model.bind.return_value = mock_bound
        mock_get_model.return_value = mock_model

        mock_response = MagicMock()
        mock_response.content = "test"
        mock_response.tool_calls = None
        mock_bound.invoke.return_value = mock_response

        mock_convert.return_value = []
        mock_hash.return_value = (b"hash", "input_hex", "output_hex")
        mock_keys = MagicMock()
        mock_keys.sign_data.return_value = "sig"
        mock_keys.get_tee_id.return_value = "abc"
        mock_tee_keys.return_value = mock_keys

        req = CreateChatCompletionRequest(
            model="gpt-4.1",
            messages=[],
            temperature=1.0,
            response_format={"type": "json_object"},
        )

        _create_non_streaming_response(req)

        mock_model.bind.assert_called_once_with(response_format={"type": "json_object"})
        mock_bound.invoke.assert_called_once()

    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_text_format_does_not_bind(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash
    ):
        from tee_gateway.controllers.chat_controller import (
            _create_non_streaming_response,
        )

        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_response = MagicMock()
        mock_response.content = "test"
        mock_response.tool_calls = None
        mock_model.invoke.return_value = mock_response

        mock_convert.return_value = []
        mock_hash.return_value = (b"hash", "input_hex", "output_hex")
        mock_keys = MagicMock()
        mock_keys.sign_data.return_value = "sig"
        mock_keys.get_tee_id.return_value = "abc"
        mock_tee_keys.return_value = mock_keys

        req = CreateChatCompletionRequest(
            model="gpt-4.1",
            messages=[],
            temperature=1.0,
            response_format={"type": "text"},
        )

        _create_non_streaming_response(req)

        mock_model.bind.assert_not_called()
        mock_model.invoke.assert_called_once()

    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_no_format_does_not_bind(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash
    ):
        from tee_gateway.controllers.chat_controller import (
            _create_non_streaming_response,
        )

        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_response = MagicMock()
        mock_response.content = "result"
        mock_response.tool_calls = None
        mock_model.invoke.return_value = mock_response

        mock_convert.return_value = []
        mock_hash.return_value = (b"hash", "input_hex", "output_hex")
        mock_keys = MagicMock()
        mock_keys.sign_data.return_value = "sig"
        mock_keys.get_tee_id.return_value = "abc"
        mock_tee_keys.return_value = mock_keys

        req = CreateChatCompletionRequest(
            model="gpt-4.1",
            messages=[],
            temperature=1.0,
        )

        _create_non_streaming_response(req)

        mock_model.bind.assert_not_called()

    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_json_schema_binds_full_schema(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash
    ):
        from tee_gateway.controllers.chat_controller import (
            _create_non_streaming_response,
        )

        mock_model = MagicMock()
        mock_bound = MagicMock()
        mock_model.bind.return_value = mock_bound
        mock_get_model.return_value = mock_model

        mock_response = MagicMock()
        mock_response.content = '{"name": "Alice", "age": 30}'
        mock_response.tool_calls = None
        mock_bound.invoke.return_value = mock_response

        mock_convert.return_value = []
        mock_hash.return_value = (b"hash", "input_hex", "output_hex")
        mock_keys = MagicMock()
        mock_keys.sign_data.return_value = "sig"
        mock_keys.get_tee_id.return_value = "abc"
        mock_tee_keys.return_value = mock_keys

        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "user_info",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                },
            },
        }

        req = CreateChatCompletionRequest(
            model="gpt-4.1",
            messages=[],
            temperature=1.0,
            response_format=rf,
        )

        _create_non_streaming_response(req)

        mock_model.bind.assert_called_once_with(response_format=rf)


class TestResponseFormatWithTools(unittest.TestCase):
    """Tests that response_format works alongside tool binding."""

    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_tools_and_response_format_both_bind(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash
    ):
        from tee_gateway.controllers.chat_controller import (
            _create_non_streaming_response,
        )

        mock_model = MagicMock()
        mock_after_tools = MagicMock()
        mock_after_format = MagicMock()
        mock_model.bind_tools.return_value = mock_after_tools
        mock_after_tools.bind.return_value = mock_after_format
        mock_get_model.return_value = mock_model

        mock_response = MagicMock()
        mock_response.content = '{"result": 42}'
        mock_response.tool_calls = None
        mock_after_format.invoke.return_value = mock_response

        mock_convert.return_value = []
        mock_hash.return_value = (b"hash", "input_hex", "output_hex")
        mock_keys = MagicMock()
        mock_keys.sign_data.return_value = "sig"
        mock_keys.get_tee_id.return_value = "abc"
        mock_tee_keys.return_value = mock_keys

        req = CreateChatCompletionRequest(
            model="gpt-4.1",
            messages=[],
            temperature=1.0,
            tools=[
                {"type": "function", "function": {"name": "calc", "parameters": {}}}
            ],
            response_format={"type": "json_object"},
        )

        _create_non_streaming_response(req)

        mock_model.bind_tools.assert_called_once()
        mock_after_tools.bind.assert_called_once_with(
            response_format={"type": "json_object"}
        )
        mock_after_format.invoke.assert_called_once()


class TestAnthropicTitleInjection(unittest.TestCase):
    """Tests that the schema 'name' is injected as 'title' for LangChain-Anthropic."""

    @patch("tee_gateway.controllers.chat_controller.get_provider_from_model")
    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_name_injected_as_title(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash, mock_provider
    ):
        """Schema 'name' from json_schema wrapper is added as 'title' in the schema dict."""
        from tee_gateway.controllers.chat_controller import _invoke_anthropic_structured
        from langchain_core.messages import AIMessage

        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {"name": "Alice", "age": 30}

        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                    # NOTE: no "title" key here — must be injected from "name"
                },
            },
        }

        result = _invoke_anthropic_structured(mock_model, rf, [])

        called_schema = mock_model.with_structured_output.call_args[0][0]
        self.assertEqual(called_schema["title"], "person")
        self.assertIsInstance(result, AIMessage)
        self.assertEqual(json.loads(result.content), {"name": "Alice", "age": 30})

    @patch("tee_gateway.controllers.chat_controller.get_provider_from_model")
    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_existing_title_not_overwritten(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash, mock_provider
    ):
        """If the schema already has a 'title', it is left untouched."""
        from tee_gateway.controllers.chat_controller import _invoke_anthropic_structured

        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {"x": 1}

        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "wrapper_name",
                "schema": {
                    "type": "object",
                    "title": "existing_title",
                    "properties": {"x": {"type": "integer"}},
                },
            },
        }

        _invoke_anthropic_structured(mock_model, rf, [])

        called_schema = mock_model.with_structured_output.call_args[0][0]
        self.assertEqual(called_schema["title"], "existing_title")

    def test_json_object_raises_for_anthropic(self):
        """json_object raises a clear ValueError for Anthropic."""
        from tee_gateway.controllers.chat_controller import _invoke_anthropic_structured

        mock_model = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            _invoke_anthropic_structured(mock_model, {"type": "json_object"}, [])
        self.assertIn("json_object", str(ctx.exception))
        self.assertIn("json_schema", str(ctx.exception))


class TestStreamingResponseFormatBinding(unittest.TestCase):
    """Tests that response_format is bound correctly in the streaming path."""

    def _base_streaming_request(self, model="gpt-4.1", response_format=None):
        return CreateChatCompletionRequest(
            model=model,
            messages=[],
            temperature=1.0,
            stream=True,
            response_format=response_format,
        )

    @patch("tee_gateway.controllers.chat_controller.get_provider_from_model")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_json_object_binds_in_streaming(
        self, mock_get_model, mock_convert, mock_provider
    ):
        """json_object is bound to the model in the streaming path."""
        from tee_gateway.controllers.chat_controller import _create_streaming_response

        mock_provider.return_value = "openai"
        mock_model = MagicMock()
        mock_bound = MagicMock()
        mock_model.bind.return_value = mock_bound
        mock_get_model.return_value = mock_model
        mock_convert.return_value = []

        _create_streaming_response(
            self._base_streaming_request(response_format={"type": "json_object"})
        )

        mock_model.bind.assert_called_once_with(response_format={"type": "json_object"})

    @patch("tee_gateway.controllers.chat_controller.get_provider_from_model")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_text_format_does_not_bind_in_streaming(
        self, mock_get_model, mock_convert, mock_provider
    ):
        """text format skips binding in the streaming path."""
        from tee_gateway.controllers.chat_controller import _create_streaming_response

        mock_provider.return_value = "openai"
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_convert.return_value = []

        _create_streaming_response(
            self._base_streaming_request(response_format={"type": "text"})
        )

        mock_model.bind.assert_not_called()

    @patch("tee_gateway.controllers.chat_controller.get_provider_from_model")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_no_format_does_not_bind_in_streaming(
        self, mock_get_model, mock_convert, mock_provider
    ):
        """Omitting response_format skips binding in the streaming path."""
        from tee_gateway.controllers.chat_controller import _create_streaming_response

        mock_provider.return_value = "openai"
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_convert.return_value = []

        _create_streaming_response(self._base_streaming_request())

        mock_model.bind.assert_not_called()

    @patch("tee_gateway.controllers.chat_controller.get_provider_from_model")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_json_schema_binds_full_schema_in_streaming(
        self, mock_get_model, mock_convert, mock_provider
    ):
        """The full json_schema dict is bound to the model in the streaming path."""
        from tee_gateway.controllers.chat_controller import _create_streaming_response

        mock_provider.return_value = "openai"
        mock_model = MagicMock()
        mock_bound = MagicMock()
        mock_model.bind.return_value = mock_bound
        mock_get_model.return_value = mock_model
        mock_convert.return_value = []

        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        }
        _create_streaming_response(self._base_streaming_request(response_format=rf))

        mock_model.bind.assert_called_once_with(response_format=rf)

    @patch("tee_gateway.controllers.chat_controller.get_provider_from_model")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_anthropic_does_not_bind_in_streaming(
        self, mock_get_model, mock_convert, mock_provider
    ):
        """Anthropic models never call model.bind() — structured output goes via with_structured_output."""
        from tee_gateway.controllers.chat_controller import _create_streaming_response

        mock_provider.return_value = "anthropic"
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {"name": "Alice", "age": 30}
        mock_get_model.return_value = mock_model
        mock_convert.return_value = []

        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        _create_streaming_response(
            self._base_streaming_request(model="claude-sonnet-4-5", response_format=rf)
        )

        mock_model.bind.assert_not_called()
        mock_model.with_structured_output.assert_called_once()


class TestStreamingAnthropicStructuredOutput(unittest.TestCase):
    """Tests the SSE output of Anthropic structured output in the streaming path."""

    def _consume_sse(self, response) -> list[dict]:
        """Drain the SSE generator and return parsed data objects (skips [DONE])."""
        events = []
        for line in response.response:
            if isinstance(line, bytes):
                line = line.decode()
            line = line.strip()
            if line.startswith("data: "):
                payload = line[6:]
                if payload != "[DONE]":
                    events.append(json.loads(payload))
        return events

    @patch("tee_gateway.controllers.chat_controller.get_provider_from_model")
    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_anthropic_structured_content_emitted_as_single_chunk(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash, mock_provider
    ):
        """Anthropic structured output is emitted as one complete content chunk."""
        from tee_gateway.controllers.chat_controller import _create_streaming_response

        mock_provider.return_value = "anthropic"
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {"name": "Alice", "age": 30}
        mock_get_model.return_value = mock_model
        mock_convert.return_value = []

        mock_hash.return_value = (b"hash", "input_hex", "output_hex")
        mock_keys = MagicMock()
        mock_keys.sign_data.return_value = "sig"
        mock_keys.get_tee_id.return_value = "abc"
        mock_tee_keys.return_value = mock_keys

        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                },
            },
        }
        req = CreateChatCompletionRequest(
            model="claude-sonnet-4-5",
            messages=[],
            temperature=1.0,
            stream=True,
            response_format=rf,
        )

        resp = _create_streaming_response(req)
        events = self._consume_sse(resp)

        # First chunk carries the full structured content
        content_chunk = events[0]
        delta_content = content_chunk["choices"][0]["delta"]["content"]
        self.assertEqual(json.loads(delta_content), {"name": "Alice", "age": 30})

        # Final chunk carries the TEE signature fields
        final_chunk = events[-1]
        self.assertIn("tee_signature", final_chunk)
        self.assertIn("tee_request_hash", final_chunk)
        self.assertIn("tee_output_hash", final_chunk)

    @patch("tee_gateway.controllers.chat_controller.get_provider_from_model")
    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_anthropic_structured_output_in_tee_hash(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash, mock_provider
    ):
        """The structured JSON content (not raw dict repr) is passed to compute_tee_msg_hash."""
        from tee_gateway.controllers.chat_controller import _create_streaming_response

        mock_provider.return_value = "anthropic"
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {"answer": 42}
        mock_get_model.return_value = mock_model
        mock_convert.return_value = []

        mock_hash.return_value = (b"hash", "input_hex", "output_hex")
        mock_keys = MagicMock()
        mock_keys.sign_data.return_value = "sig"
        mock_keys.get_tee_id.return_value = "abc"
        mock_tee_keys.return_value = mock_keys

        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                },
            },
        }
        req = CreateChatCompletionRequest(
            model="claude-sonnet-4-5",
            messages=[],
            temperature=1.0,
            stream=True,
            response_format=rf,
        )

        resp = _create_streaming_response(req)
        self._consume_sse(resp)

        # compute_tee_msg_hash must receive a JSON string, not a Python dict repr
        _, output_content_arg, _ = mock_hash.call_args[0]
        parsed = json.loads(output_content_arg)
        self.assertEqual(parsed, {"answer": 42})


if __name__ == "__main__":
    unittest.main()
