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

        from langchain_core.messages import AIMessage as _AIMessage

        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        # include_raw=True returns {"raw": AIMessage, "parsed": dict, "parsing_error": None}
        mock_structured.invoke.return_value = {
            "raw": _AIMessage(content='{"name": "Alice", "age": 30}'),
            "parsed": {"name": "Alice", "age": 30},
            "parsing_error": None,
        }

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
        from langchain_core.messages import AIMessage as _AIMessage

        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {
            "raw": _AIMessage(content='{"name": "Alice", "age": 30}'),
            "parsed": {"name": "Alice", "age": 30},
            "parsing_error": None,
        }
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

        from langchain_core.messages import AIMessage as _AIMessage

        mock_provider.return_value = "anthropic"
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {
            "raw": _AIMessage(content='{"name": "Alice", "age": 30}'),
            "parsed": {"name": "Alice", "age": 30},
            "parsing_error": None,
        }
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

        from langchain_core.messages import AIMessage as _AIMessage

        mock_provider.return_value = "anthropic"
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {
            "raw": _AIMessage(content='{"answer": 42}'),
            "parsed": {"answer": 42},
            "parsing_error": None,
        }
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


class TestJsonObjectKeywordInjection(unittest.TestCase):
    """Tests that a 'json' system message is injected for json_object mode.

    OpenAI rejects requests with response_format.type='json_object' unless the
    word 'json' appears somewhere in the messages. The gateway injects a brief
    system instruction when no message satisfies this requirement.
    """

    def _setup_non_streaming(self, mock_get_model, mock_convert, mock_tee_keys, mock_hash):
        """Return (mock_model, mock_bound) wired for a minimal non-streaming response."""
        mock_model = MagicMock()
        mock_bound = MagicMock()
        mock_model.bind.return_value = mock_bound
        mock_get_model.return_value = mock_model

        mock_response = MagicMock()
        mock_response.content = '{"ok": true}'
        mock_response.tool_calls = None
        mock_response.usage_metadata = None
        mock_bound.invoke.return_value = mock_response

        mock_hash.return_value = (b"hash", "in_hex", "out_hex")
        mock_keys = MagicMock()
        mock_keys.sign_data.return_value = "sig"
        mock_keys.get_tee_id.return_value = "abc"
        mock_tee_keys.return_value = mock_keys

        return mock_model, mock_bound

    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_system_message_injected_when_no_json_keyword(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash
    ):
        """SystemMessage 'Respond in JSON format.' is prepended when no message contains 'json'."""
        from tee_gateway.controllers.chat_controller import _create_non_streaming_response
        from langchain_core.messages import HumanMessage, SystemMessage

        mock_model, mock_bound = self._setup_non_streaming(
            mock_get_model, mock_convert, mock_tee_keys, mock_hash
        )
        mock_convert.return_value = [HumanMessage(content="Tell me about a fictional person.")]

        req = CreateChatCompletionRequest(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Tell me about a fictional person."}],
            temperature=1.0,
            response_format={"type": "json_object"},
        )
        _create_non_streaming_response(req)

        called_messages = mock_bound.invoke.call_args[0][0]
        self.assertIsInstance(called_messages[0], SystemMessage)
        self.assertIn("json", called_messages[0].content.lower())

    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_system_message_not_injected_when_json_already_present(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash
    ):
        """No injection when a user message already contains the word 'json'."""
        from tee_gateway.controllers.chat_controller import _create_non_streaming_response
        from langchain_core.messages import HumanMessage, SystemMessage

        mock_model, mock_bound = self._setup_non_streaming(
            mock_get_model, mock_convert, mock_tee_keys, mock_hash
        )
        mock_convert.return_value = [
            HumanMessage(content="Reply with json data about a person.")
        ]

        req = CreateChatCompletionRequest(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Reply with json data about a person."}],
            temperature=1.0,
            response_format={"type": "json_object"},
        )
        _create_non_streaming_response(req)

        called_messages = mock_bound.invoke.call_args[0][0]
        # The first message should be the original HumanMessage, not an injected SystemMessage
        self.assertIsInstance(called_messages[0], HumanMessage)

    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_json_keyword_check_is_case_insensitive(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash
    ):
        """'JSON' (uppercase) in an existing message suppresses injection."""
        from tee_gateway.controllers.chat_controller import _create_non_streaming_response
        from langchain_core.messages import HumanMessage, SystemMessage

        mock_model, mock_bound = self._setup_non_streaming(
            mock_get_model, mock_convert, mock_tee_keys, mock_hash
        )
        mock_convert.return_value = [HumanMessage(content="Return a JSON object.")]

        req = CreateChatCompletionRequest(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Return a JSON object."}],
            temperature=1.0,
            response_format={"type": "json_object"},
        )
        _create_non_streaming_response(req)

        called_messages = mock_bound.invoke.call_args[0][0]
        self.assertIsInstance(called_messages[0], HumanMessage)

    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_injection_does_not_occur_for_json_schema_mode(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash
    ):
        """json_schema mode never triggers keyword injection."""
        from tee_gateway.controllers.chat_controller import _create_non_streaming_response
        from langchain_core.messages import HumanMessage, SystemMessage

        mock_model, mock_bound = self._setup_non_streaming(
            mock_get_model, mock_convert, mock_tee_keys, mock_hash
        )
        mock_convert.return_value = [HumanMessage(content="Tell me about a person.")]

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
        req = CreateChatCompletionRequest(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Tell me about a person."}],
            temperature=1.0,
            response_format=rf,
        )
        _create_non_streaming_response(req)

        called_messages = mock_bound.invoke.call_args[0][0]
        self.assertIsInstance(called_messages[0], HumanMessage)

    @patch("tee_gateway.controllers.chat_controller.get_provider_from_model")
    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_injection_in_streaming_path(
        self, mock_get_model, mock_convert, mock_tee_keys, mock_hash, mock_provider
    ):
        """The keyword injection also fires in the streaming path."""
        from tee_gateway.controllers.chat_controller import _create_streaming_response
        from langchain_core.messages import HumanMessage, SystemMessage

        mock_provider.return_value = "openai"
        mock_model = MagicMock()
        mock_bound = MagicMock()
        mock_model.bind.return_value = mock_bound
        mock_get_model.return_value = mock_model

        mock_convert.return_value = [HumanMessage(content="Tell me about a fictional person.")]

        mock_hash.return_value = (b"hash", "in_hex", "out_hex")
        mock_keys = MagicMock()
        mock_keys.sign_data.return_value = "sig"
        mock_keys.get_tee_id.return_value = "abc"
        mock_tee_keys.return_value = mock_keys

        # Streaming: mock_bound.stream() must return an iterable
        mock_chunk = MagicMock()
        mock_chunk.content = '{"ok": true}'
        mock_chunk.tool_call_chunks = []
        mock_chunk.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        mock_bound.stream.return_value = iter([mock_chunk])

        req = CreateChatCompletionRequest(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Tell me about a fictional person."}],
            temperature=1.0,
            stream=True,
            response_format={"type": "json_object"},
        )
        resp = _create_streaming_response(req)
        # Consume the generator to trigger the streaming logic
        list(resp.response)

        called_messages = mock_bound.stream.call_args[0][0]
        self.assertIsInstance(called_messages[0], SystemMessage)
        self.assertIn("json", called_messages[0].content.lower())


class TestAnthropicUsageMetadataPreservation(unittest.TestCase):
    """Tests that usage_metadata is carried through _invoke_anthropic_structured.

    Without this, the x402 cost calculator has no token counts to work with
    and cannot charge the session for Anthropic non-streaming requests.
    """

    def test_usage_metadata_copied_from_raw_message(self):
        """usage_metadata from the raw Anthropic AIMessage is preserved on the return value."""
        from tee_gateway.controllers.chat_controller import _invoke_anthropic_structured
        from langchain_core.messages import AIMessage

        raw_msg = AIMessage(
            content='{"name": "Alice", "age": 30}',
            usage_metadata={"input_tokens": 42, "output_tokens": 17, "total_tokens": 59},
        )

        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {
            "raw": raw_msg,
            "parsed": {"name": "Alice", "age": 30},
            "parsing_error": None,
        }

        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                },
            },
        }
        result = _invoke_anthropic_structured(mock_model, rf, [])

        self.assertIsNotNone(result.usage_metadata)
        self.assertEqual(result.usage_metadata["input_tokens"], 42)
        self.assertEqual(result.usage_metadata["output_tokens"], 17)
        self.assertEqual(result.usage_metadata["total_tokens"], 59)

    def test_no_usage_metadata_when_raw_has_none(self):
        """Returned AIMessage has no usage_metadata when the raw message has none."""
        from tee_gateway.controllers.chat_controller import _invoke_anthropic_structured
        from langchain_core.messages import AIMessage

        raw_msg = AIMessage(content='{"x": 1}')  # no usage_metadata

        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {
            "raw": raw_msg,
            "parsed": {"x": 1},
            "parsing_error": None,
        }

        rf = {
            "type": "json_schema",
            "json_schema": {"name": "out", "schema": {"type": "object"}},
        }
        result = _invoke_anthropic_structured(mock_model, rf, [])

        # usage_metadata should be falsy (None or empty dict) — not set from raw
        self.assertFalse(result.usage_metadata)

    @patch("tee_gateway.controllers.chat_controller.compute_tee_msg_hash")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.convert_messages")
    @patch("tee_gateway.controllers.chat_controller.get_provider_from_model")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    def test_non_streaming_response_includes_usage_dict(
        self, mock_get_model, mock_provider, mock_convert, mock_tee_keys, mock_hash
    ):
        """The non-streaming response body contains a 'usage' dict when Anthropic returns token counts."""
        from tee_gateway.controllers.chat_controller import _create_non_streaming_response
        from langchain_core.messages import AIMessage

        mock_provider.return_value = "anthropic"
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured

        raw_msg = AIMessage(
            content='{"name": "Bob", "age": 25}',
            usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
        )
        mock_structured.invoke.return_value = {
            "raw": raw_msg,
            "parsed": {"name": "Bob", "age": 25},
            "parsing_error": None,
        }
        mock_get_model.return_value = mock_model
        mock_convert.return_value = []

        mock_hash.return_value = (b"hash", "in_hex", "out_hex")
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
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    "required": ["name", "age"],
                    "additionalProperties": False,
                },
            },
        }
        req = CreateChatCompletionRequest(
            model="claude-sonnet-4-5",
            messages=[],
            temperature=1.0,
            response_format=rf,
        )
        response = _create_non_streaming_response(req)

        self.assertIn("usage", response)
        self.assertEqual(response["usage"]["prompt_tokens"], 50)
        self.assertEqual(response["usage"]["completion_tokens"], 20)
        self.assertEqual(response["usage"]["total_tokens"], 70)


if __name__ == "__main__":
    unittest.main()
