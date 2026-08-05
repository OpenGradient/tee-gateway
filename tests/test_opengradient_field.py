"""Verify the `opengradient` cost block is embedded on responses.

These tests are the only thing keeping `compute_session_cost`'s result from
silently going missing on a controller response — if that block is absent,
x402's `_session_cost_calculator` swallows the error and the client is never
charged. The runtime CRITICAL log is the safety net; this is the unit-test
catch.
"""

import json
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from tee_gateway.models.create_chat_completion_request import (
    CreateChatCompletionRequest,
)
from tee_gateway.models import ChatCompletionRequestUserMessage
from tee_gateway.models.create_completion_request import CreateCompletionRequest
from tee_gateway.pricing import SessionCost


_FAKE_USAGE = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def _fake_cost() -> SessionCost:
    return SessionCost(
        cost_opg=12345,
        cost_usd=Decimal("0.001"),
        opg_price_usd=Decimal("0.5"),
    )


def _fake_tee_keys() -> MagicMock:
    keys = MagicMock()
    keys.sign_data.return_value = "0xsig"
    keys.get_tee_id.return_value = "deadbeef"
    return keys


def _chat_request() -> CreateChatCompletionRequest:
    return CreateChatCompletionRequest(
        model="gpt-4.1-mini",
        messages=[ChatCompletionRequestUserMessage(role="user", content="hi")],
        stream=False,
    )


class TestChatNonStreamingOpengradient(unittest.TestCase):
    def test_opengradient_block_embedded_when_cost_computed(self):
        from tee_gateway.controllers import chat_controller

        fake_response = MagicMock()
        fake_response.content = "hello"
        fake_response.tool_calls = None
        fake_model = MagicMock()
        fake_model.invoke.return_value = fake_response

        with (
            patch.object(
                chat_controller, "get_chat_model_cached", return_value=fake_model
            ),
            patch.object(chat_controller, "extract_usage", return_value=_FAKE_USAGE),
            patch.object(
                chat_controller, "compute_session_cost", return_value=_fake_cost()
            ),
            patch.object(
                chat_controller, "get_tee_keys", return_value=_fake_tee_keys()
            ),
            patch.object(
                chat_controller,
                "compute_tee_msg_hash",
                return_value=(b"h", "ih", "oh"),
            ),
        ):
            resp = chat_controller._create_non_streaming_response(_chat_request())

        self.assertIsInstance(resp, dict)
        self.assertIn("opengradient", resp)
        self.assertEqual(resp["opengradient"]["cost_opg"], "12345")

    def test_xai_non_streaming_disables_provider_streaming(self):
        from tee_gateway.controllers import chat_controller

        fake_response = MagicMock()
        fake_response.content = "hello"
        fake_response.tool_calls = None
        fake_model = MagicMock()
        fake_model.invoke.return_value = fake_response
        request = _chat_request()
        request.model = "grok-4.3"

        with (
            patch.object(
                chat_controller, "get_chat_model_cached", return_value=fake_model
            ),
            patch.object(
                chat_controller, "get_provider_from_model", return_value="x-ai"
            ),
            patch.object(chat_controller, "extract_usage", return_value=_FAKE_USAGE),
            patch.object(
                chat_controller, "compute_session_cost", return_value=_fake_cost()
            ),
            patch.object(
                chat_controller, "get_tee_keys", return_value=_fake_tee_keys()
            ),
            patch.object(
                chat_controller,
                "compute_tee_msg_hash",
                return_value=(b"h", "ih", "oh"),
            ),
        ):
            chat_controller._create_non_streaming_response(request)

        fake_model.invoke.assert_called_once()
        self.assertFalse(fake_model.invoke.call_args.kwargs["stream"])

    def test_opengradient_block_absent_when_compute_returns_none(self):
        from tee_gateway.controllers import chat_controller

        fake_response = MagicMock()
        fake_response.content = "hello"
        fake_response.tool_calls = None
        fake_model = MagicMock()
        fake_model.invoke.return_value = fake_response

        with (
            patch.object(
                chat_controller, "get_chat_model_cached", return_value=fake_model
            ),
            patch.object(chat_controller, "extract_usage", return_value=_FAKE_USAGE),
            patch.object(chat_controller, "compute_session_cost", return_value=None),
            patch.object(
                chat_controller, "get_tee_keys", return_value=_fake_tee_keys()
            ),
            patch.object(
                chat_controller,
                "compute_tee_msg_hash",
                return_value=(b"h", "ih", "oh"),
            ),
        ):
            resp = chat_controller._create_non_streaming_response(_chat_request())

        self.assertNotIn("opengradient", resp)


class TestChatStreamingOpengradient(unittest.TestCase):
    def test_final_sse_event_carries_opengradient(self):
        from tee_gateway.controllers import chat_controller

        chunk = MagicMock()
        chunk.content = "hello"
        chunk.tool_call_chunks = []
        chunk.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }
        fake_model = MagicMock()
        fake_model.stream.return_value = iter([chunk])

        with (
            patch.object(
                chat_controller, "get_chat_model_cached", return_value=fake_model
            ),
            patch.object(
                chat_controller, "get_provider_from_model", return_value="openai"
            ),
            patch.object(
                chat_controller, "compute_session_cost", return_value=_fake_cost()
            ),
            patch.object(
                chat_controller, "get_tee_keys", return_value=_fake_tee_keys()
            ),
            patch.object(
                chat_controller,
                "compute_tee_msg_hash",
                return_value=(b"h", "ih", "oh"),
            ),
        ):
            req = _chat_request()
            req.stream = True
            response = chat_controller._create_streaming_response(req)
            chunks = [
                c.decode() if isinstance(c, bytes) else c for c in response.response
            ]

        # The final SSE data event before [DONE] carries the opengradient block.
        data_events = [
            c for c in chunks if c.startswith("data: ") and "[DONE]" not in c
        ]
        final = json.loads(data_events[-1][len("data: ") :].strip())
        self.assertIn("opengradient", final)
        self.assertEqual(final["opengradient"]["cost_opg"], "12345")

    def test_xai_stream_keeps_latest_cumulative_usage_snapshot(self):
        # xAI reports cumulative usage on every chunk; the final usage must be
        # the last snapshot, not the sum of all snapshots.
        from tee_gateway.controllers import chat_controller

        first = MagicMock()
        first.content = "hel"
        first.tool_call_chunks = []
        first.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        second = MagicMock()
        second.content = "lo"
        second.tool_call_chunks = []
        second.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }
        fake_model = MagicMock()
        fake_model.stream.return_value = iter([first, second])

        with (
            patch.object(
                chat_controller, "get_chat_model_cached", return_value=fake_model
            ),
            patch.object(
                chat_controller, "get_provider_from_model", return_value="x-ai"
            ),
            patch.object(
                chat_controller, "compute_session_cost", return_value=_fake_cost()
            ),
            patch.object(
                chat_controller, "get_tee_keys", return_value=_fake_tee_keys()
            ),
            patch.object(
                chat_controller,
                "compute_tee_msg_hash",
                return_value=(b"h", "ih", "oh"),
            ),
        ):
            request = _chat_request()
            request.model = "grok-4.3"
            request.stream = True
            response = chat_controller._create_streaming_response(request)
            chunks = [
                c.decode() if isinstance(c, bytes) else c for c in response.response
            ]

        data_events = [
            c for c in chunks if c.startswith("data: ") and "[DONE]" not in c
        ]
        final = json.loads(data_events[-1][len("data: ") :].strip())
        self.assertEqual(
            final["usage"],
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )


class TestCompletionsOpengradient(unittest.TestCase):
    def test_opengradient_block_embedded_on_completion(self):
        from tee_gateway.controllers import completions_controller

        fake_response = MagicMock()
        fake_response.content = "world"
        fake_model = MagicMock()
        fake_model.invoke.return_value = fake_response

        fake_request = MagicMock()
        fake_request.is_json = True
        fake_request.get_json.return_value = {
            "model": "gpt-4.1-mini",
            "prompt": "hi",
        }

        body = CreateCompletionRequest(model="gpt-4.1-mini", prompt="hi")

        with (
            patch("connexion.request", fake_request),
            patch.object(
                completions_controller,
                "get_chat_model_cached",
                return_value=fake_model,
            ),
            patch.object(
                completions_controller, "extract_usage", return_value=_FAKE_USAGE
            ),
            patch.object(
                completions_controller,
                "compute_session_cost",
                return_value=_fake_cost(),
            ),
            patch.object(
                completions_controller,
                "get_tee_keys",
                return_value=_fake_tee_keys(),
            ),
            patch.object(
                completions_controller,
                "compute_tee_msg_hash",
                return_value=(b"h", "ih", "oh"),
            ),
        ):
            resp = completions_controller.create_completion(body)

        self.assertIsInstance(resp, dict)
        self.assertIn("opengradient", resp)
        self.assertEqual(resp["opengradient"]["cost_opg"], "12345")

    def test_xai_completion_disables_provider_streaming(self):
        from tee_gateway.controllers import completions_controller

        fake_response = MagicMock()
        fake_response.content = "world"
        fake_model = MagicMock()
        fake_model.invoke.return_value = fake_response

        fake_request = MagicMock()
        fake_request.is_json = True
        fake_request.get_json.return_value = {
            "model": "grok-4-fast",
            "prompt": "hi",
        }

        body = CreateCompletionRequest(model="grok-4-fast", prompt="hi")

        with (
            patch("connexion.request", fake_request),
            patch.object(
                completions_controller,
                "get_chat_model_cached",
                return_value=fake_model,
            ),
            patch.object(
                completions_controller, "extract_usage", return_value=_FAKE_USAGE
            ),
            patch.object(
                completions_controller,
                "compute_session_cost",
                return_value=_fake_cost(),
            ),
            patch.object(
                completions_controller,
                "get_tee_keys",
                return_value=_fake_tee_keys(),
            ),
            patch.object(
                completions_controller,
                "compute_tee_msg_hash",
                return_value=(b"h", "ih", "oh"),
            ),
        ):
            completions_controller.create_completion(body)

        fake_model.invoke.assert_called_once()
        self.assertFalse(fake_model.invoke.call_args.kwargs["stream"])


class TestSessionCostCalculatorMissingBlock(unittest.TestCase):
    def test_critical_log_when_opengradient_missing(self):
        from tee_gateway import __main__ as gateway_main

        with self.assertLogs(gateway_main.logger, level="CRITICAL") as cm:
            with self.assertRaises(Exception):
                gateway_main._session_cost_calculator(
                    {"response_json": {"id": "chatcmpl-x"}}
                )

        self.assertTrue(
            any("opengradient cost block missing" in msg for msg in cm.output),
            f"expected CRITICAL about missing opengradient, got: {cm.output}",
        )


if __name__ == "__main__":
    unittest.main()
