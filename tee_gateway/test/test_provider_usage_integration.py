"""Live provider response-parsing, usage, and billing integration tests.

These tests make real, billable requests to representative baseline, newer,
and image models. They are excluded from normal test runs unless enabled:

    RUN_PROVIDER_INTEGRATION_TESTS=1 OPENAI_API_KEY=... ... \
        uv run --group test pytest \
        tee_gateway/test/test_provider_usage_integration.py -v

The tests exercise the same chat-controller and OHTTP billing projection paths
used by the gateway. They do not exercise HPKE or x402 settlement. A complete
run currently makes text and image requests in both response modes and can be
materially more expensive than the unit suite.
"""

from __future__ import annotations

import base64
import json
import os
import unittest
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast
from unittest.mock import MagicMock, patch

from tee_gateway import image_generation
from tee_gateway.config import ProviderConfig
from tee_gateway.controllers import chat_controller, ohttp_controller
from tee_gateway.llm_backend import set_provider_config
from tee_gateway.model_registry import get_model_config
from tee_gateway.models import ChatCompletionRequestUserMessage
from tee_gateway.models.create_chat_completion_request import (
    CreateChatCompletionRequest,
)
from tee_gateway.price_feed import set_price_feed

if os.getenv("RUN_PROVIDER_INTEGRATION_TESTS") != "1":
    raise unittest.SkipTest(
        "Set RUN_PROVIDER_INTEGRATION_TESTS=1 to run live provider tests"
    )


@dataclass(frozen=True)
class _ProviderCase:
    provider: str
    model: str
    secret_name: str


PROVIDER_SMOKE_MODELS = (
    _ProviderCase("OpenAI", "gpt-4.1-nano", "OPENAI_API_KEY"),
    _ProviderCase("Anthropic", "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    _ProviderCase("Google", "gemini-3.5-flash-lite", "GOOGLE_API_KEY"),
    _ProviderCase("xAI", "grok-4-fast", "XAI_API_KEY"),
    _ProviderCase("ByteDance", "deepseek-v4-flash", "ARK_API_KEY"),
    _ProviderCase("OpenRouter", "hermes-4-70b", "OPENROUTER_API_KEY"),
    _ProviderCase("Z.ai via ModelArk", "glm-5.2", "ARK_API_KEY"),
)

# The newest model per provider — one each, so the run stays cheap. Replace a
# provider's entry when a newer model is registered, so a dispatch of this
# suite exercises what actually shipped rather than last quarter's flagship.
NEW_CHAT_MODELS = (
    _ProviderCase("OpenAI", "gpt-6-astra", "OPENAI_API_KEY"),
    _ProviderCase("Anthropic", "claude-fable-5-1", "ANTHROPIC_API_KEY"),
    _ProviderCase("Google", "gemini-3.8-flash", "GOOGLE_API_KEY"),
    _ProviderCase("xAI", "grok-4.5", "XAI_API_KEY"),
    _ProviderCase("ByteDance", "seed-2.0-lite", "ARK_API_KEY"),
    _ProviderCase("OpenRouter", "hy3", "OPENROUTER_API_KEY"),
)

IMAGE_MODELS = (
    _ProviderCase("OpenAI", "gpt-image-2", "OPENAI_API_KEY"),
    _ProviderCase("Google", "gemini-3.1-flash-image", "GOOGLE_API_KEY"),
    _ProviderCase("xAI", "grok-imagine-image", "XAI_API_KEY"),
    _ProviderCase("ByteDance", "seedream-5.0-lite", "ARK_API_KEY"),
)

CHAT_MODELS = PROVIDER_SMOKE_MODELS + NEW_CHAT_MODELS
ALL_LIVE_MODELS = CHAT_MODELS + IMAGE_MODELS

_missing_secrets = sorted(
    {case.secret_name for case in ALL_LIVE_MODELS if not os.getenv(case.secret_name)}
)
if _missing_secrets:
    raise RuntimeError("Missing provider API keys: " + ", ".join(_missing_secrets))


class _FixedPriceFeed:
    """Deterministic OPG/USD price for testing cost projection."""

    def get_price(self) -> Decimal:
        return Decimal("0.20")


def _chat_request(*, model: str, stream: bool) -> CreateChatCompletionRequest:
    return CreateChatCompletionRequest(
        model=model,
        messages=[
            ChatCompletionRequestUserMessage(
                role="user",
                content="Reply with exactly: OK",
            )
        ],
        max_tokens=512,
        temperature=0,
        stream=stream,
    )


def _image_request(*, model: str, stream: bool) -> CreateChatCompletionRequest:
    return CreateChatCompletionRequest(
        model=model,
        messages=[
            ChatCompletionRequestUserMessage(
                role="user",
                content="Create a simple solid red square on a white background.",
            )
        ],
        max_tokens=4096,
        temperature=0,
        stream=stream,
    )


def _assert_positive_cost(test: unittest.TestCase, response: dict) -> dict[str, Any]:
    cost = cast(dict[str, Any], response.get("opengradient"))
    test.assertIsInstance(cost, dict, response)
    for field in ("cost_opg", "cost_usd", "opg_price_usd"):
        test.assertIn(field, cost, response)
        test.assertNotIn(cost[field], (None, ""), response)

    cost_opg = int(cost["cost_opg"])
    cost_usd = Decimal(cost["cost_usd"])
    opg_price_usd = Decimal(cost["opg_price_usd"])
    test.assertGreater(cost_opg, 0)
    test.assertGreater(cost_usd, 0)
    test.assertGreater(opg_price_usd, 0)
    test.assertEqual(
        cost_usd,
        (Decimal(cost_opg) / (Decimal(10) ** 18)) * opg_price_usd,
    )
    return cost


def _assert_usage_and_cost(test: unittest.TestCase, response: dict, model: str) -> None:
    usage = cast(dict[str, Any], response.get("usage"))
    test.assertIsInstance(usage, dict, response)
    test.assertGreater(usage["prompt_tokens"], 0)
    test.assertGreater(usage["completion_tokens"], 0)
    test.assertGreaterEqual(
        usage["total_tokens"],
        usage["prompt_tokens"] + usage["completion_tokens"],
    )

    cost = _assert_positive_cost(test, response)
    cfg = get_model_config(model)
    raw_token_cost = (
        Decimal(usage["prompt_tokens"]) * cfg.input_price_usd
        + Decimal(usage["completion_tokens"]) * cfg.output_price_usd
    )
    settled_cost = Decimal(cost["cost_usd"])
    test.assertGreaterEqual(settled_cost, raw_token_cost)
    test.assertLess(settled_cost - raw_token_cost, Decimal("0.000000000001"))


def _assert_cost_headers(test: unittest.TestCase, response: dict) -> None:
    headers = ohttp_controller._extract_cost_headers(
        json.dumps(response).encode("utf-8")
    )
    test.assertEqual(
        headers["X-Inference-Cost-USD"], response["opengradient"]["cost_usd"]
    )
    test.assertEqual(
        headers["X-Inference-Cost-OPG"], response["opengradient"]["cost_opg"]
    )


def _assert_images(test: unittest.TestCase, images: Any) -> None:
    test.assertIsInstance(images, list)
    test.assertGreater(len(images), 0, images)
    for image in images:
        test.assertIsInstance(image, str)
        header, separator, encoded = image.partition(",")
        test.assertTrue(separator, image[:100])
        test.assertTrue(header.startswith("data:image/"), header)
        test.assertIn(";base64", header)
        test.assertGreater(len(base64.b64decode(encoded, validate=True)), 0)


def _assert_image_usage_and_cost(
    test: unittest.TestCase, response: dict, model: str
) -> None:
    usage = cast(dict[str, Any], response.get("usage"))
    test.assertIsInstance(usage, dict, response)
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        test.assertIn(field, usage, response)

    cfg = get_model_config(model)
    if cfg.image_output:
        test.assertGreater(usage["prompt_tokens"], 0)
        test.assertGreater(usage["completion_tokens"], 0)
        test.assertGreaterEqual(
            usage["total_tokens"],
            usage["prompt_tokens"] + usage["completion_tokens"],
        )
        _assert_positive_cost(test, response)
        return

    test.assertTrue(cfg.image_generation)
    test.assertEqual(
        usage, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    cost = _assert_positive_cost(test, response)
    test.assertIsNotNone(cfg.per_image_price_usd)
    test.assertEqual(Decimal(cost["cost_usd"]), cfg.per_image_price_usd)


def _stream_events(response) -> list[dict]:
    payloads: list[str] = []
    for chunk in response.response:
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload:
                payloads.append(payload)

    if payloads and payloads[-1] != "[DONE]":
        try:
            error_event = json.loads(payloads[-1])
        except (TypeError, ValueError):
            error_event = None
        if isinstance(error_event, dict) and error_event.get("error"):
            exception_type = error_event.get("exception_type", "ProviderError")
            raise AssertionError(
                f"Streaming provider error ({exception_type}): {error_event['error']}"
            )
    if not payloads or payloads[-1] != "[DONE]":
        raise AssertionError("Streaming response must end with [DONE]")
    if "[DONE]" in payloads[:-1]:
        raise AssertionError("Streaming response emitted [DONE] before the end")
    return [json.loads(payload) for payload in payloads[:-1]]


class TestLiveProviderUsageBilling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_provider_config(
            ProviderConfig(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
                google_api_key=os.environ["GOOGLE_API_KEY"],
                xai_api_key=os.environ["XAI_API_KEY"],
                bytedance_api_key=os.environ["ARK_API_KEY"],
                openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
            )
        )
        set_price_feed(_FixedPriceFeed())  # type: ignore[arg-type]

    def setUp(self) -> None:
        self.tee_keys = MagicMock()
        self.tee_keys.sign_data.return_value = "integration-test-signature"
        self.tee_keys.get_tee_id.return_value = "00" * 32

    def test_non_streaming_usage_and_cost_for_chat_models(self) -> None:
        for case in CHAT_MODELS:
            with self.subTest(provider=case.provider, model=case.model):
                with patch.object(
                    chat_controller,
                    "get_tee_keys",
                    return_value=self.tee_keys,
                ):
                    response = chat_controller._create_non_streaming_response(
                        _chat_request(model=case.model, stream=False)
                    )

                self.assertIsInstance(response, dict, response)
                self.assertNotIn("error", response)
                self.assertTrue(response["choices"][0]["message"]["content"])
                _assert_usage_and_cost(self, response, case.model)
                _assert_cost_headers(self, response)

    def test_streaming_usage_and_cost_for_chat_models(self) -> None:
        for case in CHAT_MODELS:
            with self.subTest(provider=case.provider, model=case.model):
                with patch.object(
                    chat_controller,
                    "get_tee_keys",
                    return_value=self.tee_keys,
                ):
                    response = chat_controller._create_streaming_response(
                        _chat_request(model=case.model, stream=True)
                    )
                    events = _stream_events(response)

                self.assertGreaterEqual(len(events), 2, events)
                content_events = [
                    event
                    for event in events[:-1]
                    if event.get("choices", [{}])[0].get("delta", {}).get("content")
                ]
                self.assertTrue(content_events, events)
                for event in events[:-1]:
                    self.assertNotIn("usage", event)
                    self.assertNotIn("opengradient", event)

                final = events[-1]
                self.assertNotIn("error", final)
                self.assertIn("tee_signature", final)
                self.assertIsNotNone(final["choices"][0]["finish_reason"])
                _assert_usage_and_cost(self, final, case.model)

                billing_frame = ohttp_controller._build_billing_frame(final)
                self.assertTrue(
                    billing_frame.startswith(ohttp_controller.OHTTP_BILLING_FRAME_MAGIC)
                )

    def test_non_streaming_image_models_return_images_and_cost(self) -> None:
        for case in IMAGE_MODELS:
            with self.subTest(provider=case.provider, model=case.model):
                with (
                    patch.object(
                        chat_controller,
                        "get_tee_keys",
                        return_value=self.tee_keys,
                    ),
                    patch.object(
                        image_generation,
                        "get_tee_keys",
                        return_value=self.tee_keys,
                    ),
                ):
                    response = chat_controller._create_non_streaming_response(
                        _image_request(model=case.model, stream=False)
                    )

                self.assertIsInstance(response, dict, response)
                self.assertNotIn("error", response)
                _assert_images(self, response["choices"][0]["message"].get("images"))
                _assert_image_usage_and_cost(self, response, case.model)
                _assert_cost_headers(self, response)

    def test_streaming_image_models_return_images_and_billing_frame(self) -> None:
        for case in IMAGE_MODELS:
            with self.subTest(provider=case.provider, model=case.model):
                with (
                    patch.object(
                        chat_controller,
                        "get_tee_keys",
                        return_value=self.tee_keys,
                    ),
                    patch.object(
                        image_generation,
                        "get_tee_keys",
                        return_value=self.tee_keys,
                    ),
                ):
                    response = chat_controller._create_streaming_response(
                        _image_request(model=case.model, stream=True)
                    )
                    events = _stream_events(response)

                self.assertGreaterEqual(len(events), 1, events)
                for event in events[:-1]:
                    self.assertNotIn("usage", event)
                    self.assertNotIn("opengradient", event)
                    self.assertNotIn("images", event)

                final = events[-1]
                self.assertNotIn("error", final)
                self.assertIn("tee_signature", final)
                self.assertIsNotNone(final["choices"][0]["finish_reason"])
                _assert_images(self, final.get("images"))
                _assert_image_usage_and_cost(self, final, case.model)

                billing_frame = ohttp_controller._build_billing_frame(final)
                self.assertTrue(
                    billing_frame.startswith(ohttp_controller.OHTTP_BILLING_FRAME_MAGIC)
                )


if __name__ == "__main__":
    unittest.main()
