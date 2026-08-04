"""Live provider usage and billing integration tests.

These tests make real, billable requests to one inexpensive chat model from
each supported provider. They are excluded from normal test runs unless
explicitly enabled:

    RUN_PROVIDER_INTEGRATION_TESTS=1 OPENAI_API_KEY=... ... \
        uv run --group test pytest \
        tee_gateway/test/test_provider_usage_integration.py -v

The tests exercise the same chat-controller and OHTTP billing projection paths
used by the gateway. They do not exercise HPKE or x402 settlement.
"""

from __future__ import annotations

import json
import os
import unittest
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast
from unittest.mock import MagicMock, patch

from tee_gateway.config import ProviderConfig
from tee_gateway.controllers import chat_controller, ohttp_controller
from tee_gateway.llm_backend import set_provider_config
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


PROVIDERS = (
    _ProviderCase("OpenAI", "gpt-4.1-nano", "OPENAI_API_KEY"),
    _ProviderCase("Anthropic", "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    _ProviderCase("Google", "gemini-3.5-flash-lite", "GOOGLE_API_KEY"),
    _ProviderCase("xAI", "grok-4-fast", "XAI_API_KEY"),
    _ProviderCase("ByteDance", "deepseek-v4-flash", "ARK_API_KEY"),
    _ProviderCase("Nous", "hermes-4-70b", "NOUS_API_KEY"),
    _ProviderCase("Z.ai", "glm-5.2", "ZAI_API_KEY"),
)

_missing_secrets = [
    case.secret_name for case in PROVIDERS if not os.getenv(case.secret_name)
]
if _missing_secrets:
    raise RuntimeError(
        "Missing provider API keys: " + ", ".join(sorted(_missing_secrets))
    )


class _FixedPriceFeed:
    """Deterministic OPG/USD price for testing cost projection."""

    def get_price(self) -> Decimal:
        return Decimal("0.20")


def _request(*, model: str, stream: bool) -> CreateChatCompletionRequest:
    return CreateChatCompletionRequest(
        model=model,
        messages=[
            ChatCompletionRequestUserMessage(
                role="user",
                content="Reply with exactly: OK",
            )
        ],
        max_tokens=128,
        temperature=0,
        stream=stream,
    )


def _assert_cost_block(test: unittest.TestCase, response: dict) -> None:
    usage = cast(dict[str, Any], response.get("usage"))
    test.assertIsInstance(usage, dict, response)
    test.assertGreater(usage["prompt_tokens"], 0)
    test.assertGreater(usage["completion_tokens"], 0)
    test.assertGreaterEqual(
        usage["total_tokens"],
        usage["prompt_tokens"] + usage["completion_tokens"],
    )

    cost = cast(dict[str, Any], response.get("opengradient"))
    test.assertIsInstance(cost, dict, response)
    for field in ("cost_opg", "cost_usd", "opg_price_usd"):
        test.assertIn(field, cost, response)
        test.assertNotIn(cost[field], (None, ""), response)
    test.assertGreater(int(cost["cost_opg"]), 0)
    test.assertGreater(Decimal(cost["cost_usd"]), 0)
    test.assertGreater(Decimal(cost["opg_price_usd"]), 0)


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
                nous_api_key=os.environ["NOUS_API_KEY"],
                zai_api_key=os.environ["ZAI_API_KEY"],
            )
        )
        set_price_feed(_FixedPriceFeed())  # type: ignore[arg-type]

    def setUp(self) -> None:
        self.tee_keys = MagicMock()
        self.tee_keys.sign_data.return_value = "integration-test-signature"
        self.tee_keys.get_tee_id.return_value = "00" * 32

    def test_non_streaming_usage_and_cost_for_every_provider(self) -> None:
        for case in PROVIDERS:
            with self.subTest(provider=case.provider, model=case.model):
                with patch.object(
                    chat_controller,
                    "get_tee_keys",
                    return_value=self.tee_keys,
                ):
                    response = chat_controller._create_non_streaming_response(
                        _request(model=case.model, stream=False)
                    )

                self.assertIsInstance(response, dict, response)
                self.assertNotIn("error", response)
                _assert_cost_block(self, response)

                headers = ohttp_controller._extract_cost_headers(
                    json.dumps(response).encode("utf-8")
                )
                self.assertEqual(
                    headers["X-Inference-Cost-USD"],
                    response["opengradient"]["cost_usd"],
                )
                self.assertEqual(
                    headers["X-Inference-Cost-OPG"],
                    response["opengradient"]["cost_opg"],
                )

    def test_streaming_usage_and_cost_for_every_provider(self) -> None:
        for case in PROVIDERS:
            with self.subTest(provider=case.provider, model=case.model):
                with patch.object(
                    chat_controller,
                    "get_tee_keys",
                    return_value=self.tee_keys,
                ):
                    response = chat_controller._create_streaming_response(
                        _request(model=case.model, stream=True)
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
                _assert_cost_block(self, final)

                billing_frame = ohttp_controller._build_billing_frame(final)
                self.assertTrue(
                    billing_frame.startswith(ohttp_controller.OHTTP_BILLING_FRAME_MAGIC)
                )


if __name__ == "__main__":
    unittest.main()
