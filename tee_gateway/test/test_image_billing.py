"""Regression tests for native-image-generation billing.

Gemini image-output models (e.g. ``gemini-2.5-flash-image``) bill each generated
image as ~1290 output tokens reported in ``candidates_token_count``. Our billing
relies on langchain-google-genai folding that field into
``usage_metadata.output_tokens`` so the image rides the normal token-priced path
(``output_tokens -> completion_tokens -> output_price_usd``). These tests pin
that assumption: if a future library bump stops folding image tokens into
``output_tokens``, or our pricing stops charging them, they fail loudly.

No network or API key required — we construct a synthetic Gemini response object
and inject a stub price feed.
"""

import unittest
from decimal import Decimal
from unittest.mock import MagicMock

from google.genai.types import GenerateContentResponse
from langchain_google_genai.chat_models import _response_to_result

from tee_gateway.price_feed import get_price_feed, set_price_feed
from tee_gateway.pricing import compute_session_cost

IMAGE_MODEL = "gemini-2.5-flash-image"
# Google's fixed image-output token count for a standard 1024x1024 image.
IMAGE_TOKENS = 1290


def _gemini_image_response(
    *, candidates_tokens: int, thoughts_tokens: int = 0, prompt_tokens: int = 9
) -> GenerateContentResponse:
    """Build a synthetic Gemini response: a caption part + an inline image part,
    with image output accounted for in ``candidates_token_count``."""
    return GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Here is your image."},
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": "aGVsbG8=",  # "hello"
                                }
                            },
                        ],
                    },
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": prompt_tokens,
                "candidates_token_count": candidates_tokens,
                "thoughts_token_count": thoughts_tokens,
                "total_token_count": prompt_tokens
                + candidates_tokens
                + thoughts_tokens,
            },
        }
    )


class TestLangChainImageTokenFolding(unittest.TestCase):
    """The langchain-google-genai mapping our billing depends on."""

    def test_image_tokens_fold_into_output_tokens(self):
        resp = _gemini_image_response(candidates_tokens=IMAGE_TOKENS)
        msg = _response_to_result(resp).generations[0].message

        self.assertIsNotNone(msg.usage_metadata)
        # The 1290 image tokens land in output_tokens (-> completion_tokens).
        self.assertEqual(msg.usage_metadata["output_tokens"], IMAGE_TOKENS)
        self.assertEqual(msg.usage_metadata["input_tokens"], 9)

    def test_thought_tokens_are_added_to_output_tokens(self):
        # langchain adds thoughts_token_count on top of candidates_token_count.
        resp = _gemini_image_response(
            candidates_tokens=IMAGE_TOKENS, thoughts_tokens=50
        )
        msg = _response_to_result(resp).generations[0].message

        self.assertEqual(msg.usage_metadata["output_tokens"], IMAGE_TOKENS + 50)


class TestImageBilling(unittest.TestCase):
    """The folded output_tokens actually translate into a charge."""

    def setUp(self):
        self._prev_feed = None
        try:
            self._prev_feed = get_price_feed()
        except RuntimeError:
            self._prev_feed = None
        # 1 OPG == 1 USD keeps the OPG<->USD reconciliation arithmetic trivial.
        stub = MagicMock()
        stub.get_price.return_value = Decimal("1")
        set_price_feed(stub)

    def tearDown(self):
        if self._prev_feed is not None:
            set_price_feed(self._prev_feed)

    def _usage_dict(self, response) -> dict:
        """Mirror how chat_controller shapes usage_metadata into the OpenAI form."""
        um = response.generations[0].message.usage_metadata
        return {
            "prompt_tokens": um["input_tokens"],
            "completion_tokens": um["output_tokens"],
            "total_tokens": um["total_tokens"],
        }

    def test_generated_image_is_charged_as_output_tokens(self):
        resp = _response_to_result(
            _gemini_image_response(candidates_tokens=IMAGE_TOKENS)
        )
        cost = compute_session_cost(IMAGE_MODEL, self._usage_dict(resp))

        self.assertIsNotNone(cost)
        # Expected raw cost: 9 input + 1290 output tokens at the registry rates.
        from tee_gateway.model_registry import get_model_config

        cfg = get_model_config(IMAGE_MODEL)
        expected = (Decimal(9) * cfg.input_price_usd) + (
            Decimal(IMAGE_TOKENS) * cfg.output_price_usd
        )
        # settled_usd rounds the OPG integer up, so it is >= raw by at most one
        # smallest unit (1e-18 USD here) — assert effectively-equal.
        self.assertAlmostEqual(cost.cost_usd, expected, places=9)
        self.assertGreater(cost.cost_opg, 0)

    def test_more_images_cost_more(self):
        """Cost scales with image tokens — not a flat per-request fee."""
        one = _response_to_result(
            _gemini_image_response(candidates_tokens=IMAGE_TOKENS)
        )
        two = _response_to_result(
            _gemini_image_response(candidates_tokens=IMAGE_TOKENS * 2)
        )
        cost_one = compute_session_cost(IMAGE_MODEL, self._usage_dict(one))
        cost_two = compute_session_cost(IMAGE_MODEL, self._usage_dict(two))

        self.assertIsNotNone(cost_one)
        self.assertIsNotNone(cost_two)
        self.assertGreater(cost_two.cost_opg, cost_one.cost_opg)


if __name__ == "__main__":
    unittest.main()
