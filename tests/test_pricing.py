"""
Unit tests for dynamic pricing / cost calculation across all supported models.

Tests verify that:
  - Every user-facing model name resolves to the correct ModelConfig
  - calculate_session_cost produces the right amount in OPG token
    smallest-units for supported models
  - Edge cases (no usage, unknown model, bad context) are handled correctly
"""

import unittest
from decimal import Decimal

from tee_gateway.definitions import BASE_MAINNET_OPG_ADDRESS
from tee_gateway.model_registry import (
    _MODEL_LOOKUP,
    get_model_config,
)
from tee_gateway.util import calculate_session_cost

# All pricing tests assume OPG = $1.00 so USD cost == OPG token amount.
_OPG_PRICE_USD = Decimal("1")
_get_price = lambda: _OPG_PRICE_USD  # noqa: E731


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opg_requirements() -> dict:
    """Fake PaymentRequirements dict for OPG (18 decimals)."""
    return {"asset": BASE_MAINNET_OPG_ADDRESS, "amount": "50000000000000000"}


def _ctx(model: str, input_tokens: int, output_tokens: int, requirements=None) -> dict:
    """Build a minimal calculator context."""
    return {
        "request_json": {"model": model, "messages": []},
        "response_json": {
            "model": model,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        },
        "payment_requirements": requirements or _opg_requirements(),
    }


def _expected_cost_opg(model: str, input_tokens: int, output_tokens: int) -> int:
    """Compute expected cost in OPG smallest units (18 decimals, ROUND_CEILING)."""
    from decimal import ROUND_CEILING

    cfg = get_model_config(model)
    total_usd = (
        Decimal(input_tokens) * cfg.input_price_usd
        + Decimal(output_tokens) * cfg.output_price_usd
    )
    return int((total_usd * Decimal(10**18)).to_integral_value(rounding=ROUND_CEILING))


# ---------------------------------------------------------------------------
# Model registry tests
# ---------------------------------------------------------------------------


class TestModelRegistry(unittest.TestCase):
    """All user-facing model names must resolve without error."""

    def test_all_lookup_keys_resolve(self):
        """Every key in _MODEL_LOOKUP must resolve to a valid ModelConfig."""
        for name, enum_val in _MODEL_LOOKUP.items():
            with self.subTest(model=name):
                cfg = get_model_config(name)
                self.assertIsNotNone(cfg)
                self.assertIsNotNone(cfg.provider)
                self.assertIsNotNone(cfg.api_name)
                self.assertGreater(cfg.input_price_usd, 0)
                self.assertGreater(cfg.output_price_usd, 0)

    # ── Anthropic Sonnet ────────────────────────────────────────────────────

    def test_claude_sonnet_4_5_resolves(self):
        cfg = get_model_config("claude-sonnet-4-5")
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000003"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000015"))

    def test_claude_sonnet_4_6_resolves(self):
        cfg = get_model_config("claude-sonnet-4-6")
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000003"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000015"))

    # ── Anthropic Haiku ─────────────────────────────────────────────────────

    def test_claude_haiku_4_5_resolves(self):
        cfg = get_model_config("claude-haiku-4-5")
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000001"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000005"))

    # ── Anthropic Opus ──────────────────────────────────────────────────────

    def test_claude_opus_4_5_resolves(self):
        cfg = get_model_config("claude-opus-4-5")
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000005"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000025"))

    def test_claude_opus_4_6_resolves(self):
        cfg = get_model_config("claude-opus-4-6")
        self.assertEqual(cfg.provider, "anthropic")

    # ── OpenAI ──────────────────────────────────────────────────────────────

    def test_gpt_4_1_resolves(self):
        cfg = get_model_config("gpt-4.1")
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000002"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000008"))

    def test_gpt_4_1_full_date_resolves(self):
        cfg = get_model_config("gpt-4.1-2025-04-14")
        self.assertEqual(cfg, get_model_config("gpt-4.1"))

    def test_o4_mini_resolves(self):
        cfg = get_model_config("o4-mini")
        self.assertEqual(cfg.provider, "openai")

    def test_gpt_5_resolves(self):
        cfg = get_model_config("gpt-5")
        self.assertEqual(cfg.provider, "openai")

    def test_gpt_5_mini_resolves(self):
        cfg = get_model_config("gpt-5-mini")
        self.assertEqual(cfg.provider, "openai")

    def test_gpt_5_2_resolves(self):
        cfg = get_model_config("gpt-5.2")
        self.assertEqual(cfg.provider, "openai")

    def test_gpt_4_1_mini_resolves(self):
        cfg = get_model_config("gpt-4.1-mini")
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000004"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.0000016"))

    def test_gpt_4_1_mini_dated_resolves(self):
        cfg = get_model_config("gpt-4.1-mini-2025-04-14")
        self.assertEqual(cfg, get_model_config("gpt-4.1-mini"))

    def test_gpt_4_1_nano_resolves(self):
        cfg = get_model_config("gpt-4.1-nano")
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000001"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.0000004"))

    def test_gpt_4_1_nano_dated_resolves(self):
        cfg = get_model_config("gpt-4.1-nano-2025-04-14")
        self.assertEqual(cfg, get_model_config("gpt-4.1-nano"))

    def test_o3_resolves(self):
        cfg = get_model_config("o3")
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.force_temperature, 1.0)

    def test_o3_dated_resolves(self):
        cfg = get_model_config("o3-2025-04-16")
        self.assertEqual(cfg, get_model_config("o3"))

    def test_gpt_5_4_resolves(self):
        cfg = get_model_config("gpt-5.4")
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000025"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000015"))

    def test_gpt_5_4_mini_resolves(self):
        cfg = get_model_config("gpt-5.4-mini")
        self.assertEqual(cfg.provider, "openai")

    def test_gpt_5_4_nano_resolves(self):
        cfg = get_model_config("gpt-5.4-nano")
        self.assertEqual(cfg.provider, "openai")

    def test_gpt_5_5_resolves(self):
        cfg = get_model_config("gpt-5.5")
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000005"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.00003"))

    # ── Google ──────────────────────────────────────────────────────────────

    def test_gemini_2_5_flash_resolves(self):
        cfg = get_model_config("gemini-2.5-flash")
        self.assertEqual(cfg.provider, "google")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000003"))

    def test_gemini_2_5_pro_resolves(self):
        cfg = get_model_config("gemini-2.5-pro")
        self.assertEqual(cfg.provider, "google")

    def test_gemini_2_5_flash_lite_resolves(self):
        cfg = get_model_config("gemini-2.5-flash-lite")
        self.assertEqual(cfg.provider, "google")

    def test_gemini_3_flash_preview_resolves(self):
        cfg = get_model_config("gemini-3-flash-preview")
        self.assertEqual(cfg.provider, "google")

    def test_gemini_3_1_pro_preview_resolves(self):
        cfg = get_model_config("gemini-3.1-pro-preview")
        self.assertEqual(cfg.provider, "google")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000002"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000012"))
        self.assertEqual(cfg.thinking_budget, 128)

    def test_gemini_3_1_flash_lite_preview_resolves(self):
        cfg = get_model_config("gemini-3.1-flash-lite-preview")
        self.assertEqual(cfg.provider, "google")
        self.assertEqual(cfg.input_price_usd, Decimal("0.00000025"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.0000015"))
        self.assertEqual(cfg.thinking_budget, 0)

    # ── xAI Grok ────────────────────────────────────────────────────────────

    def test_grok_4_resolves(self):
        cfg = get_model_config("grok-4")
        self.assertEqual(cfg.provider, "x-ai")

    def test_grok_4_fast_resolves(self):
        cfg = get_model_config("grok-4-fast")
        self.assertEqual(cfg.provider, "x-ai")

    def test_grok_4_1_fast_resolves(self):
        cfg = get_model_config("grok-4-1-fast")
        self.assertEqual(cfg.provider, "x-ai")

    def test_grok_4_1_fast_dot_notation_resolves(self):
        cfg = get_model_config("grok-4.1-fast")
        self.assertEqual(cfg, get_model_config("grok-4-1-fast"))

    def test_grok_3_mini_resolves(self):
        cfg = get_model_config("grok-3-mini")
        self.assertEqual(cfg.provider, "x-ai")

    def test_grok_3_resolves(self):
        cfg = get_model_config("grok-3")
        self.assertEqual(cfg.provider, "x-ai")

    def test_grok_4_20_reasoning_resolves(self):
        cfg = get_model_config("grok-4.20-reasoning")
        self.assertEqual(cfg.provider, "x-ai")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000002"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000006"))

    def test_grok_4_20_non_reasoning_resolves(self):
        cfg = get_model_config("grok-4.20-non-reasoning")
        self.assertEqual(cfg.provider, "x-ai")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000002"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000006"))

    def test_grok_code_fast_1_resolves(self):
        cfg = get_model_config("grok-code-fast-1")
        self.assertEqual(cfg.provider, "x-ai")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000002"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.0000015"))

    def test_claude_opus_4_7_resolves(self):
        cfg = get_model_config("claude-opus-4-7")
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000005"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000025"))

    # ── Errors ───────────────────────────────────────────────────────────────

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            get_model_config("gpt-4o")  # not in registry

    def test_unknown_sonnet_variant_raises(self):
        with self.assertRaises(ValueError):
            get_model_config("claude-sonnet-99")


# ---------------------------------------------------------------------------
# Pricing calculation tests
# ---------------------------------------------------------------------------


class TestCalculateSessionCostOPG(unittest.TestCase):
    """calculate_session_cost with OPG (18 decimals)."""

    def _calc(self, model, input_tokens, output_tokens):
        return calculate_session_cost(
            _ctx(model, input_tokens, output_tokens, _opg_requirements()), _get_price
        )

    # ── OpenAI ──────────────────────────────────────────────────────────────

    def test_gpt_4_1_cost(self):
        cost = self._calc("gpt-4.1", 1000, 500)
        expected = _expected_cost_opg("gpt-4.1", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.000002 + 500*0.000008 = 0.002 + 0.004 = 0.006 USD = 6e15 wei
        self.assertEqual(cost, 6_000_000_000_000_000)

    def test_gpt_5_mini_cost(self):
        cost = self._calc("gpt-5-mini", 1000, 500)
        expected = _expected_cost_opg("gpt-5-mini", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.00000025 + 500*0.000002 = 0.00025 + 0.001 = 0.00125 USD
        self.assertEqual(cost, 1_250_000_000_000_000)

    def test_o4_mini_cost(self):
        cost = self._calc("o4-mini", 2000, 1000)
        expected = _expected_cost_opg("o4-mini", 2000, 1000)
        self.assertEqual(cost, expected)

    def test_gpt_4_1_mini_cost(self):
        cost = self._calc("gpt-4.1-mini", 1000, 500)
        expected = _expected_cost_opg("gpt-4.1-mini", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.0000004 + 500*0.0000016 = 0.0004 + 0.0008 = 0.0012 USD = 1.2e15 wei
        self.assertEqual(cost, 1_200_000_000_000_000)

    def test_gpt_4_1_nano_cost(self):
        cost = self._calc("gpt-4.1-nano", 1000, 500)
        expected = _expected_cost_opg("gpt-4.1-nano", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.0000001 + 500*0.0000004 = 0.0001 + 0.0002 = 0.0003 USD = 3e14 wei
        self.assertEqual(cost, 300_000_000_000_000)

    def test_o3_cost(self):
        cost = self._calc("o3", 1000, 500)
        expected = _expected_cost_opg("o3", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.00001 + 500*0.00004 = 0.01 + 0.02 = 0.03 USD = 3e16 wei
        self.assertEqual(cost, 30_000_000_000_000_000)

    def test_gpt_5_4_cost(self):
        cost = self._calc("gpt-5.4", 1000, 500)
        expected = _expected_cost_opg("gpt-5.4", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.0000025 + 500*0.000015 = 0.0025 + 0.0075 = 0.01 USD = 1e16 wei
        self.assertEqual(cost, 10_000_000_000_000_000)

    def test_gpt_5_4_mini_cost(self):
        cost = self._calc("gpt-5.4-mini", 1000, 500)
        expected = _expected_cost_opg("gpt-5.4-mini", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.00000075 + 500*0.0000045 = 0.00075 + 0.00225 = 0.003 USD = 3e15 wei
        self.assertEqual(cost, 3_000_000_000_000_000)

    def test_gpt_5_4_nano_cost(self):
        cost = self._calc("gpt-5.4-nano", 1000, 500)
        expected = _expected_cost_opg("gpt-5.4-nano", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.0000002 + 500*0.00000125 = 0.0002 + 0.000625 = 0.000825 USD = 8.25e14 wei
        self.assertEqual(cost, 825_000_000_000_000)

    def test_gpt_5_5_cost(self):
        cost = self._calc("gpt-5.5", 1000, 500)
        expected = _expected_cost_opg("gpt-5.5", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.000005 + 500*0.00003 = 0.005 + 0.015 = 0.02 USD = 2e16 wei
        self.assertEqual(cost, 20_000_000_000_000_000)

    # ── Anthropic Sonnet ────────────────────────────────────────────────────

    def test_claude_sonnet_4_5_cost(self):
        cost = self._calc("claude-sonnet-4-5", 1000, 500)
        expected = _expected_cost_opg("claude-sonnet-4-5", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.000003 + 500*0.000015 = 0.003 + 0.0075 = 0.0105 USD = 10.5e15 wei
        self.assertEqual(cost, 10_500_000_000_000_000)

    def test_claude_sonnet_4_6_cost(self):
        cost = self._calc("claude-sonnet-4-6", 1000, 500)
        self.assertEqual(cost, self._calc("claude-sonnet-4-5", 1000, 500))

    # ── Anthropic Haiku ─────────────────────────────────────────────────────

    def test_claude_haiku_4_5_cost(self):
        cost = self._calc("claude-haiku-4-5", 1000, 500)
        expected = _expected_cost_opg("claude-haiku-4-5", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.000001 + 500*0.000005 = 0.001 + 0.0025 = 0.0035 USD = 3.5e15 wei
        self.assertEqual(cost, 3_500_000_000_000_000)

    # ── Anthropic Opus ──────────────────────────────────────────────────────

    def test_claude_opus_4_5_cost(self):
        cost = self._calc("claude-opus-4-5", 1000, 500)
        expected = _expected_cost_opg("claude-opus-4-5", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.000005 + 500*0.000025 = 0.005 + 0.0125 = 0.0175 USD = 17.5e15 wei
        self.assertEqual(cost, 17_500_000_000_000_000)

    def test_claude_opus_4_6_cost(self):
        cost = self._calc("claude-opus-4-6", 1000, 500)
        self.assertEqual(cost, self._calc("claude-opus-4-5", 1000, 500))

    def test_claude_opus_4_7_cost(self):
        cost = self._calc("claude-opus-4-7", 1000, 500)
        expected = _expected_cost_opg("claude-opus-4-7", 1000, 500)
        self.assertEqual(cost, expected)
        # Same price tier as opus-4-5/4-6: 1000*0.000005 + 500*0.000025 = 0.0175 USD
        self.assertEqual(cost, 17_500_000_000_000_000)

    # ── Google Gemini ────────────────────────────────────────────────────────

    def test_gemini_2_5_flash_cost(self):
        cost = self._calc("gemini-2.5-flash", 1000, 500)
        expected = _expected_cost_opg("gemini-2.5-flash", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.0000003 + 500*0.0000025 = 0.0003 + 0.00125 = 0.00155 USD
        self.assertEqual(cost, 1_550_000_000_000_000)

    def test_gemini_2_5_flash_lite_cost(self):
        cost = self._calc("gemini-2.5-flash-lite", 1000, 500)
        expected = _expected_cost_opg("gemini-2.5-flash-lite", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.0000001 + 500*0.0000004 = 0.0001 + 0.0002 = 0.0003 USD
        self.assertEqual(cost, 300_000_000_000_000)

    def test_gemini_2_5_pro_cost(self):
        cost = self._calc("gemini-2.5-pro", 1000, 500)
        expected = _expected_cost_opg("gemini-2.5-pro", 1000, 500)
        self.assertEqual(cost, expected)

    def test_gemini_3_flash_preview_cost(self):
        cost = self._calc("gemini-3-flash-preview", 1000, 500)
        expected = _expected_cost_opg("gemini-3-flash-preview", 1000, 500)
        self.assertEqual(cost, expected)

    def test_gemini_3_1_pro_preview_cost(self):
        cost = self._calc("gemini-3.1-pro-preview", 1000, 500)
        expected = _expected_cost_opg("gemini-3.1-pro-preview", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.000002 + 500*0.000012 = 0.002 + 0.006 = 0.008 USD = 8e15 wei
        self.assertEqual(cost, 8_000_000_000_000_000)

    def test_gemini_3_1_flash_lite_preview_cost(self):
        cost = self._calc("gemini-3.1-flash-lite-preview", 1000, 500)
        expected = _expected_cost_opg("gemini-3.1-flash-lite-preview", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.00000025 + 500*0.0000015 = 0.00025 + 0.00075 = 0.001 USD = 1e15 wei
        self.assertEqual(cost, 1_000_000_000_000_000)

    # ── xAI Grok ────────────────────────────────────────────────────────────

    def test_grok_4_cost(self):
        cost = self._calc("grok-4", 1000, 500)
        expected = _expected_cost_opg("grok-4", 1000, 500)
        self.assertEqual(cost, expected)
        # Same pricing tier as claude-sonnet-4-5
        self.assertEqual(cost, 10_500_000_000_000_000)

    def test_grok_4_fast_cost(self):
        cost = self._calc("grok-4-fast", 1000, 500)
        expected = _expected_cost_opg("grok-4-fast", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.0000002 + 500*0.0000005 = 0.0002 + 0.00025 = 0.00045 USD
        self.assertEqual(cost, 450_000_000_000_000)

    def test_grok_4_1_fast_cost(self):
        cost = self._calc("grok-4-1-fast", 1000, 500)
        self.assertEqual(cost, self._calc("grok-4-fast", 1000, 500))

    def test_grok_4_20_reasoning_cost(self):
        cost = self._calc("grok-4.20-reasoning", 1000, 500)
        expected = _expected_cost_opg("grok-4.20-reasoning", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.000002 + 500*0.000006 = 0.002 + 0.003 = 0.005 USD = 5e15 wei
        self.assertEqual(cost, 5_000_000_000_000_000)

    def test_grok_4_20_non_reasoning_cost(self):
        cost = self._calc("grok-4.20-non-reasoning", 1000, 500)
        self.assertEqual(cost, self._calc("grok-4.20-reasoning", 1000, 500))

    def test_grok_code_fast_1_cost(self):
        cost = self._calc("grok-code-fast-1", 1000, 500)
        expected = _expected_cost_opg("grok-code-fast-1", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.0000002 + 500*0.0000015 = 0.0002 + 0.00075 = 0.00095 USD = 9.5e14 wei
        self.assertEqual(cost, 950_000_000_000_000)

    def test_grok_3_mini_cost(self):
        cost = self._calc("grok-3-mini", 1000, 500)
        expected = _expected_cost_opg("grok-3-mini", 1000, 500)
        self.assertEqual(cost, expected)

    def test_grok_3_cost(self):
        cost = self._calc("grok-3", 1000, 500)
        expected = _expected_cost_opg("grok-3", 1000, 500)
        self.assertEqual(cost, expected)

    # ── Haiku is cheaper than Sonnet ────────────────────────────────────────

    def test_haiku_cheaper_than_sonnet(self):
        haiku = self._calc("claude-haiku-4-5", 1000, 1000)
        sonnet = self._calc("claude-sonnet-4-5", 1000, 1000)
        self.assertLess(haiku, sonnet)

    def test_gemini_flash_lite_cheaper_than_flash(self):
        lite = self._calc("gemini-2.5-flash-lite", 1000, 1000)
        flash = self._calc("gemini-2.5-flash", 1000, 1000)
        self.assertLess(lite, flash)

    def test_grok_4_fast_cheaper_than_grok_4(self):
        fast = self._calc("grok-4-fast", 1000, 1000)
        full = self._calc("grok-4", 1000, 1000)
        self.assertLess(fast, full)


class TestCalculateSessionCostEdgeCases(unittest.TestCase):
    """Edge cases for calculate_session_cost."""

    def test_zero_tokens_returns_zero(self):
        cost = calculate_session_cost(_ctx("claude-sonnet-4-5", 0, 0), _get_price)
        self.assertEqual(cost, 0)

    def test_missing_usage_raises(self):
        ctx = {
            "request_json": {"model": "claude-sonnet-4-5"},
            "response_json": {"model": "claude-sonnet-4-5"},  # no usage
            "payment_requirements": _opg_requirements(),
        }
        with self.assertRaises(ValueError):
            calculate_session_cost(ctx, _get_price)

    def test_unknown_asset_raises(self):
        ctx = _ctx("claude-sonnet-4-5", 100, 100)
        ctx["payment_requirements"] = {"asset": "0xdeadbeef", "amount": "1000"}
        with self.assertRaises(ValueError):
            calculate_session_cost(ctx, _get_price)

    def test_missing_asset_raises(self):
        ctx = _ctx("claude-sonnet-4-5", 100, 100)
        ctx["payment_requirements"] = {"amount": "1000"}  # no asset
        with self.assertRaises(ValueError):
            calculate_session_cost(ctx, _get_price)

    def test_unknown_model_raises_value_error(self):
        ctx = _ctx("gpt-4o", 100, 100)
        with self.assertRaises(ValueError):
            calculate_session_cost(ctx, _get_price)

    def test_missing_request_json_raises_value_error(self):
        ctx = {
            "request_json": None,
            "response_json": {
                "model": "claude-sonnet-4-5",
                "usage": {"prompt_tokens": 100, "completion_tokens": 100},
            },
            "payment_requirements": _opg_requirements(),
        }
        with self.assertRaises(ValueError):
            calculate_session_cost(ctx, _get_price)

    def test_model_from_request_takes_priority(self):
        """request_json model name is used even if response_json has a different model."""
        ctx = {
            "request_json": {"model": "claude-haiku-4-5"},
            "response_json": {
                "model": "claude-sonnet-4-5",  # response says Sonnet
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            },
            "payment_requirements": _opg_requirements(),
        }
        cost = calculate_session_cost(ctx, _get_price)
        # Should be priced as Haiku (from request), not Sonnet
        haiku_cost = _expected_cost_opg("claude-haiku-4-5", 1000, 500)
        self.assertEqual(cost, haiku_cost)

    def test_rounding_ceiling(self):
        """Fractional token costs are always rounded UP."""
        # 1 output token of Haiku: 0.000005 USD = 5e12 wei — exact, no rounding needed
        cost = calculate_session_cost(_ctx("claude-haiku-4-5", 0, 1), _get_price)
        self.assertEqual(cost, 5_000_000_000_000)

        # 1 input token of Gemini Flash Lite: 0.0000001 USD = 1e11 wei — exact
        cost = calculate_session_cost(_ctx("gemini-2.5-flash-lite", 1, 0), _get_price)
        self.assertEqual(cost, 100_000_000_000)

    def test_model_name_case_insensitive(self):
        """Model names are normalized to lowercase before lookup."""
        cost_lower = calculate_session_cost(
            _ctx("claude-sonnet-4-5", 100, 100), _get_price
        )
        cost_upper = calculate_session_cost(
            _ctx("CLAUDE-SONNET-4-5", 100, 100), _get_price
        )
        self.assertEqual(cost_lower, cost_upper)



if __name__ == "__main__":
    unittest.main()
