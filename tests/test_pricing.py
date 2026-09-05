"""
Unit tests for dynamic pricing / cost calculation across all supported models.

Tests verify that:
  - Every user-facing model name resolves to the correct ModelConfig
  - compute_session_cost produces the right amount in OPG token
    smallest-units for supported models
  - Edge cases (unknown model) are handled correctly
"""

import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from tee_gateway.model_registry import (
    _MODEL_LOOKUP,
    get_model_config,
)
from tee_gateway.pricing import compute_session_cost


# All pricing tests assume OPG = $1.00 so USD cost == OPG token amount.
_OPG_PRICE_USD = Decimal("1")


def _calc_opg(model: str, input_tokens: int, output_tokens: int) -> int:
    """Call compute_session_cost with the test price feed and return the OPG
    integer.  Returns -1 when the function returns None so tests can assert on
    failure paths without raising."""
    usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    fake_feed = SimpleNamespace(get_price=lambda: _OPG_PRICE_USD)
    with patch("tee_gateway.price_feed.get_price_feed", return_value=fake_feed):
        result = compute_session_cost(model, usage)
    return -1 if result is None else result.cost_opg


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
                if cfg.image_generation:
                    # Endpoint-based image models (Grok, Seedream) bill a flat
                    # per-image price; their token prices are intentionally 0.
                    self.assertIsNotNone(cfg.per_image_price_usd)
                    self.assertGreater(cfg.per_image_price_usd, 0)
                else:
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

    def test_claude_sonnet_5_resolves(self):
        cfg = get_model_config("claude-sonnet-5")
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.api_name, "claude-sonnet-5")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000003"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000015"))
        # Adaptive-thinking-only; rejects the `temperature` field (HTTP 400)
        self.assertFalse(cfg.supports_temperature)

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

    def test_claude_opus_4_8_resolves(self):
        cfg = get_model_config("claude-opus-4-8")
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000005"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000025"))
        # Opus 4.7+ rejects the `temperature` field (HTTP 400)
        self.assertFalse(cfg.supports_temperature)

    def test_claude_opus_5_resolves(self):
        cfg = get_model_config("claude-opus-5")
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.api_name, "claude-opus-5")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000005"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000025"))
        # Adaptive-thinking-only; rejects the `temperature` field (HTTP 400)
        self.assertFalse(cfg.supports_temperature)

    def test_claude_fable_5_resolves(self):
        cfg = get_model_config("claude-fable-5")
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.api_name, "claude-fable-5")
        self.assertEqual(cfg.input_price_usd, Decimal("0.00001"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.00005"))
        # Adaptive-thinking-only; rejects the `temperature` field (HTTP 400)
        self.assertFalse(cfg.supports_temperature)

    def test_claude_fable_5_1_resolves(self):
        cfg = get_model_config("claude-fable-5-1")
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.api_name, "claude-fable-5-1")
        self.assertEqual(cfg.input_price_usd, Decimal("0.00001"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.00005"))
        # Adaptive-thinking-only; rejects the `temperature` field (HTTP 400)
        self.assertFalse(cfg.supports_temperature)

    def test_claude_fable_5_1_dotted_alias_resolves(self):
        cfg = get_model_config("claude-fable-5.1")
        self.assertEqual(cfg, get_model_config("claude-fable-5-1"))

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

    def test_gpt_5_6_sol_resolves(self):
        cfg = get_model_config("gpt-5.6-sol")
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.api_name, "gpt-5.6-sol")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000005"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.00003"))

    def test_gpt_5_6_alias_resolves_to_sol(self):
        cfg = get_model_config("gpt-5.6")
        self.assertEqual(cfg, get_model_config("gpt-5.6-sol"))

    def test_gpt_5_6_terra_resolves(self):
        cfg = get_model_config("gpt-5.6-terra")
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000025"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000015"))

    def test_gpt_5_6_luna_resolves(self):
        cfg = get_model_config("gpt-5.6-luna")
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000001"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000006"))

    def test_gpt_6_astra_resolves(self):
        cfg = get_model_config("gpt-6-astra")
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.api_name, "gpt-6-astra")
        self.assertEqual(cfg.input_price_usd, Decimal("0.00001"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.00005"))
        self.assertTrue(cfg.responses_api_for_tools)

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

    def test_gemini_3_5_flash_resolves(self):
        cfg = get_model_config("gemini-3.5-flash")
        self.assertEqual(cfg.provider, "google")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000015"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000009"))

    def test_gemini_3_5_flash_lite_resolves(self):
        cfg = get_model_config("gemini-3.5-flash-lite")
        self.assertEqual(cfg.provider, "google")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000003"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.0000025"))

    def test_gemini_3_6_flash_resolves(self):
        cfg = get_model_config("gemini-3.6-flash")
        self.assertEqual(cfg.provider, "google")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000015"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.0000075"))

    def test_gemini_3_7_flash_resolves(self):
        cfg = get_model_config("gemini-3.7-flash")
        self.assertEqual(cfg.provider, "google")
        self.assertEqual(cfg.api_name, "gemini-3.7-flash")
        self.assertEqual(cfg.input_price_usd, Decimal("0.00000075"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.00000375"))

    def test_gemini_3_8_flash_resolves(self):
        cfg = get_model_config("gemini-3.8-flash")
        self.assertEqual(cfg.provider, "google")
        self.assertEqual(cfg.api_name, "gemini-3.8-flash")
        self.assertEqual(cfg.input_price_usd, Decimal("0.00000075"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.00000375"))

    def test_gemini_3_1_flash_image_resolves(self):
        cfg = get_model_config("gemini-3.1-flash-image")
        self.assertEqual(cfg.provider, "google")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000005"))
        # Output is dual-rate: text/thinking at output_price_usd, images at
        # image_output_price_usd ($3 vs $60 per MTok).
        self.assertEqual(cfg.output_price_usd, Decimal("0.000003"))
        self.assertEqual(cfg.image_output_price_usd, Decimal("0.00006"))
        self.assertTrue(cfg.image_output)

    # ── xAI Grok ────────────────────────────────────────────────────────────

    def test_grok_4_6_resolves(self):
        cfg = get_model_config("grok-4.6")
        self.assertEqual(cfg.provider, "x-ai")
        self.assertEqual(cfg.api_name, "grok-4.6")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000002"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000006"))

    def test_grok_4_5_resolves(self):
        cfg = get_model_config("grok-4.5")
        self.assertEqual(cfg.provider, "x-ai")
        self.assertEqual(cfg.api_name, "grok-4.5")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000002"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000006"))

    def test_grok_4_5_latest_resolves(self):
        cfg = get_model_config("grok-4.5-latest")
        self.assertEqual(cfg, get_model_config("grok-4.5"))

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

    # ── ByteDance (BytePlus ModelArk) ───────────────────────────────────────

    def test_seed_1_6_resolves(self):
        cfg = get_model_config("seed-1.6")
        self.assertEqual(cfg.provider, "bytedance")
        self.assertEqual(cfg.api_name, "seed-1-6-250615")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000008"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000008"))

    def test_seed_1_6_dated_alias_resolves(self):
        cfg = get_model_config("seed-1-6-250615")
        self.assertEqual(cfg, get_model_config("seed-1.6"))

    def test_seed_1_8_resolves(self):
        cfg = get_model_config("seed-1.8")
        self.assertEqual(cfg.provider, "bytedance")
        self.assertEqual(cfg.api_name, "seed-1-8-251228")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000008"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000008"))

    def test_seed_1_8_dated_alias_resolves(self):
        cfg = get_model_config("seed-1-8-251228")
        self.assertEqual(cfg, get_model_config("seed-1.8"))

    def test_seed_2_0_lite_resolves(self):
        cfg = get_model_config("seed-2.0-lite")
        self.assertEqual(cfg.provider, "bytedance")
        self.assertEqual(cfg.api_name, "seed-2-0-lite-260228")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000004"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.0000016"))

    def test_seed_2_0_lite_dated_alias_resolves(self):
        cfg = get_model_config("seed-2-0-lite-260228")
        self.assertEqual(cfg, get_model_config("seed-2.0-lite"))

    def test_dola_seed_2_0_mini_resolves(self):
        cfg = get_model_config("dola-seed-2.0-mini")
        self.assertEqual(cfg.provider, "bytedance")
        self.assertEqual(cfg.api_name, "ep-20260624214211-j4vhk")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000001"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.0000004"))

    def test_dola_seed_2_0_mini_aliases_resolve(self):
        self.assertEqual(
            get_model_config("dola-seed-2-0-mini"),
            get_model_config("dola-seed-2.0-mini"),
        )

    def test_deepseek_v4_flash_resolves(self):
        cfg = get_model_config("deepseek-v4-flash")
        self.assertEqual(cfg.provider, "bytedance")
        self.assertEqual(cfg.api_name, "deepseek-v4-flash-260425")
        self.assertEqual(cfg.input_price_usd, Decimal("0.00000014"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.00000028"))

    def test_deepseek_v4_flash_dated_alias_resolves(self):
        cfg = get_model_config("deepseek-v4-flash-260425")
        self.assertEqual(cfg, get_model_config("deepseek-v4-flash"))

    def test_deepseek_v4_pro_resolves(self):
        cfg = get_model_config("deepseek-v4-pro")
        self.assertEqual(cfg.provider, "bytedance")
        self.assertEqual(cfg.api_name, "deepseek-v4-pro-260425")
        self.assertEqual(cfg.input_price_usd, Decimal("0.00000174"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.00000348"))

    def test_deepseek_v4_pro_aliases_resolve(self):
        self.assertEqual(
            get_model_config("deepseek-v4-pro-260425"),
            get_model_config("deepseek-v4-pro"),
        )

    def test_seedream_5_0_lite_resolves(self):
        cfg = get_model_config("seedream-5.0-lite")
        self.assertEqual(cfg.provider, "bytedance")
        self.assertEqual(cfg.api_name, "ep-20260624213657-7zc5n")
        self.assertTrue(cfg.image_generation)
        self.assertEqual(cfg.per_image_price_usd, Decimal("0.035"))

    def test_seedream_5_0_lite_aliases_resolve(self):
        self.assertEqual(
            get_model_config("seedream-5-0-lite"),
            get_model_config("seedream-5.0-lite"),
        )

    def test_seedance_5_0_resolves(self):
        cfg = get_model_config("seedance-5.0")
        self.assertEqual(cfg.provider, "bytedance")
        self.assertEqual(cfg.api_name, "ep-20260803211347-hq9k8")
        self.assertTrue(cfg.image_generation)
        self.assertEqual(cfg.per_image_price_usd, Decimal("0.09"))

    def test_seedance_5_0_aliases_resolve(self):
        self.assertEqual(
            get_model_config("seedance-5-0"),
            get_model_config("seedance-5.0"),
        )
        self.assertEqual(
            get_model_config("ep-20260803211347-hq9k8"),
            get_model_config("seedance-5.0"),
        )

    # ── Nous Research models (OpenRouter) ───────────────────────────────────

    def test_hermes_4_405b_resolves(self):
        cfg = get_model_config("hermes-4-405b")
        self.assertEqual(cfg.provider, "openrouter")
        self.assertEqual(cfg.api_name, "nousresearch/hermes-4-405b")
        self.assertEqual(cfg.input_price_usd, Decimal("0.000001"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.000003"))

    def test_hermes_4_70b_resolves(self):
        cfg = get_model_config("hermes-4-70b")
        self.assertEqual(cfg.provider, "openrouter")
        self.assertEqual(cfg.api_name, "nousresearch/hermes-4-70b")
        self.assertEqual(cfg.input_price_usd, Decimal("0.00000013"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.0000004"))

    def test_openrouter_canonical_hermes_aliases_resolve(self):
        self.assertEqual(
            get_model_config("nousresearch/hermes-4-405b"),
            get_model_config("hermes-4-405b"),
        )
        self.assertEqual(
            get_model_config("nousresearch/hermes-4-70b"),
            get_model_config("hermes-4-70b"),
        )

    def test_hy3_resolves(self):
        cfg = get_model_config("hy3")
        self.assertEqual(cfg.provider, "openrouter")
        self.assertEqual(cfg.api_name, "tencent/hy3")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000000825"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.00000033"))
        self.assertEqual(cfg, get_model_config("tencent/hy3"))
        self.assertEqual(cfg, get_model_config("tencent/hy3:floor"))

    # ── Z.ai (Model API) ───────────────────────────────────────────────────

    def test_glm_5_3_resolves(self):
        cfg = get_model_config("glm-5.3")
        self.assertEqual(cfg.provider, "zai")
        self.assertEqual(cfg.api_name, "glm-5.3")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000014"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.0000044"))

    def test_glm_5_3_flash_resolves(self):
        cfg = get_model_config("glm-5.3-flash")
        self.assertEqual(cfg.provider, "zai")
        self.assertEqual(cfg.api_name, "glm-5.3-flash")
        self.assertEqual(cfg.input_price_usd, Decimal("0.00000015"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.0000005"))

    def test_glm_5_2_resolves(self):
        # GLM-5.2 is served via a BytePlus ModelArk deployment endpoint, not
        # Z.ai's own API; pricing is unchanged.
        cfg = get_model_config("glm-5.2")
        self.assertEqual(cfg.provider, "bytedance")
        self.assertEqual(cfg.api_name, "ep-20260803211658-fwpzs")
        self.assertEqual(cfg.input_price_usd, Decimal("0.0000014"))
        self.assertEqual(cfg.output_price_usd, Decimal("0.0000044"))

    def test_glm_5_2_ep_alias_resolves(self):
        self.assertEqual(
            get_model_config("ep-20260803211658-fwpzs"),
            get_model_config("glm-5.2"),
        )

    def test_glm_image_resolves(self):
        cfg = get_model_config("glm-image")
        self.assertEqual(cfg.provider, "zai")
        self.assertEqual(cfg.api_name, "glm-image")
        self.assertTrue(cfg.image_generation)
        self.assertEqual(cfg.per_image_price_usd, Decimal("0.015"))

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
    """compute_session_cost with OPG (18 decimals)."""

    def _calc(self, model, input_tokens, output_tokens):
        return _calc_opg(model, input_tokens, output_tokens)

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

    def test_gpt_5_6_sol_cost(self):
        cost = self._calc("gpt-5.6-sol", 1000, 500)
        expected = _expected_cost_opg("gpt-5.6-sol", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.000005 + 500*0.00003 = 0.005 + 0.015 = 0.02 USD = 2e16 wei
        self.assertEqual(cost, 20_000_000_000_000_000)

    def test_gpt_5_6_alias_cost(self):
        self.assertEqual(
            self._calc("gpt-5.6", 1000, 500),
            self._calc("gpt-5.6-sol", 1000, 500),
        )

    def test_gpt_5_6_terra_cost(self):
        cost = self._calc("gpt-5.6-terra", 1000, 500)
        expected = _expected_cost_opg("gpt-5.6-terra", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.0000025 + 500*0.000015 = 0.0025 + 0.0075 = 0.01 USD = 1e16 wei
        self.assertEqual(cost, 10_000_000_000_000_000)

    def test_gpt_5_6_luna_cost(self):
        cost = self._calc("gpt-5.6-luna", 1000, 500)
        expected = _expected_cost_opg("gpt-5.6-luna", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.000001 + 500*0.000006 = 0.001 + 0.003 = 0.004 USD = 4e15 wei
        self.assertEqual(cost, 4_000_000_000_000_000)

    def test_gpt_6_astra_cost(self):
        cost = self._calc("gpt-6-astra", 1000, 500)
        expected = _expected_cost_opg("gpt-6-astra", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.00001 + 500*0.00005 = 0.01 + 0.025 = 0.035 USD = 3.5e16 wei
        self.assertEqual(cost, 35_000_000_000_000_000)

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

    def test_claude_opus_5_cost(self):
        cost = self._calc("claude-opus-5", 1000, 500)
        expected = _expected_cost_opg("claude-opus-5", 1000, 500)
        self.assertEqual(cost, expected)
        # Same price tier as opus-4-5/4-6/4-7/4-8: 1000*0.000005 + 500*0.000025 = 0.0175 USD
        self.assertEqual(cost, 17_500_000_000_000_000)

    # ── Anthropic Fable ─────────────────────────────────────────────────────

    def test_claude_fable_5_1_cost(self):
        cost = self._calc("claude-fable-5-1", 1000, 500)
        expected = _expected_cost_opg("claude-fable-5-1", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.00001 + 500*0.00005 = 0.01 + 0.025 = 0.035 USD = 3.5e16 wei
        self.assertEqual(cost, 35_000_000_000_000_000)

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

    def test_gemini_3_5_flash_lite_cost(self):
        cost = self._calc("gemini-3.5-flash-lite", 1000, 500)
        self.assertEqual(cost, _expected_cost_opg("gemini-3.5-flash-lite", 1000, 500))
        self.assertEqual(cost, 1_550_000_000_000_000)

    def test_gemini_3_6_flash_cost(self):
        cost = self._calc("gemini-3.6-flash", 1000, 500)
        self.assertEqual(cost, _expected_cost_opg("gemini-3.6-flash", 1000, 500))
        self.assertEqual(cost, 5_250_000_000_000_000)

    def test_gemini_3_7_flash_cost(self):
        cost = self._calc("gemini-3.7-flash", 1000, 500)
        self.assertEqual(cost, _expected_cost_opg("gemini-3.7-flash", 1000, 500))
        self.assertEqual(cost, 2_625_000_000_000_000)

    def test_gemini_3_8_flash_cost(self):
        cost = self._calc("gemini-3.8-flash", 1000, 500)
        self.assertEqual(cost, _expected_cost_opg("gemini-3.8-flash", 1000, 500))
        self.assertEqual(cost, 2_625_000_000_000_000)

    # ── xAI Grok ────────────────────────────────────────────────────────────

    def test_grok_4_6_cost(self):
        cost = self._calc("grok-4.6", 1000, 500)
        self.assertEqual(cost, _expected_cost_opg("grok-4.6", 1000, 500))
        self.assertEqual(cost, 5_000_000_000_000_000)

    def test_grok_4_5_cost(self):
        cost = self._calc("grok-4.5", 1000, 500)
        expected = _expected_cost_opg("grok-4.5", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.000002 + 500*0.000006 = 0.002 + 0.003 = 0.005 USD = 5e15 wei
        self.assertEqual(cost, 5_000_000_000_000_000)

    def test_grok_4_5_latest_cost(self):
        self.assertEqual(
            self._calc("grok-4.5-latest", 1000, 500),
            self._calc("grok-4.5", 1000, 500),
        )

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

    # ── ByteDance (BytePlus ModelArk) ───────────────────────────────────────

    def test_seed_1_6_cost(self):
        cost = self._calc("seed-1.6", 1000, 500)
        expected = _expected_cost_opg("seed-1.6", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.0000008 + 500*0.000008 = 0.0008 + 0.004 = 0.0048 USD = 4.8e15 wei
        self.assertEqual(cost, 4_800_000_000_000_000)

    def test_seed_1_8_cost(self):
        cost = self._calc("seed-1.8", 1000, 500)
        # Same pricing tier as seed-1.6
        self.assertEqual(cost, self._calc("seed-1.6", 1000, 500))

    def test_seed_2_0_lite_cost(self):
        cost = self._calc("seed-2.0-lite", 1000, 500)
        expected = _expected_cost_opg("seed-2.0-lite", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.0000004 + 500*0.0000016 = 0.0004 + 0.0008 = 0.0012 USD = 1.2e15 wei
        self.assertEqual(cost, 1_200_000_000_000_000)

    def test_seed_2_0_lite_cheaper_than_seed_1_6(self):
        lite = self._calc("seed-2.0-lite", 1000, 1000)
        full = self._calc("seed-1.6", 1000, 1000)
        self.assertLess(lite, full)

    def test_deepseek_v4_flash_cost(self):
        cost = self._calc("deepseek-v4-flash", 1000, 500)
        expected = _expected_cost_opg("deepseek-v4-flash", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.00000014 + 500*0.00000028 = 0.00014 + 0.00014 = 0.00028 USD
        self.assertEqual(cost, 280_000_000_000_000)

    def test_deepseek_v4_pro_cost(self):
        cost = self._calc("deepseek-v4-pro", 1000, 500)
        expected = _expected_cost_opg("deepseek-v4-pro", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.00000174 + 500*0.00000348 = 0.00174 + 0.00174 = 0.00348 USD
        self.assertEqual(cost, 3_480_000_000_000_000)

    # ── OpenRouter ─────────────────────────────────────────────────────────

    def test_hy3_cost(self):
        self.assertEqual(
            self._calc("hy3", 1000, 500),
            247_500_000_000_000,
        )

    # ── Z.ai ─────────────────────────────────────────────────────────────

    def test_glm_5_3_cost(self):
        cost = self._calc("glm-5.3", 1000, 500)
        expected = _expected_cost_opg("glm-5.3", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.0000014 + 500*0.0000044 = 0.0014 + 0.0022 = 0.0036 USD
        self.assertEqual(cost, 3_600_000_000_000_000)

    def test_glm_5_3_flash_cost(self):
        cost = self._calc("glm-5.3-flash", 1000, 500)
        expected = _expected_cost_opg("glm-5.3-flash", 1000, 500)
        self.assertEqual(cost, expected)
        # 1000*0.00000015 + 500*0.0000005 = 0.00015 + 0.00025 = 0.0004 USD
        self.assertEqual(cost, 400_000_000_000_000)

    def test_glm_5_3_flash_cheaper_than_glm_5_3(self):
        flash = self._calc("glm-5.3-flash", 1000, 1000)
        full = self._calc("glm-5.3", 1000, 1000)
        self.assertLess(flash, full)

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
    """Edge cases for compute_session_cost."""

    def test_zero_tokens_returns_zero(self):
        self.assertEqual(_calc_opg("claude-sonnet-4-5", 0, 0), 0)

    def test_unknown_model_returns_none(self):
        # gpt-4o is not in the registry — get_model_config raises, caught and
        # returned as None.
        self.assertEqual(_calc_opg("gpt-4o", 100, 100), -1)

    def test_rounding_ceiling(self):
        """Fractional token costs are always rounded UP."""
        # 1 output token of Haiku: 0.000005 USD = 5e12 wei — exact
        self.assertEqual(_calc_opg("claude-haiku-4-5", 0, 1), 5_000_000_000_000)
        # 1 input token of Gemini Flash Lite: 0.0000001 USD = 1e11 wei — exact
        self.assertEqual(_calc_opg("gemini-2.5-flash-lite", 1, 0), 100_000_000_000)

    def test_model_name_case_insensitive(self):
        self.assertEqual(
            _calc_opg("claude-sonnet-4-5", 100, 100),
            _calc_opg("CLAUDE-SONNET-4-5", 100, 100),
        )


if __name__ == "__main__":
    unittest.main()
