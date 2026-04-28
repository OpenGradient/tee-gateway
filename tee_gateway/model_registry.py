"""
Single source of truth for all supported models.

Every model the gateway can route MUST be registered here with pricing.
Unknown models are rejected — there is no fallback.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, unique
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    provider: str  # "openai" | "anthropic" | "google" | "x-ai"
    api_name: str  # model name sent to provider API
    input_price_usd: Decimal  # USD per token
    output_price_usd: Decimal  # USD per token
    force_temperature: Optional[float] = None
    thinking_budget: Optional[int] = None


@unique
class SupportedModel(Enum):
    # ── OpenAI ──────────────────────────────────────────────────────────
    GPT_4_1 = ModelConfig(
        provider="openai",
        api_name="gpt-4.1-2025-04-14",
        input_price_usd=Decimal("0.000002"),
        output_price_usd=Decimal("0.000008"),
    )
    GPT_4_1_MINI = ModelConfig(
        provider="openai",
        api_name="gpt-4.1-mini",
        input_price_usd=Decimal("0.0000004"),
        output_price_usd=Decimal("0.0000016"),
    )
    GPT_4_1_NANO = ModelConfig(
        provider="openai",
        api_name="gpt-4.1-nano",
        input_price_usd=Decimal("0.0000001"),
        output_price_usd=Decimal("0.0000004"),
    )
    O3 = ModelConfig(
        provider="openai",
        api_name="o3",
        input_price_usd=Decimal("0.00001"),
        output_price_usd=Decimal("0.00004"),
        force_temperature=1.0,
    )
    O4_MINI = ModelConfig(
        provider="openai",
        api_name="o4-mini",
        input_price_usd=Decimal("0.0000011"),
        output_price_usd=Decimal("0.0000044"),
        force_temperature=1.0,
    )
    GPT_5 = ModelConfig(
        provider="openai",
        api_name="gpt-5",
        input_price_usd=Decimal("0.00000125"),
        output_price_usd=Decimal("0.00001"),
    )
    GPT_5_MINI = ModelConfig(
        provider="openai",
        api_name="gpt-5-mini",
        input_price_usd=Decimal("0.00000025"),
        output_price_usd=Decimal("0.000002"),
    )
    GPT_5_2 = ModelConfig(
        provider="openai",
        api_name="gpt-5.2",
        input_price_usd=Decimal("0.00000175"),
        output_price_usd=Decimal("0.000014"),
    )
    GPT_5_4 = ModelConfig(
        provider="openai",
        api_name="gpt-5.4",
        input_price_usd=Decimal("0.0000025"),
        output_price_usd=Decimal("0.000015"),
    )
    GPT_5_4_MINI = ModelConfig(
        provider="openai",
        api_name="gpt-5.4-mini",
        input_price_usd=Decimal("0.00000075"),
        output_price_usd=Decimal("0.0000045"),
    )
    GPT_5_4_NANO = ModelConfig(
        provider="openai",
        api_name="gpt-5.4-nano",
        input_price_usd=Decimal("0.0000002"),
        output_price_usd=Decimal("0.00000125"),
    )
    GPT_5_5 = ModelConfig(
        provider="openai",
        api_name="gpt-5.5",
        input_price_usd=Decimal("0.000005"),
        output_price_usd=Decimal("0.00003"),
    )

    # ── Anthropic ───────────────────────────────────────────────────────
    CLAUDE_SONNET_4_5 = ModelConfig(
        provider="anthropic",
        api_name="claude-sonnet-4-5",
        input_price_usd=Decimal("0.000003"),
        output_price_usd=Decimal("0.000015"),
    )
    CLAUDE_SONNET_4_6 = ModelConfig(
        provider="anthropic",
        api_name="claude-sonnet-4-6",
        input_price_usd=Decimal("0.000003"),
        output_price_usd=Decimal("0.000015"),
    )
    CLAUDE_HAIKU_4_5 = ModelConfig(
        provider="anthropic",
        api_name="claude-haiku-4-5-20251001",
        input_price_usd=Decimal("0.000001"),
        output_price_usd=Decimal("0.000005"),
    )
    CLAUDE_OPUS_4_5 = ModelConfig(
        provider="anthropic",
        api_name="claude-opus-4-5-20251101",
        input_price_usd=Decimal("0.000005"),
        output_price_usd=Decimal("0.000025"),
    )
    CLAUDE_OPUS_4_6 = ModelConfig(
        provider="anthropic",
        api_name="claude-opus-4-6",
        input_price_usd=Decimal("0.000005"),
        output_price_usd=Decimal("0.000025"),
    )
    CLAUDE_OPUS_4_7 = ModelConfig(
        provider="anthropic",
        api_name="claude-opus-4-7",
        input_price_usd=Decimal("0.000005"),
        output_price_usd=Decimal("0.000025"),
    )

    # ── Google Gemini ───────────────────────────────────────────────────
    # Note: gemini-2.5-flash, gemini-2.5-pro, and gemini-2.5-flash-lite are scheduled
    # for deprecation on June 17, 2026 (flash-lite: July 22, 2026). Use the Gemini 3
    # replacements below for new integrations.
    GEMINI_2_5_FLASH = ModelConfig(
        provider="google",
        api_name="gemini-2.5-flash",
        input_price_usd=Decimal("0.0000003"),
        output_price_usd=Decimal("0.0000025"),
        thinking_budget=0,
    )
    GEMINI_2_5_PRO = ModelConfig(
        provider="google",
        api_name="gemini-2.5-pro",
        input_price_usd=Decimal("0.00000125"),
        output_price_usd=Decimal("0.00001"),
        thinking_budget=128,
    )
    GEMINI_2_5_FLASH_LITE = ModelConfig(
        provider="google",
        api_name="gemini-2.5-flash-lite",
        input_price_usd=Decimal("0.0000001"),
        output_price_usd=Decimal("0.0000004"),
        thinking_budget=0,
    )
    GEMINI_3_FLASH_PREVIEW = ModelConfig(
        provider="google",
        api_name="gemini-3-flash-preview",
        input_price_usd=Decimal("0.0000005"),
        output_price_usd=Decimal("0.000003"),
    )
    GEMINI_3_1_PRO_PREVIEW = ModelConfig(
        provider="google",
        api_name="gemini-3.1-pro-preview",
        input_price_usd=Decimal("0.000002"),
        output_price_usd=Decimal("0.000012"),
        thinking_budget=128,
    )
    GEMINI_3_1_FLASH_LITE_PREVIEW = ModelConfig(
        provider="google",
        api_name="gemini-3.1-flash-lite-preview",
        input_price_usd=Decimal("0.00000025"),
        output_price_usd=Decimal("0.0000015"),
        thinking_budget=0,
    )

    # ── xAI Grok ────────────────────────────────────────────────────────
    GROK_4 = ModelConfig(
        provider="x-ai",
        api_name="grok-4",
        input_price_usd=Decimal("0.000003"),
        output_price_usd=Decimal("0.000015"),
    )
    GROK_4_FAST = ModelConfig(
        provider="x-ai",
        api_name="grok-4-fast",
        input_price_usd=Decimal("0.0000002"),
        output_price_usd=Decimal("0.0000005"),
    )
    GROK_4_1_FAST = ModelConfig(
        provider="x-ai",
        api_name="grok-4-1-fast",
        input_price_usd=Decimal("0.0000002"),
        output_price_usd=Decimal("0.0000005"),
    )
    GROK_4_1_FAST_NON_REASONING = ModelConfig(
        provider="x-ai",
        api_name="grok-4-1-fast-non-reasoning",
        input_price_usd=Decimal("0.0000002"),
        output_price_usd=Decimal("0.0000005"),
    )
    GROK_4_20_REASONING = ModelConfig(
        provider="x-ai",
        api_name="grok-4.20-reasoning",
        input_price_usd=Decimal("0.000002"),
        output_price_usd=Decimal("0.000006"),
    )
    GROK_4_20_NON_REASONING = ModelConfig(
        provider="x-ai",
        api_name="grok-4.20-non-reasoning",
        input_price_usd=Decimal("0.000002"),
        output_price_usd=Decimal("0.000006"),
    )
    GROK_CODE_FAST_1 = ModelConfig(
        provider="x-ai",
        api_name="grok-code-fast-1",
        input_price_usd=Decimal("0.0000002"),
        output_price_usd=Decimal("0.0000015"),
    )

    # ── Legacy models (not in current SDK — retained for older SDK versions) ──
    GROK_3_MINI = ModelConfig(
        provider="x-ai",
        api_name="grok-3-mini",
        input_price_usd=Decimal("0.0000003"),
        output_price_usd=Decimal("0.0000005"),
    )
    GROK_3 = ModelConfig(
        provider="x-ai",
        api_name="grok-3-latest",
        input_price_usd=Decimal("0.000003"),
        output_price_usd=Decimal("0.000015"),
    )


# Canonical lookup: user-facing model name → SupportedModel
# The "user-facing name" is what callers pass in the `model` field of requests.
_MODEL_LOOKUP: dict[str, SupportedModel] = {
    # OpenAI
    "gpt-4.1-2025-04-14": SupportedModel.GPT_4_1,
    "gpt-4.1": SupportedModel.GPT_4_1,
    "gpt-4.1-mini": SupportedModel.GPT_4_1_MINI,
    "gpt-4.1-mini-2025-04-14": SupportedModel.GPT_4_1_MINI,
    "gpt-4.1-nano": SupportedModel.GPT_4_1_NANO,
    "gpt-4.1-nano-2025-04-14": SupportedModel.GPT_4_1_NANO,
    "o3": SupportedModel.O3,
    "o3-2025-04-16": SupportedModel.O3,
    "o4-mini": SupportedModel.O4_MINI,
    "gpt-5": SupportedModel.GPT_5,
    "gpt-5-mini": SupportedModel.GPT_5_MINI,
    "gpt-5.2": SupportedModel.GPT_5_2,
    "gpt-5.4": SupportedModel.GPT_5_4,
    "gpt-5.4-mini": SupportedModel.GPT_5_4_MINI,
    "gpt-5.4-nano": SupportedModel.GPT_5_4_NANO,
    "gpt-5.5": SupportedModel.GPT_5_5,
    # Anthropic
    "claude-sonnet-4-5": SupportedModel.CLAUDE_SONNET_4_5,
    "claude-sonnet-4-6": SupportedModel.CLAUDE_SONNET_4_6,
    "claude-haiku-4-5": SupportedModel.CLAUDE_HAIKU_4_5,
    "claude-opus-4-5": SupportedModel.CLAUDE_OPUS_4_5,
    "claude-opus-4-6": SupportedModel.CLAUDE_OPUS_4_6,
    "claude-opus-4-7": SupportedModel.CLAUDE_OPUS_4_7,
    # Google
    "gemini-2.5-flash": SupportedModel.GEMINI_2_5_FLASH,
    "gemini-2.5-pro": SupportedModel.GEMINI_2_5_PRO,
    "gemini-2.5-flash-lite": SupportedModel.GEMINI_2_5_FLASH_LITE,
    "gemini-3-flash-preview": SupportedModel.GEMINI_3_FLASH_PREVIEW,
    "gemini-3.1-pro-preview": SupportedModel.GEMINI_3_1_PRO_PREVIEW,
    "gemini-3.1-flash-lite-preview": SupportedModel.GEMINI_3_1_FLASH_LITE_PREVIEW,
    # xAI
    "grok-4": SupportedModel.GROK_4,
    "grok-4-fast": SupportedModel.GROK_4_FAST,
    "grok-4-1-fast": SupportedModel.GROK_4_1_FAST,
    "grok-4.1-fast": SupportedModel.GROK_4_1_FAST,
    "grok-4-1-fast-non-reasoning": SupportedModel.GROK_4_1_FAST_NON_REASONING,
    "grok-4.20-reasoning": SupportedModel.GROK_4_20_REASONING,
    "grok-4.20-non-reasoning": SupportedModel.GROK_4_20_NON_REASONING,
    "grok-code-fast-1": SupportedModel.GROK_CODE_FAST_1,
    # Legacy — not in current SDK, retained for older SDK versions
    "grok-3-mini-beta": SupportedModel.GROK_3_MINI,  # old beta alias
    "grok-3-mini": SupportedModel.GROK_3_MINI,
    "grok-3-beta": SupportedModel.GROK_3,  # old beta alias
    "grok-3": SupportedModel.GROK_3,
}

# Build the rate card automatically from the enum (for backward compat with util.py)
MODEL_RATE_CARD_USD: dict[str, dict[str, Decimal]] = {}
for _name, _model in _MODEL_LOOKUP.items():
    cfg = _model.value
    MODEL_RATE_CARD_USD[_name] = {
        "input": cfg.input_price_usd,
        "output": cfg.output_price_usd,
    }


def get_model_config(model: str) -> ModelConfig:
    """Look up model config by user-facing name. Raises ValueError if unknown."""
    normalized = model.strip().lower()
    entry = _MODEL_LOOKUP.get(normalized)
    if entry is None:
        supported = sorted(_MODEL_LOOKUP.keys())
        raise ValueError(
            f"Unsupported model: {model!r}. Supported models: {', '.join(supported)}"
        )
    return entry.value


def get_rate_card(model: str) -> dict[str, Decimal]:
    """Return {"input": ..., "output": ...} pricing for a model. Raises on unknown."""
    cfg = get_model_config(model)
    return {"input": cfg.input_price_usd, "output": cfg.output_price_usd}
