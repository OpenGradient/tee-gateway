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
    provider: str  # "openai" | "anthropic" | "google" | "x-ai" | "bytedance" | "nous"
    api_name: str  # model name sent to provider API
    input_price_usd: Decimal  # USD per token
    output_price_usd: Decimal  # USD per token
    force_temperature: Optional[float] = None
    thinking_budget: Optional[int] = None
    # Anthropic deprecated `temperature` for Opus 4.7 — the API returns 400 if
    # the field is present at all. Set False on models that reject it.
    supports_temperature: bool = True
    # Image-output models (e.g. Gemini "nano banana") return generated images as
    # inline content blocks of a chat response. The backend requests the IMAGE
    # modality and the controller surfaces the image data on the response message.
    image_output: bool = False
    # Image-generation models served via a dedicated OpenAI-compatible
    # ``POST /images/generations`` endpoint (e.g. xAI Grok, ByteDance Seedream)
    # rather than the chat path. These are billed per generated image
    # (``per_image_price_usd``) instead of per token.
    image_generation: bool = False
    # Flat USD price per generated image, for ``image_generation`` models. Token
    # prices are ignored for these models (set to 0 in the registry).
    per_image_price_usd: Optional[Decimal] = None
    # USD per image-modality output token, for ``image_output`` models (Gemini
    # "nano banana"). These providers bill image output at a higher rate than
    # text/thinking output: image tokens at this rate, text + thinking tokens at
    # ``output_price_usd``. ``None`` => single-rate billing (all output at
    # ``output_price_usd``). langchain folds image+text+thinking into one
    # ``output_tokens`` count and only breaks out thinking (``reasoning``), so the
    # billing splits reasoning at ``output_price_usd`` and the remainder here.
    image_output_price_usd: Optional[Decimal] = None
    # Per-search USD surcharge billed when native web search is used. ``None``
    # means "use the provider default" (see WEB_SEARCH_PRICE_USD_BY_PROVIDER);
    # set an explicit value here to override a single model's web-search price.
    web_search_price_usd: Optional[Decimal] = None


# Default per-search USD price charged when a model uses native web search.
# The billable "unit" differs per provider (see extract_web_search_count in
# llm_backend.py) and these mirror each provider's public list price:
#   - OpenAI:    per web_search tool call          (~$10 / 1k calls)
#   - Anthropic: per web_search request            ($10 / 1k searches)
#   - xAI:       per web_search tool call / source ($25 / 1k units)
#   - Google:    per grounded request               ($35 / 1k requests)
# Providers without native web search are omitted (charged nothing).
WEB_SEARCH_PRICE_USD_BY_PROVIDER: dict[str, Decimal] = {
    "openai": Decimal("0.01"),
    "anthropic": Decimal("0.01"),
    "x-ai": Decimal("0.025"),
    "google": Decimal("0.035"),
}


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
        supports_temperature=False,
    )
    CLAUDE_OPUS_4_8 = ModelConfig(
        provider="anthropic",
        api_name="claude-opus-4-8",
        input_price_usd=Decimal("0.000005"),
        output_price_usd=Decimal("0.000025"),
        supports_temperature=False,
    )
    # Claude Fable 5 — Anthropic's most capable widely released model (GA on the
    # first-party API from 2026-06-09). Adaptive-thinking-only; like Opus 4.7+ it
    # rejects `temperature` (HTTP 400), so supports_temperature=False.
    CLAUDE_FABLE_5 = ModelConfig(
        provider="anthropic",
        api_name="claude-fable-5",
        input_price_usd=Decimal("0.00001"),
        output_price_usd=Decimal("0.00005"),
        supports_temperature=False,
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
    # Native image generation ("nano banana"). Google bills output at two rates:
    # text/thinking at $1.50/MTok and images at $30/MTok (~1290 tokens per
    # 1024x1024 image ≈ $0.039/image); input (text/image) is $0.30/MTok.
    GEMINI_2_5_FLASH_IMAGE = ModelConfig(
        provider="google",
        api_name="gemini-2.5-flash-image",
        input_price_usd=Decimal("0.0000003"),
        output_price_usd=Decimal("0.0000015"),
        image_output=True,
        image_output_price_usd=Decimal("0.00003"),
    )
    # Native image generation ("nano banana 2"), the latest Gemini image model.
    # Google bills output at two rates: text/thinking at $3/MTok and images at
    # $60/MTok (~1120 tokens per 1K image ≈ $0.067/image; $0.045/0.101/0.151 at
    # 0.5K/2K/4K); input (text/image) is $0.50/MTok.
    GEMINI_3_1_FLASH_IMAGE = ModelConfig(
        provider="google",
        api_name="gemini-3.1-flash-image",
        input_price_usd=Decimal("0.0000005"),
        output_price_usd=Decimal("0.000003"),
        image_output=True,
        image_output_price_usd=Decimal("0.00006"),
    )
    GEMINI_3_5_FLASH = ModelConfig(
        provider="google",
        api_name="gemini-3.5-flash",
        input_price_usd=Decimal("0.0000015"),
        output_price_usd=Decimal("0.000009"),
    )

    # ── xAI Grok ────────────────────────────────────────────────────────
    GROK_4_3 = ModelConfig(
        provider="x-ai",
        api_name="grok-4.3",
        input_price_usd=Decimal("0.00000125"),
        output_price_usd=Decimal("0.0000025"),
    )
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
    # Image generation via xAI's OpenAI-compatible /images/generations endpoint
    # (Aurora). Billed at a flat $0.07 per generated image; token prices unused.
    GROK_2_IMAGE = ModelConfig(
        provider="x-ai",
        api_name="grok-2-image-1212",
        input_price_usd=Decimal("0"),
        output_price_usd=Decimal("0"),
        image_generation=True,
        per_image_price_usd=Decimal("0.07"),
    )

    # ── ByteDance (BytePlus ModelArk, OpenAI-compatible) ────────────────
    SEED_1_6 = ModelConfig(
        provider="bytedance",
        api_name="seed-1-6-250615",
        input_price_usd=Decimal("0.0000008"),
        output_price_usd=Decimal("0.000008"),
    )
    SEED_1_8 = ModelConfig(
        provider="bytedance",
        api_name="seed-1-8-251228",
        input_price_usd=Decimal("0.0000008"),
        output_price_usd=Decimal("0.000008"),
    )
    SEED_2_0_LITE = ModelConfig(
        provider="bytedance",
        api_name="seed-2-0-lite-260228",
        input_price_usd=Decimal("0.0000004"),
        output_price_usd=Decimal("0.0000016"),
    )
    # Seedream 4.0 image generation via ModelArk's OpenAI-compatible
    # /images/generations endpoint. Billed at a flat $0.03 per generated image;
    # token prices unused.
    SEEDREAM_4_0 = ModelConfig(
        provider="bytedance",
        api_name="seedream-4-0-250828",
        input_price_usd=Decimal("0"),
        output_price_usd=Decimal("0"),
        image_generation=True,
        per_image_price_usd=Decimal("0.03"),
    )

    # ── Nous Research (Nous Portal, OpenAI-compatible) ──────────────────
    # Hermes 4 family, served via Nous's OpenAI-compatible inference API.
    # Pricing mirrors Nous Portal's published per-token rates.
    #
    # The Nous inference API is an OpenRouter-style aggregator with case-sensitive
    # model-id matching. The bare lowercase "hermes-4-70b" is NOT an accepted
    # alias and is rejected with HTTP 400; the accepted form is the capitalized
    # "Hermes-4-70B" (the canonical id "nousresearch/hermes-4-70b" also works).
    HERMES_4_405B = ModelConfig(
        provider="nous",
        api_name="Hermes-4-405B",
        input_price_usd=Decimal("0.00000009"),
        output_price_usd=Decimal("0.00000037"),
    )
    HERMES_4_70B = ModelConfig(
        provider="nous",
        api_name="Hermes-4-70B",
        input_price_usd=Decimal("0.00000013"),
        output_price_usd=Decimal("0.0000004"),
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
    "claude-opus-4-8": SupportedModel.CLAUDE_OPUS_4_8,
    "claude-fable-5": SupportedModel.CLAUDE_FABLE_5,
    # Google
    "gemini-2.5-flash": SupportedModel.GEMINI_2_5_FLASH,
    "gemini-2.5-pro": SupportedModel.GEMINI_2_5_PRO,
    "gemini-2.5-flash-lite": SupportedModel.GEMINI_2_5_FLASH_LITE,
    "gemini-3-flash-preview": SupportedModel.GEMINI_3_FLASH_PREVIEW,
    "gemini-3.1-pro-preview": SupportedModel.GEMINI_3_1_PRO_PREVIEW,
    "gemini-3.1-flash-lite-preview": SupportedModel.GEMINI_3_1_FLASH_LITE_PREVIEW,
    "gemini-2.5-flash-image": SupportedModel.GEMINI_2_5_FLASH_IMAGE,
    "gemini-3.1-flash-image": SupportedModel.GEMINI_3_1_FLASH_IMAGE,
    "gemini-3.5-flash": SupportedModel.GEMINI_3_5_FLASH,
    # xAI
    "grok-4.3": SupportedModel.GROK_4_3,
    "grok-4": SupportedModel.GROK_4,
    "grok-4-fast": SupportedModel.GROK_4_FAST,
    "grok-4-1-fast": SupportedModel.GROK_4_1_FAST,
    "grok-4.1-fast": SupportedModel.GROK_4_1_FAST,
    "grok-4-1-fast-non-reasoning": SupportedModel.GROK_4_1_FAST_NON_REASONING,
    "grok-4.20-reasoning": SupportedModel.GROK_4_20_REASONING,
    "grok-4.20-non-reasoning": SupportedModel.GROK_4_20_NON_REASONING,
    "grok-code-fast-1": SupportedModel.GROK_CODE_FAST_1,
    "grok-2-image": SupportedModel.GROK_2_IMAGE,
    "grok-2-image-1212": SupportedModel.GROK_2_IMAGE,
    "grok-2-image-latest": SupportedModel.GROK_2_IMAGE,
    # ByteDance
    "seed-1-6-250615": SupportedModel.SEED_1_6,
    "seed-1.6": SupportedModel.SEED_1_6,
    "seed-1-8-251228": SupportedModel.SEED_1_8,
    "seed-1.8": SupportedModel.SEED_1_8,
    "seed-2-0-lite-260228": SupportedModel.SEED_2_0_LITE,
    "seed-2.0-lite": SupportedModel.SEED_2_0_LITE,
    "seedream-4-0-250828": SupportedModel.SEEDREAM_4_0,
    "seedream-4.0": SupportedModel.SEEDREAM_4_0,
    "seedream-4-0": SupportedModel.SEEDREAM_4_0,
    # Nous Research
    "hermes-4-405b": SupportedModel.HERMES_4_405B,
    "hermes-4-70b": SupportedModel.HERMES_4_70B,
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


def get_web_search_price_usd(model: str) -> Decimal:
    """Return the per-search USD surcharge for a model's native web search.

    Falls back to the provider default when the model does not override it, and
    to ``Decimal("0")`` for providers with no native web search support. Raises
    ValueError if the model is unknown.
    """
    cfg = get_model_config(model)
    if cfg.web_search_price_usd is not None:
        return cfg.web_search_price_usd
    return WEB_SEARCH_PRICE_USD_BY_PROVIDER.get(cfg.provider, Decimal("0"))


def provider_supports_web_search(provider: str) -> bool:
    """Whether the given provider has native web search the gateway can enable."""
    return provider in WEB_SEARCH_PRICE_USD_BY_PROVIDER
