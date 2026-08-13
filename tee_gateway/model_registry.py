"""
Single source of truth for all supported models.

Every model the gateway can route MUST be registered here with pricing.
Unknown models are rejected — there is no fallback.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, unique
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ModelConfig:
    # "openai" | "anthropic" | "google" | "x-ai" | "bytedance" | "nous" | "zai"
    provider: str
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
    # ── /images/generations request shaping (image_generation models only) ──
    # The ``response_format`` to request. ``"b64_json"`` returns inline bytes;
    # ``"url"`` returns a hosted link the gateway fetches and inlines (so the
    # client always receives bytes). ``None`` omits the field for endpoints that
    # don't document it (Z.ai GLM-Image).
    image_response_format: Optional[str] = "b64_json"
    # Whether to send the OpenAI-style ``n`` count. Some endpoints (Z.ai
    # GLM-Image, ByteDance Seedance) don't document it and reject/ignore it.
    image_send_n: bool = True
    # Whether the endpoint accepts reference images for image-to-image editing.
    # Text-to-image-only endpoints reject it. HOW the references are delivered
    # depends on ``image_edit_endpoint`` below.
    image_supports_reference: bool = False
    # Where/how reference images are sent (image_supports_reference models only):
    #   * ``None`` — inline them in the ``image`` field of the JSON
    #     ``/images/generations`` request (a string, or an array for multi-ref
    #     edits). Used by ByteDance Seedream/Seedance.
    #   * a path (e.g. ``"/images/edits"``) — POST multipart/form-data to that
    #     endpoint with each reference uploaded as a repeated ``image[]`` file.
    #     Used by OpenAI gpt-image, whose editing/compositing lives on a separate
    #     endpoint from text-to-image generation.
    image_edit_endpoint: Optional[str] = None
    # Static extra params merged verbatim into the request payload (e.g. size,
    # watermark). Keyed by field name; values must be JSON-serializable.
    image_extra_params: Optional[Mapping[str, Any]] = None
    # USD per image-modality output token, for ``image_output`` models (Gemini
    # "nano banana"). These providers bill image output at a higher rate than
    # text/thinking output: image tokens at this rate, text + thinking tokens at
    # ``output_price_usd``. ``None`` => single-rate billing (all output at
    # ``output_price_usd``). langchain folds image+text+thinking into one
    # ``output_tokens`` count and only breaks out thinking (``reasoning``), so the
    # billing splits reasoning at ``output_price_usd`` and the remainder here.
    image_output_price_usd: Optional[Decimal] = None
    # OpenAI's newest reasoning models (the gpt-5.6 family) apply a default
    # ``reasoning_effort`` that the Chat Completions endpoint rejects when
    # function tools are also present ("Function tools with reasoning_effort are
    # not supported ... use /v1/responses or set reasoning_effort to 'none'").
    # When True and function tools are bound, the gateway routes the request
    # through OpenAI's Responses API instead, which supports reasoning and
    # function tools together (preserving reasoning rather than disabling it).
    # OpenAI provider only; ignored elsewhere.
    responses_api_for_tools: bool = False


# Flat USD price per call to the /v1/web_search endpoint.
#
# The gateway runs searches against Exa (see web_search.py), so there is one
# cost to pass through, independent of any model. At our request shape — one
# Exa search plus page text for up to `MAX_NUM_RESULTS` results — Exa charges
# $7/1k requests and $1/1k pages per content type, i.e. ~$0.013 for a 6-result
# search. This rate covers that with a small margin, and is below what three of
# the four native provider searches used to cost (xAI $0.025/unit, Google
# $0.035/request).
#
# The billable unit is "one search that reached Exa": a request that fails
# validation or errors out at Exa returns without a cost block and is not
# settled (see WebSearchOutcome.billable).
WEB_SEARCH_PRICE_USD: Decimal = Decimal("0.015")

# ByteDance ModelArk image *deployment* endpoints (api_name "ep-…", e.g. Seedance
# 4.5, Seedream 5.0 Lite) return the URL response format and require these extra
# params. The gateway fetches the returned URL and inlines the bytes, so the
# client still receives inline bytes. Shared so the two ep- models stay in sync.
_BYTEDANCE_EP_IMAGE_PARAMS: dict[str, Any] = {
    "sequential_image_generation": "disabled",
    "watermark": False,
    "size": "2K",
    "stream": False,
}

# Seedance 5.0's deployment endpoint rejects ``sequential_image_generation``
# ("not supported by the current model" -> HTTP 400), so it takes the shared
# params minus that field.
_SEEDANCE_5_IMAGE_PARAMS: dict[str, Any] = {
    "watermark": False,
    "size": "2K",
    "stream": False,
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
    GPT_5_6_SOL = ModelConfig(
        provider="openai",
        api_name="gpt-5.6-sol",
        input_price_usd=Decimal("0.000005"),
        output_price_usd=Decimal("0.00003"),
        responses_api_for_tools=True,
    )
    GPT_5_6_TERRA = ModelConfig(
        provider="openai",
        api_name="gpt-5.6-terra",
        input_price_usd=Decimal("0.0000025"),
        output_price_usd=Decimal("0.000015"),
        responses_api_for_tools=True,
    )
    GPT_5_6_LUNA = ModelConfig(
        provider="openai",
        api_name="gpt-5.6-luna",
        input_price_usd=Decimal("0.000001"),
        output_price_usd=Decimal("0.000006"),
        responses_api_for_tools=True,
    )
    # Image generation via OpenAI's /images/generations endpoint (gpt-image).
    # Unlike DALL·E, gpt-image models always return base64 (``b64_json``) and
    # reject the ``response_format`` field, so it's omitted. Image-to-image
    # editing and multi-image compositing (e.g. "add this logo to this photo")
    # live on OpenAI's separate ``/images/edits`` endpoint, which takes the
    # reference images as multipart file uploads rather than a JSON ``image``
    # field — so reference turns are routed there via ``image_edit_endpoint``
    # (up to 10 references per request). Size/quality are pinned so the flat
    # per-image price stays predictable. Billed at a flat $0.05 per generated
    # image; token prices unused.
    GPT_IMAGE_2 = ModelConfig(
        provider="openai",
        api_name="gpt-image-2",
        input_price_usd=Decimal("0"),
        output_price_usd=Decimal("0"),
        image_generation=True,
        per_image_price_usd=Decimal("0.05"),
        image_response_format=None,
        image_supports_reference=True,
        image_edit_endpoint="/images/edits",
        image_extra_params={"size": "1024x1024", "quality": "medium"},
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
    # Claude Sonnet 5 — near-Opus quality on coding/agentic work at Sonnet cost.
    # Adaptive-thinking-only; like Opus 4.7+ it rejects the `temperature` field
    # (HTTP 400), so supports_temperature=False. Priced at the standard Sonnet
    # sticker ($3/$15 per MTok; an intro rate applied through 2026-08-31).
    CLAUDE_SONNET_5 = ModelConfig(
        provider="anthropic",
        api_name="claude-sonnet-5",
        input_price_usd=Decimal("0.000003"),
        output_price_usd=Decimal("0.000015"),
        supports_temperature=False,
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
    # Claude Opus 5 — the current Opus, a drop-in upgrade at Opus 4.8's pricing
    # ($5/$25 per MTok). Adaptive-thinking-only; like Opus 4.7+ it rejects the
    # `temperature` field (HTTP 400), so supports_temperature=False.
    CLAUDE_OPUS_5 = ModelConfig(
        provider="anthropic",
        api_name="claude-opus-5",
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
    GEMINI_3_5_FLASH_LITE = ModelConfig(
        provider="google",
        api_name="gemini-3.5-flash-lite",
        input_price_usd=Decimal("0.0000003"),
        output_price_usd=Decimal("0.0000025"),
    )
    GEMINI_3_6_FLASH = ModelConfig(
        provider="google",
        api_name="gemini-3.6-flash",
        input_price_usd=Decimal("0.0000015"),
        output_price_usd=Decimal("0.0000075"),
    )
    # Promotional standard pricing through December 31, 2026.
    GEMINI_3_7_FLASH = ModelConfig(
        provider="google",
        api_name="gemini-3.7-flash",
        input_price_usd=Decimal("0.00000075"),
        output_price_usd=Decimal("0.00000375"),
    )

    # ── xAI Grok ────────────────────────────────────────────────────────
    GROK_4_6 = ModelConfig(
        provider="x-ai",
        api_name="grok-4.6",
        input_price_usd=Decimal("0.000002"),
        output_price_usd=Decimal("0.000006"),
    )
    GROK_4_5 = ModelConfig(
        provider="x-ai",
        api_name="grok-4.5",
        input_price_usd=Decimal("0.000002"),
        output_price_usd=Decimal("0.000006"),
    )
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
    # Image generation via xAI's OpenAI-compatible /images/generations endpoint.
    # grok-2-image-1212 was retired in February 2026; grok-imagine-image is its
    # current replacement and returns hosted URLs that the gateway fetches and
    # inlines. Billed at a flat $0.02 per generated image; token prices unused.
    GROK_2_IMAGE = ModelConfig(
        provider="x-ai",
        api_name="grok-imagine-image",
        input_price_usd=Decimal("0"),
        output_price_usd=Decimal("0"),
        image_generation=True,
        per_image_price_usd=Decimal("0.02"),
        image_response_format="url",
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
    # Dola Seed 2.0 Mini uncensored deployment. The 128K context endpoint uses
    # the lower billing tier: $0.0001/K input and $0.0004/K output.
    DOLA_SEED_2_0_MINI = ModelConfig(
        provider="bytedance",
        api_name="ep-20260624214211-j4vhk",
        input_price_usd=Decimal("0.0000001"),
        output_price_usd=Decimal("0.0000004"),
    )
    DEEPSEEK_V4_FLASH = ModelConfig(
        provider="bytedance",
        api_name="deepseek-v4-flash-260425",
        input_price_usd=Decimal("0.00000014"),
        output_price_usd=Decimal("0.00000028"),
    )
    DEEPSEEK_V4_PRO = ModelConfig(
        provider="bytedance",
        api_name="deepseek-v4-pro-260425",
        input_price_usd=Decimal("0.00000174"),
        output_price_usd=Decimal("0.00000348"),
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
        image_supports_reference=True,
    )
    # Seedream 5.0 Lite image generation via a ModelArk deployment endpoint.
    # Seedream 5.0 Lite image generation via a ModelArk deployment endpoint
    # (api_name "ep-…"). Like Seedance it returns hosted URLs (fetched and inlined
    # by the gateway) and takes the shared ep- deployment params. Billed per image.
    SEEDREAM_5_0_LITE = ModelConfig(
        provider="bytedance",
        api_name="ep-20260624213657-7zc5n",
        input_price_usd=Decimal("0"),
        output_price_usd=Decimal("0"),
        image_generation=True,
        per_image_price_usd=Decimal("0.035"),
        image_response_format="url",
        image_send_n=False,
        image_supports_reference=True,
        image_extra_params=_BYTEDANCE_EP_IMAGE_PARAMS,
    )
    # Seedance 4.5 image generation via a ModelArk deployment endpoint.
    # Returns hosted URLs (fetched and inlined by the gateway) and takes the
    # shared ep- deployment params. Billed per image.
    SEEDANCE_4_5 = ModelConfig(
        provider="bytedance",
        api_name="ep-20260624042612-7dxcv",
        input_price_usd=Decimal("0"),
        output_price_usd=Decimal("0"),
        image_generation=True,
        per_image_price_usd=Decimal("0.05"),
        image_response_format="url",
        image_send_n=False,
        image_supports_reference=True,
        image_extra_params=_BYTEDANCE_EP_IMAGE_PARAMS,
    )
    # Seedance 5.0 image generation via a ModelArk deployment endpoint.
    # Returns hosted URLs (fetched and inlined by the gateway). Unlike the
    # other ep- models it rejects sequential_image_generation, so it takes its
    # own param set. BytePlus bills it tiered by output size ($0.045/image at
    # <=2.61MP, $0.09 above); the gateway pins size "2K" (~4.2MP), which
    # always lands in the upper tier, so bill flat $0.09.
    SEEDANCE_5_0 = ModelConfig(
        provider="bytedance",
        api_name="ep-20260803211347-hq9k8",
        input_price_usd=Decimal("0"),
        output_price_usd=Decimal("0"),
        image_generation=True,
        per_image_price_usd=Decimal("0.09"),
        image_response_format="url",
        image_send_n=False,
        image_supports_reference=True,
        image_extra_params=_SEEDANCE_5_IMAGE_PARAMS,
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

    # ── Z.ai (Model API, OpenAI-compatible) ─────────────────────────────
    # GLM-5.2 is served via a BytePlus ModelArk deployment endpoint (api_name
    # "ep-…") rather than Z.ai's own API — same model, same per-1M-token
    # pricing ($1.40 input, $4.40 output), routed through the bytedance client.
    GLM_5_2 = ModelConfig(
        provider="bytedance",
        api_name="ep-20260803211658-fwpzs",
        input_price_usd=Decimal("0.0000014"),
        output_price_usd=Decimal("0.0000044"),
    )
    # GLM-Image uses Z.ai's image endpoint and is billed per generated image.
    # Z.ai returns hosted URLs only (fetched and inlined by the gateway) and
    # documents neither ``n`` nor ``response_format``, so both are omitted.
    GLM_IMAGE = ModelConfig(
        provider="zai",
        api_name="glm-image",
        input_price_usd=Decimal("0"),
        output_price_usd=Decimal("0"),
        image_generation=True,
        per_image_price_usd=Decimal("0.015"),
        image_response_format=None,
        image_send_n=False,
        image_extra_params={"size": "1280x1280"},
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
    "gpt-5.6": SupportedModel.GPT_5_6_SOL,
    "gpt-5.6-sol": SupportedModel.GPT_5_6_SOL,
    "gpt-5.6-terra": SupportedModel.GPT_5_6_TERRA,
    "gpt-5.6-luna": SupportedModel.GPT_5_6_LUNA,
    "gpt-image-2": SupportedModel.GPT_IMAGE_2,
    # Anthropic
    "claude-sonnet-4-5": SupportedModel.CLAUDE_SONNET_4_5,
    "claude-sonnet-4-6": SupportedModel.CLAUDE_SONNET_4_6,
    "claude-sonnet-5": SupportedModel.CLAUDE_SONNET_5,
    "claude-haiku-4-5": SupportedModel.CLAUDE_HAIKU_4_5,
    "claude-opus-4-5": SupportedModel.CLAUDE_OPUS_4_5,
    "claude-opus-4-6": SupportedModel.CLAUDE_OPUS_4_6,
    "claude-opus-4-7": SupportedModel.CLAUDE_OPUS_4_7,
    "claude-opus-4-8": SupportedModel.CLAUDE_OPUS_4_8,
    "claude-opus-5": SupportedModel.CLAUDE_OPUS_5,
    "claude-fable-5": SupportedModel.CLAUDE_FABLE_5,
    # Google
    "gemini-2.5-flash": SupportedModel.GEMINI_2_5_FLASH,
    "gemini-2.5-pro": SupportedModel.GEMINI_2_5_PRO,
    "gemini-2.5-flash-lite": SupportedModel.GEMINI_2_5_FLASH_LITE,
    "gemini-3-flash-preview": SupportedModel.GEMINI_3_FLASH_PREVIEW,
    "gemini-3.1-pro-preview": SupportedModel.GEMINI_3_1_PRO_PREVIEW,
    "gemini-2.5-flash-image": SupportedModel.GEMINI_2_5_FLASH_IMAGE,
    "gemini-3.1-flash-image": SupportedModel.GEMINI_3_1_FLASH_IMAGE,
    "gemini-3.5-flash": SupportedModel.GEMINI_3_5_FLASH,
    "gemini-3.5-flash-lite": SupportedModel.GEMINI_3_5_FLASH_LITE,
    "gemini-3.6-flash": SupportedModel.GEMINI_3_6_FLASH,
    "gemini-3.7-flash": SupportedModel.GEMINI_3_7_FLASH,
    # xAI
    "grok-4.6": SupportedModel.GROK_4_6,
    "grok-4.5": SupportedModel.GROK_4_5,
    "grok-4.5-latest": SupportedModel.GROK_4_5,
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
    "grok-imagine-image": SupportedModel.GROK_2_IMAGE,
    "grok-imagine-image-2026-03-02": SupportedModel.GROK_2_IMAGE,
    # ByteDance
    "seed-1-6-250615": SupportedModel.SEED_1_6,
    "seed-1.6": SupportedModel.SEED_1_6,
    "seed-1-8-251228": SupportedModel.SEED_1_8,
    "seed-1.8": SupportedModel.SEED_1_8,
    "seed-2-0-lite-260228": SupportedModel.SEED_2_0_LITE,
    "seed-2.0-lite": SupportedModel.SEED_2_0_LITE,
    "dola-seed-2.0-mini": SupportedModel.DOLA_SEED_2_0_MINI,
    "dola-seed-2-0-mini": SupportedModel.DOLA_SEED_2_0_MINI,
    "deepseek-v4-flash-260425": SupportedModel.DEEPSEEK_V4_FLASH,
    "deepseek-v4-flash": SupportedModel.DEEPSEEK_V4_FLASH,
    "deepseek-v4-pro-260425": SupportedModel.DEEPSEEK_V4_PRO,
    "deepseek-v4-pro": SupportedModel.DEEPSEEK_V4_PRO,
    "seedream-4-0-250828": SupportedModel.SEEDREAM_4_0,
    "seedream-4.0": SupportedModel.SEEDREAM_4_0,
    "seedream-4-0": SupportedModel.SEEDREAM_4_0,
    "seedream-5.0-lite": SupportedModel.SEEDREAM_5_0_LITE,
    "seedream-5-0-lite": SupportedModel.SEEDREAM_5_0_LITE,
    "ep-20260624042612-7dxcv": SupportedModel.SEEDANCE_4_5,
    "seedance-4.5": SupportedModel.SEEDANCE_4_5,
    "seedance-4-5": SupportedModel.SEEDANCE_4_5,
    "ep-20260803211347-hq9k8": SupportedModel.SEEDANCE_5_0,
    "seedance-5.0": SupportedModel.SEEDANCE_5_0,
    "seedance-5-0": SupportedModel.SEEDANCE_5_0,
    # Nous Research
    "hermes-4-405b": SupportedModel.HERMES_4_405B,
    "hermes-4-70b": SupportedModel.HERMES_4_70B,
    # Z.ai
    "glm-5.2": SupportedModel.GLM_5_2,
    "ep-20260803211658-fwpzs": SupportedModel.GLM_5_2,
    "glm-image": SupportedModel.GLM_IMAGE,
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
