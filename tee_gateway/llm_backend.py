"""
LLM routing backend for TEE-LLM-Router.

Provides LangChain model instantiation, provider routing, message conversion,
and shared HTTP clients. Replaces src/server.py as the direct LLM backend
called from the Flask/connexion controllers.
"""

import json
import logging
from typing import List, Dict, Optional, Any, Generator
from functools import lru_cache

import httpx
from pydantic import SecretStr
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI, Modality
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI

from tee_gateway.config import ProviderConfig
from tee_gateway.model_registry import get_model_config, resolve_image_size

logger = logging.getLogger(__name__)

# HTTP Client Configuration
ANTHROPIC_TIMEOUT = 120.0

_TIMEOUT = httpx.Timeout(
    timeout=120.0,
    connect=15.0,
    read=60.0,
    write=60.0,
    pool=10.0,
)

_LIMITS = httpx.Limits(
    max_keepalive_connections=10,
    max_connections=50,
    keepalive_expiry=60 * 20,  # 20 minutes
)

# BytePlus ModelArk OpenAI-compatible endpoint (ap-southeast)
BYTEDANCE_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"

# Nous Research OpenAI-compatible inference endpoint (Nous Portal).
NOUS_BASE_URL = "https://inference-api.nousresearch.com/v1"

# Z.ai Model API OpenAI-compatible endpoint. The full chat URL is
# https://api.z.ai/api/paas/v4/chat/completions; ChatOpenAI appends
# /chat/completions to this base URL. Do not confuse this paid Model API with
# the subscription Coding Plan endpoint at /api/coding/paas/v4.
ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"

# Shared synchronous HTTP clients for each provider.
# Initialized to None; built by set_provider_config() after key injection.
openai_http_client: Optional[httpx.Client] = None
xai_http_client: Optional[httpx.Client] = None
bytedance_http_client: Optional[httpx.Client] = None
nous_http_client: Optional[httpx.Client] = None
zai_http_client: Optional[httpx.Client] = None


_provider_config: Optional[ProviderConfig] = None


def set_provider_config(config: ProviderConfig) -> None:
    """Store the provider config and rebuild HTTP clients. Called once after key injection."""
    global _provider_config, openai_http_client, xai_http_client, bytedance_http_client
    global nous_http_client, zai_http_client

    old_openai = openai_http_client
    old_xai = xai_http_client
    old_bytedance = bytedance_http_client
    old_nous = nous_http_client
    old_zai = zai_http_client

    openai_http_client = httpx.Client(
        base_url="https://api.openai.com/v1",
        headers={"Authorization": f"Bearer {config.openai_api_key or ''}"},
        timeout=_TIMEOUT,
        limits=_LIMITS,
        http2=True,
        follow_redirects=False,
    )
    xai_http_client = httpx.Client(
        base_url="https://api.x.ai/v1",
        headers={"Authorization": f"Bearer {config.xai_api_key or ''}"},
        timeout=_TIMEOUT,
        limits=_LIMITS,
        http2=True,
        follow_redirects=False,
    )
    bytedance_http_client = httpx.Client(
        base_url=BYTEDANCE_BASE_URL,
        headers={"Authorization": f"Bearer {config.bytedance_api_key or ''}"},
        timeout=_TIMEOUT,
        limits=_LIMITS,
        http2=True,
        follow_redirects=False,
    )
    nous_http_client = httpx.Client(
        base_url=NOUS_BASE_URL,
        headers={"Authorization": f"Bearer {config.nous_api_key or ''}"},
        timeout=_TIMEOUT,
        limits=_LIMITS,
        http2=True,
        follow_redirects=False,
    )
    zai_http_client = httpx.Client(
        base_url=ZAI_BASE_URL,
        headers={"Authorization": f"Bearer {config.zai_api_key or ''}"},
        timeout=_TIMEOUT,
        limits=_LIMITS,
        http2=True,
        follow_redirects=False,
    )

    get_chat_model_cached.cache_clear()
    _provider_config = config

    if old_openai is not None:
        old_openai.close()
    if old_xai is not None:
        old_xai.close()
    if old_bytedance is not None:
        old_bytedance.close()
    if old_nous is not None:
        old_nous.close()
    if old_zai is not None:
        old_zai.close()


def get_provider_config() -> Optional[ProviderConfig]:
    return _provider_config


def get_provider_from_model(model: str) -> str:
    """Infer provider from model name. Raises ValueError if model is unknown."""
    cfg = get_model_config(model)
    return cfg.provider


@lru_cache(maxsize=64)
def get_chat_model_cached(
    model: str,
    temperature: float,
    max_tokens: int,
    web_search: bool = False,
    image_quality: Optional[str] = None,
):
    """Get cached chat model instance using the injected ProviderConfig.

    Models are cached by (model, temperature, max_tokens, web_search,
    image_quality) tuple. Cache is cleared by set_provider_config() after key
    injection.

    When ``web_search`` is True, provider-specific native web search is enabled.
    Some providers (OpenAI, xAI) require search configuration at construction
    time; others (Anthropic, Google) enable it by binding a tool — see
    ``get_web_search_tool``. Providers without native web search ignore the flag.

    ``image_quality`` ("low" | "medium" | "high") selects the output resolution
    for inline image-output models (Gemini "nano banana"); it is ignored by
    text models and by image models without resolution control.
    """
    config = _provider_config
    if config is None:
        raise ValueError("Provider keys have not been initialized yet")

    cfg = get_model_config(model)
    provider = cfg.provider
    api_name = cfg.api_name
    effective_temp = (
        cfg.force_temperature if cfg.force_temperature is not None else temperature
    )

    logger.info(f"Creating cached chat model - Provider: {provider}, Model: {api_name}")

    if provider == "google":
        if not config.google_api_key:
            raise ValueError("google_api_key not set in ProviderConfig")

        # Image-generation models return images inline and do not support the
        # thinking budget; ask for both TEXT and IMAGE modalities so the model
        # may caption alongside the generated image.
        if cfg.image_output:
            # Map the requested quality tier to the model's resolution. Models
            # without resolution control (e.g. Gemini 2.5 Flash Image) return
            # None and use the provider default.
            image_size = resolve_image_size(model, image_quality)
            image_config = {"image_size": image_size} if image_size else None
            return ChatGoogleGenerativeAI(
                model=api_name,
                google_api_key=config.google_api_key,
                temperature=effective_temp,
                max_output_tokens=max_tokens,
                response_modalities=[Modality.TEXT, Modality.IMAGE],
                image_config=image_config,
            )

        return ChatGoogleGenerativeAI(
            model=api_name,
            google_api_key=config.google_api_key,
            temperature=effective_temp,
            max_output_tokens=max_tokens,
            thinking_budget=cfg.thinking_budget,
            include_thoughts=False if cfg.thinking_budget is not None else None,
        )

    elif provider == "openai":
        if not config.openai_api_key:
            raise ValueError("openai_api_key not set in ProviderConfig")

        if openai_http_client is None:
            raise RuntimeError("OpenAI HTTP client has not been initialized")

        # OpenAI's built-in web_search tool is only available through the
        # Responses API, so switch to it when web search is requested. The
        # responses/v1 output format surfaces web_search_call items in the
        # message content, which we count for billing.
        openai_kwargs: dict[str, Any] = {}
        if web_search:
            openai_kwargs["use_responses_api"] = True
            openai_kwargs["output_version"] = "responses/v1"

        return ChatOpenAI(
            model=api_name,
            temperature=effective_temp,
            max_tokens=max_tokens,
            http_client=openai_http_client,
            api_key=SecretStr(config.openai_api_key),
            streaming=True,
            stream_usage=True,
            **openai_kwargs,
        )  # type: ignore [call-arg]

    elif provider == "anthropic":
        if not config.anthropic_api_key:
            raise ValueError("anthropic_api_key not set in ProviderConfig")

        # Opus 4.7+ rejects `temperature` outright (HTTP 400). Pass None so
        # langchain-anthropic strips the field from the outgoing payload.
        anthropic_temperature: Optional[float] = (
            effective_temp if cfg.supports_temperature else None
        )

        return ChatAnthropic(
            model=api_name,
            api_key=SecretStr(config.anthropic_api_key),
            temperature=anthropic_temperature,
            max_tokens=max_tokens,
            timeout=ANTHROPIC_TIMEOUT,
            streaming=True,
            stream_usage=True,
        )  # type: ignore [call-arg]

    elif provider == "x-ai":
        if not config.xai_api_key:
            raise ValueError("xai_api_key not set in ProviderConfig")

        if xai_http_client is None:
            raise RuntimeError("XAI HTTP client has not been initialized")

        # xAI deprecated Live Search on Chat Completions. Web search now lives on
        # the Responses API and is enabled by binding the built-in web_search tool.
        xai_kwargs: dict[str, Any] = {}
        if web_search:
            xai_kwargs["use_responses_api"] = True

        return ChatXAI(
            model=api_name,
            api_key=SecretStr(config.xai_api_key),
            temperature=effective_temp,
            max_tokens=max_tokens,
            http_client=xai_http_client,
            streaming=True,
            stream_usage=True,
            **xai_kwargs,
        )

    elif provider == "bytedance":
        if not config.bytedance_api_key:
            raise ValueError("bytedance_api_key not set in ProviderConfig")

        if bytedance_http_client is None:
            raise RuntimeError("ByteDance HTTP client has not been initialized")

        return ChatOpenAI(
            model=api_name,
            temperature=effective_temp,
            max_tokens=max_tokens,
            http_client=bytedance_http_client,
            api_key=SecretStr(config.bytedance_api_key),
            base_url=BYTEDANCE_BASE_URL,
            streaming=True,
            stream_usage=True,
        )  # type: ignore [call-arg]

    elif provider == "nous":
        if not config.nous_api_key:
            raise ValueError("nous_api_key not set in ProviderConfig")

        if nous_http_client is None:
            raise RuntimeError("Nous HTTP client has not been initialized")

        return ChatOpenAI(
            model=api_name,
            temperature=effective_temp,
            max_tokens=max_tokens,
            http_client=nous_http_client,
            api_key=SecretStr(config.nous_api_key),
            base_url=NOUS_BASE_URL,
            streaming=True,
            stream_usage=True,
        )  # type: ignore [call-arg]

    elif provider == "zai":
        if not config.zai_api_key:
            raise ValueError("zai_api_key not set in ProviderConfig")

        if zai_http_client is None:
            raise RuntimeError("Z.ai HTTP client has not been initialized")

        return ChatOpenAI(
            model=api_name,
            temperature=effective_temp,
            max_tokens=max_tokens,
            http_client=zai_http_client,
            api_key=SecretStr(config.zai_api_key),
            base_url=ZAI_BASE_URL,
            streaming=True,
            stream_usage=True,
        )  # type: ignore [call-arg]

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# Endpoint path appended to each provider's OpenAI-compatible base URL.
_IMAGE_GENERATION_PATH = "/images/generations"


def generate_images(
    model: str, prompt: str, n: int = 1, quality: Optional[str] = None
) -> tuple[list[str], int]:
    """Generate images via a provider's OpenAI-compatible images endpoint.

    Unlike Gemini's inline-image chat models, xAI (Aurora), ByteDance
    (Seedream), and Z.ai (GLM-Image) expose image generation through a dedicated
    ``POST /images/generations`` endpoint. We request ``b64_json`` so the image
    bytes ride inline inside the OHTTP/TEE envelope for providers that support
    it. Z.ai returns temporary image URLs only.

    ``quality`` ("low" | "medium" | "high") selects the output resolution where
    the model supports it (Seedream/Seedance: 1K/2K/4K, Z.ai: pixel dimensions).
    xAI Grok exposes no resolution control and ignores the option.

    Returns ``(data_uris, image_count)`` where each entry is a ``data:`` URI. The
    count is used for per-image billing. Falls back to provider-returned URLs if
    a provider ignores ``b64_json``.
    """
    cfg = get_model_config(model)
    provider = cfg.provider

    # Map the requested quality tier to a provider-specific size string. None
    # means "no quality requested" or "model has no resolution control"; in both
    # cases the provider default below is left in place.
    size_override = resolve_image_size(model, quality)

    if provider == "x-ai":
        client = xai_http_client
    elif provider == "bytedance":
        client = bytedance_http_client
    elif provider == "zai":
        client = zai_http_client
    else:
        raise ValueError(
            f"Provider {provider!r} does not support the image-generation endpoint"
        )

    if client is None:
        raise RuntimeError(f"{provider} HTTP client has not been initialized")

    # n is clamped to the OpenAI-compatible providers' documented 1..10 range.
    # Z.ai's GLM-Image and ByteDance Seedance endpoints don't document n/
    # response_format support, so keep their payloads to documented fields.
    count = max(1, min(int(n), 10))
    payload: dict[str, Any] = {
        "model": cfg.api_name,
        "prompt": prompt,
    }
    if provider == "zai":
        payload["size"] = size_override or "1280x1280"
    elif provider == "bytedance" and cfg.api_name.startswith("ep-"):
        # Seedance deployment endpoints use URL format and require extra params.
        payload["response_format"] = "url"
        payload["sequential_image_generation"] = "disabled"
        payload["watermark"] = False
        payload["size"] = size_override or "2K"
        payload["stream"] = False
    else:
        payload["n"] = count
        payload["response_format"] = "b64_json"
        # Seedream accepts a size preset; xAI Grok has no resolution control, so
        # size_override is None there and the provider default is used.
        if size_override:
            payload["size"] = size_override

    logger.info(
        "Generating %d image(s) - Provider: %s, Model: %s",
        count,
        provider,
        cfg.api_name,
    )
    resp = client.post(_IMAGE_GENERATION_PATH, json=payload)
    resp.raise_for_status()
    data = resp.json().get("data", []) or []

    images: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        b64 = item.get("b64_json")
        if b64:
            images.append(f"data:image/jpeg;base64,{b64}")
            continue
        url = item.get("url")
        if url:
            images.append(url)

    return images, len(images)


def _parse_data_uri(uri: str) -> Optional[tuple[str, str]]:
    """Parse a ``data:<mime>;base64,<data>`` URI into ``(mime_type, base64_data)``.

    Returns ``None`` if the string is not a base64 data URI.
    """
    if not isinstance(uri, str) or not uri.startswith("data:"):
        return None
    try:
        header, data = uri.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header:
        return None
    mime_type = header[len("data:") :].split(";", 1)[0]
    return mime_type, data


def _convert_content_part(part: Any) -> Optional[Dict[str, Any]]:
    """Convert one OpenAI-format content part into a LangChain v1 standard content
    block (``text`` / ``image`` / ``file``).

    The standard blocks (``langchain_core.messages.content``) are translated into
    each provider's native API by the respective ``langchain-<provider>`` package,
    so a single representation works for Anthropic, OpenAI, Gemini and xAI. Returns
    ``None`` for empty or unrecognized parts.
    """
    if not isinstance(part, dict):
        text = str(part)
        return {"type": "text", "text": text} if text else None

    ptype = part.get("type")

    if ptype == "text":
        text = part.get("text", "") or ""
        return {"type": "text", "text": text} if text else None

    if ptype in ("image_url", "image"):
        image_url = part.get("image_url", part)
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not url:
            # Already-standard image block carrying base64 directly.
            if part.get("base64"):
                block: Dict[str, Any] = {"type": "image", "base64": part["base64"]}
                if part.get("mime_type"):
                    block["mime_type"] = part["mime_type"]
                return block
            return None
        parsed = _parse_data_uri(url)
        if parsed:
            mime_type, data = parsed
            return {"type": "image", "base64": data, "mime_type": mime_type}
        return {"type": "image", "url": url}

    if ptype in ("file", "input_file"):
        file_obj = part.get("file", part)
        if not isinstance(file_obj, dict):
            file_obj = {}
        file_id = file_obj.get("file_id") or part.get("file_id")
        if file_id:
            return {"type": "file", "file_id": file_id}

        filename = file_obj.get("filename") or part.get("filename")
        file_data = (
            file_obj.get("file_data") or file_obj.get("base64") or part.get("base64")
        )
        if file_data:
            file_mime: Optional[str]
            parsed_file = _parse_data_uri(file_data)
            if parsed_file:
                file_mime, file_b64 = parsed_file
            else:
                file_mime = part.get("mime_type") or file_obj.get("mime_type")
                file_b64 = file_data
            block = {"type": "file", "base64": file_b64}
            if file_mime:
                block["mime_type"] = file_mime
            # OpenAI requires a filename for file uploads; carry it through so
            # langchain-openai doesn't substitute a placeholder.
            if filename:
                block["filename"] = filename
            return block

        file_url = file_obj.get("file_url") or file_obj.get("url") or part.get("url")
        if file_url:
            block = {"type": "file", "url": file_url}
            if filename:
                block["filename"] = filename
            return block
        return None

    # Unknown part type: best-effort text extraction.
    text = part.get("text", "") or ""
    return {"type": "text", "text": text} if text else None


def canonical_user_content(content: Any) -> Any:
    """Canonicalize user-message content for request hashing.

    Plain-string content is returned unchanged. For multimodal content (a list of
    parts), text is kept verbatim and each attachment is reduced to its type and
    filename — the inline bytes are dropped, never hashed, so the signed request
    stays small. The attachment bytes still travel inside the encrypted transport;
    the request hash just commits to which files were sent.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    canonical = []
    for part in content:
        if not isinstance(part, dict):
            canonical.append({"type": "text", "text": str(part)})
            continue
        if part.get("type") == "text":
            canonical.append({"type": "text", "text": part.get("text", "") or ""})
            continue
        # Attachment: commit only to its type and filename, never the bytes.
        entry: Dict[str, Any] = {"type": part.get("type")}
        file_obj = part.get("file")
        filename = (
            file_obj.get("filename") if isinstance(file_obj, dict) else None
        ) or part.get("filename")
        if filename:
            entry["filename"] = filename
        canonical.append(entry)
    return canonical


def _normalize_user_content_parts(content: list) -> list:
    """Pass OpenAI content parts through to LangChain mostly unchanged.

    Text and image parts already convert correctly to every provider's native
    API in their OpenAI form, so they are forwarded as-is. Only ``file`` /
    ``input_file`` parts are rewritten into LangChain standard file blocks: the
    raw OpenAI ``{"type": "file", "file": {...}}`` shape is passed straight
    through to providers like Anthropic, which expect a ``document`` block and
    would otherwise reject it. Primitive (non-dict) parts are wrapped as text.
    """
    normalized: List[Any] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") in ("file", "input_file"):
                block = _convert_content_part(part)
                normalized.append(block if block is not None else part)
            else:
                normalized.append(part)
        else:
            normalized.append({"type": "text", "text": str(part)})
    return normalized


class AttachmentValidationError(ValueError):
    """Raised when a request's attachments aren't supported by the target model."""


def get_model_capabilities(model: str) -> Dict[str, Any]:
    """Return the LangChain capability profile for a model (``image_inputs``,
    ``pdf_inputs``, ...), or ``{}`` when the model has no profile data.

    Reads the public ``.profile`` attribute of the instantiated chat model, which
    each ``langchain-<provider>`` package populates from maintained model data.
    """
    try:
        chat = get_chat_model_cached(model, 0.0, 16)
        return getattr(chat, "profile", None) or {}
    except Exception:
        return {}


def _iter_content_parts(messages: list) -> Generator[Dict[str, Any], None, None]:
    for msg in messages:
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    yield part


def validate_attachments(messages: list, model: str) -> None:
    """Reject attachments the target model can't handle.

    Fails *open*: a modality is only rejected when the model's profile explicitly
    marks it unsupported, so models without profile data are never wrongly blocked
    (the provider would still reject a truly unsupported combination). Raises
    ``AttachmentValidationError``.
    """
    # Fast path: if the request contains no multimodal parts, skip model lookup.
    for part in _iter_content_parts(messages):
        if part.get("type") in ("image_url", "image", "file", "input_file"):
            break
    else:
        return

    caps = get_model_capabilities(model)
    image_supported = caps.get("image_inputs")
    pdf_supported = caps.get("pdf_inputs")

    if image_supported is not False and pdf_supported is not False:
        return  # model accepts every modality; nothing to gate

    for part in _iter_content_parts(messages):
        block = _convert_content_part(part)
        if block is None:
            continue
        if block["type"] == "image" and image_supported is False:
            raise AttachmentValidationError(
                f"Model {model!r} does not support image attachments."
            )
        if block["type"] == "file" and pdf_supported is False:
            raise AttachmentValidationError(
                f"Model {model!r} does not support document attachments."
            )


def convert_messages(messages: list) -> List[Any]:
    """Convert OpenAI-format message objects or dicts to LangChain message objects."""
    langchain_messages: List[BaseMessage] = []

    for msg in messages:
        # Support both OpenAPI model objects and plain dicts
        if isinstance(msg, dict):
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if content is None:
                content = ""
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
            name = msg.get("name")
        else:
            role = getattr(msg, "role", "").lower()
            content = getattr(msg, "content", "")
            if content is None:
                content = ""
            tool_calls = getattr(msg, "tool_calls", None)
            tool_call_id = getattr(msg, "tool_call_id", None)
            name = getattr(msg, "name", None)

        if role == "system":
            langchain_messages.append(SystemMessage(content=content))

        elif role == "user":
            # content may be a string or a list of multimodal content parts
            # (text / image / file). Pass parts through as-is (file parts are
            # normalized to standard LangChain blocks) so the providers handle
            # the native conversion.
            if isinstance(content, list):
                content = _normalize_user_content_parts(content)
            langchain_messages.append(HumanMessage(content=content))

        elif role == "assistant":
            if tool_calls:
                langchain_tool_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        args = func.get("arguments", "{}")
                        tc_id = tc.get("id", "")
                        func_name = func.get("name", "")
                    else:
                        func = getattr(tc, "function", None)
                        args = func.arguments if func else "{}"
                        tc_id = getattr(tc, "id", "")
                        func_name = func.name if func else ""

                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}

                    langchain_tool_calls.append(
                        {
                            "name": func_name,
                            "args": args,
                            "id": tc_id,
                            "type": "function",
                        }
                    )

                langchain_messages.append(
                    AIMessage(
                        content=content,
                        tool_calls=langchain_tool_calls,
                    )
                )
            else:
                langchain_messages.append(AIMessage(content=content))

        elif role == "tool":
            langchain_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id or "",
                    name=name or "",
                )
            )

        elif role == "function":
            # Legacy function role: treat as tool message
            langchain_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id="",
                    name=name or "",
                )
            )

    return langchain_messages


def extract_usage(response) -> Optional[Dict[str, int]]:
    """Extract token usage from a LangChain response object."""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        # Thinking tokens, when present, are folded into output_tokens but also
        # broken out here. Image-output models bill them at the cheaper
        # text/thinking rate (see compute_session_cost), so surface them.
        details = meta.get("output_token_details") or {}
        return {
            "prompt_tokens": meta.get("input_tokens", 0),
            "completion_tokens": meta.get("output_tokens", 0),
            "total_tokens": meta.get("total_tokens", 0),
            "reasoning_tokens": details.get("reasoning", 0),
        }
    return None


# Anthropic's server-side web search tool. The dated type string is the current
# tool version; max_uses caps searches per request to bound cost.
ANTHROPIC_WEB_SEARCH_TOOL: Dict[str, Any] = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}


def get_web_search_tool(provider: str) -> Optional[Dict[str, Any]]:
    """Return the provider-specific web search tool spec to bind, or None.

    OpenAI/Anthropic/Google/xAI enable web search by binding a built-in tool.
    xAI requires this on the Responses API path; its old Live Search
    ``search_parameters`` path is deprecated.
    """
    if provider == "openai":
        return {"type": "web_search"}
    if provider == "anthropic":
        return dict(ANTHROPIC_WEB_SEARCH_TOOL)
    if provider == "google":
        return {"google_search": {}}
    if provider == "x-ai":
        return {"type": "web_search"}
    # bytedance has no native web search.
    return None


def extract_web_search_count(message) -> int:
    """Best-effort count of billable web-search units the model performed.

    Each provider reports search activity differently and bills a different
    unit, so we count the unit that matches that provider's list price:

    - OpenAI (Responses API): ``web_search_call`` content blocks (per call)
    - Anthropic: ``server_tool_use`` web_search content blocks (per request)
    - xAI/OpenAI Responses API: ``web_search_call`` content blocks (per call)
      or legacy xAI ``citations`` in additional_kwargs (per source)
    - Google: 1 per grounded response (per grounded request)

    Works on a completed AIMessage or an accumulated AIMessageChunk.
    """
    if message is None:
        return 0

    count = 0

    content = getattr(message, "content", None)
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "web_search_call":
                count += 1
            elif btype == "server_tool_use" and block.get("name") == "web_search":
                count += 1

    # xAI returns the list of sources it grounded on as `citations`.
    additional_kwargs = getattr(message, "additional_kwargs", None) or {}
    citations = additional_kwargs.get("citations")
    if citations:
        count += len(citations)

    # Google reports grounding via response_metadata; billed per grounded request.
    response_metadata = getattr(message, "response_metadata", None) or {}
    if response_metadata.get("grounding_metadata"):
        count += 1

    return count
