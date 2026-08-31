"""
LLM routing backend for TEE-LLM-Router.

Provides LangChain model instantiation, provider routing, message conversion,
and shared HTTP clients. Replaces src/server.py as the direct LLM backend
called from the Flask/connexion controllers.
"""

import json
import logging
from collections.abc import Mapping
from typing import List, Dict, Optional, Any, Generator
from functools import cached_property, lru_cache

import anthropic
import httpx
from anthropic import AnthropicVertex, AsyncAnthropicVertex
from google.oauth2 import service_account
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
from tee_gateway.model_registry import get_model_config
from tee_gateway.moderation import configure_moderation_client
from tee_gateway.web_search import configure_exa_client

logger = logging.getLogger(__name__)

# HTTP Client Configuration
READ_TIMEOUT = 180.0

_TIMEOUT = httpx.Timeout(
    timeout=120.0,
    connect=15.0,
    read=READ_TIMEOUT,
    write=60.0,
    pool=30.0,
)

_LIMITS = httpx.Limits(
    max_keepalive_connections=10,
    max_connections=50,
    keepalive_expiry=45.0,
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

# Vertex AI's "global" endpoint dynamically routes to whichever region has
# capacity, is the only endpoint type without a 10% regional pricing premium,
# and is the only one that serves the newest Claude models — so it is the
# default unless the operator injects a specific gcp_location.
VERTEX_DEFAULT_LOCATION = "global"

# OAuth scope for Vertex AI requests, used when minting tokens from the
# injected service-account key.
_VERTEX_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

# Shared synchronous HTTP clients for each provider.
# Initialized to None; built by set_provider_config() after key injection.
openai_http_client: Optional[httpx.Client] = None
xai_http_client: Optional[httpx.Client] = None
bytedance_http_client: Optional[httpx.Client] = None
nous_http_client: Optional[httpx.Client] = None
zai_http_client: Optional[httpx.Client] = None


_provider_config: Optional[ProviderConfig] = None

# GCP Vertex AI routing state, derived from the injected service-account key.
# When populated, Anthropic (Claude) models are served through Vertex; the
# credentials object self-refreshes its OAuth token, so it is built once here.
# (Gemini stays on the direct API — its key already bills to a GCP project.)
_vertex_credentials: Optional[service_account.Credentials] = None
_vertex_project_id: Optional[str] = None
_vertex_location: str = VERTEX_DEFAULT_LOCATION


def _build_vertex_state(
    config: ProviderConfig,
) -> tuple[Optional[service_account.Credentials], Optional[str], str]:
    """Parse the injected GCP service-account key into Vertex routing state.

    Returns ``(credentials, project_id, location)`` — all-``None``/default when
    no key was injected. Raises ``ValueError`` on a malformed key so the
    /v1/keys call fails loudly instead of silently falling back to the direct
    provider APIs.
    """
    if not config.gcp_service_account_json:
        return None, None, VERTEX_DEFAULT_LOCATION

    try:
        sa_info = json.loads(config.gcp_service_account_json)
        credentials = service_account.Credentials.from_service_account_info(
            sa_info, scopes=list(_VERTEX_SCOPES)
        )
    except Exception as e:
        raise ValueError(f"Invalid gcp_service_account_json: {e}") from e

    project_id = config.gcp_project_id or sa_info.get("project_id")
    if not project_id:
        raise ValueError(
            "gcp_project_id not set and the service-account key carries no project_id"
        )
    location = config.gcp_location or VERTEX_DEFAULT_LOCATION
    return credentials, project_id, location


def vertex_enabled() -> bool:
    """Whether Anthropic (Claude) models are currently routed through Vertex AI."""
    return _vertex_credentials is not None


def set_provider_config(config: ProviderConfig) -> None:
    """Store the provider config and rebuild HTTP clients. Called once after key injection."""
    global _provider_config, openai_http_client, xai_http_client, bytedance_http_client
    global nous_http_client, zai_http_client
    global _vertex_credentials, _vertex_project_id, _vertex_location

    # Validate the GCP key before touching any shared state, so a malformed
    # key leaves the existing configuration fully intact.
    vertex_credentials, vertex_project_id, vertex_location = _build_vertex_state(config)

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
    # HTTP/1.1 only for BytePlus: its ap-southeast edge terminates idle/aged
    # HTTP/2 connections aggressively, and httpcore treats a single EOF on an
    # h2 connection as fatal for *all* streams multiplexed on it ("Server
    # disconnected" RemoteProtocolError), so one drop kills every in-flight
    # ByteDance request at once. HTTP/1.1 confines a dead connection to a
    # single request and its pool detects half-closed sockets before reuse.
    # Other providers stay on h2 — this failure mode is only observed here.
    bytedance_http_client = httpx.Client(
        base_url=BYTEDANCE_BASE_URL,
        headers={"Authorization": f"Bearer {config.bytedance_api_key or ''}"},
        timeout=_TIMEOUT,
        limits=_LIMITS,
        http2=False,
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

    # Web search runs inside the enclave against Exa rather than through any
    # provider's native tool, so its client is built here alongside them.
    configure_exa_client(config.exa_api_key)

    # Prompt moderation reuses the OpenAI key but gets its own client with a
    # much tighter timeout — it runs synchronously in front of every chat call.
    configure_moderation_client(config.openai_api_key)

    _vertex_credentials = vertex_credentials
    _vertex_project_id = vertex_project_id
    _vertex_location = vertex_location

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


class ChatAnthropicVertex(ChatAnthropic):
    """``ChatAnthropic`` served through GCP Vertex AI (Agent Platform).

    Keeps langchain-anthropic's whole request/response pipeline — tool binding,
    standard content blocks, streaming, usage extraction, the temperature-strip
    behavior — and swaps only the underlying SDK clients for
    ``anthropic.AnthropicVertex``, which rewrites ``/v1/messages`` calls into
    Vertex ``rawPredict``/``streamRawPredict`` requests (model moves into the
    URL, ``anthropic_version`` into the body) and signs them with the
    service-account OAuth token instead of an Anthropic API key.
    """

    vertex_project_id: str
    vertex_region: str = VERTEX_DEFAULT_LOCATION
    # google.auth credentials object (kept as Any: pydantic can't validate it)
    vertex_credentials: Any = None

    # AnthropicVertex is a sibling of Anthropic (both extend the same base
    # client) rather than a subclass, but exposes the identical messages
    # surface langchain-anthropic calls — hence the return-value ignores.
    @cached_property
    def _client(self) -> anthropic.Client:
        return AnthropicVertex(  # type: ignore [return-value]
            project_id=self.vertex_project_id,
            region=self.vertex_region,
            credentials=self.vertex_credentials,
            timeout=self.default_request_timeout,
            max_retries=self.max_retries,
        )

    @cached_property
    def _async_client(self) -> anthropic.AsyncClient:
        return AsyncAnthropicVertex(  # type: ignore [return-value]
            project_id=self.vertex_project_id,
            region=self.vertex_region,
            credentials=self.vertex_credentials,
            timeout=self.default_request_timeout,
            max_retries=self.max_retries,
        )


def non_streaming_invoke_kwargs(provider: str) -> Dict[str, Any]:
    """Kwargs for ``model.invoke()`` when a true non-streaming call is needed.

    Chat models are cached with ``streaming=True`` for the streaming endpoint,
    so ``invoke()`` streams internally and LangChain merges the chunks. xAI
    reports cumulative usage snapshots on every chunk, which that merge sums
    into inflated token counts — force a real non-streaming request so the
    provider returns one authoritative usage object.
    """
    return {"stream": False} if provider == "x-ai" else {}


@lru_cache(maxsize=64)
def get_chat_model_cached(
    model: str,
    temperature: float,
    max_tokens: int,
    force_responses_api: bool = False,
):
    """Get cached chat model instance using the injected ProviderConfig.

    Models are cached by (model, temperature, max_tokens, force_responses_api)
    tuple. Cache is cleared by set_provider_config() after key injection.

    Web search needs nothing here: it is a plain function tool the gateway binds
    and executes itself (see web_search.py), not a provider feature that has to
    be switched on at construction time.

    When ``force_responses_api`` is True, OpenAI models are constructed against
    the Responses API. The gpt-5.6 family rejects function tools combined with
    its default ``reasoning_effort`` on Chat Completions, so the chat controller
    sets this flag for those models when tools are bound. Ignored by non-OpenAI
    providers.
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
        # Gemini deliberately stays on the direct API: a paid-tier Gemini API
        # key already bills to its GCP project, so Vertex routing would buy
        # nothing here. Only Claude moves to Vertex (see the anthropic branch).
        if not config.google_api_key:
            raise ValueError("google_api_key not set in ProviderConfig")

        # Image-generation models return images inline and do not support the
        # thinking budget; ask for both TEXT and IMAGE modalities so the model
        # may caption alongside the generated image.
        if cfg.image_output:
            return ChatGoogleGenerativeAI(
                model=api_name,
                google_api_key=config.google_api_key,
                temperature=effective_temp,
                max_output_tokens=max_tokens,
                response_modalities=[Modality.TEXT, Modality.IMAGE],
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

        # Switch to the Responses API when the caller forced it because this
        # model rejects function tools + its default reasoning_effort on Chat
        # Completions (gpt-5.6 family).
        openai_kwargs: dict[str, Any] = {}
        if force_responses_api:
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
            # OpenAI retains request/response logs when `store` is unset (the
            # Responses API defaults it to true). Explicitly opt out so
            # prompts never persist outside the enclave.
            store=False,
            **openai_kwargs,
        )  # type: ignore [call-arg]

    elif provider == "anthropic":
        # Opus 4.7+ rejects `temperature` outright (HTTP 400). Pass None so
        # langchain-anthropic strips the field from the outgoing payload.
        anthropic_temperature: Optional[float] = (
            effective_temp if cfg.supports_temperature else None
        )

        # Prefer Vertex AI when a GCP service account was injected: every
        # Claude model in the registry is served on Vertex (dated snapshots
        # under their `@`-form vertex_api_name), billed against GCP.
        if vertex_enabled():
            assert _vertex_project_id is not None
            return ChatAnthropicVertex(
                model=cfg.vertex_api_name or api_name,
                temperature=anthropic_temperature,
                max_tokens=max_tokens,
                timeout=READ_TIMEOUT,
                streaming=True,
                stream_usage=True,
                vertex_project_id=_vertex_project_id,
                vertex_region=_vertex_location,
                vertex_credentials=_vertex_credentials,
            )  # type: ignore [call-arg]

        if not config.anthropic_api_key:
            raise ValueError(
                "Neither anthropic_api_key nor GCP credentials set in ProviderConfig"
            )

        return ChatAnthropic(
            model=api_name,
            api_key=SecretStr(config.anthropic_api_key),
            temperature=anthropic_temperature,
            max_tokens=max_tokens,
            timeout=READ_TIMEOUT,
            streaming=True,
            stream_usage=True,
        )  # type: ignore [call-arg]

    elif provider == "x-ai":
        if not config.xai_api_key:
            raise ValueError("xai_api_key not set in ProviderConfig")

        if xai_http_client is None:
            raise RuntimeError("XAI HTTP client has not been initialized")

        return ChatXAI(
            model=api_name,
            api_key=SecretStr(config.xai_api_key),
            temperature=effective_temp,
            max_tokens=max_tokens,
            http_client=xai_http_client,
            streaming=True,
            stream_usage=True,
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
    """Extract normalized usage from LangChain or raw provider metadata.

    LangChain normally exposes ``usage_metadata`` using input/output token
    names. OpenAI-compatible providers such as xAI can instead leave the raw
    ``token_usage`` object in ``response_metadata``. Supporting both shapes
    prevents a successful inference from losing its billing data.
    """

    def as_mapping(value: Any) -> Mapping[str, Any] | None:
        if isinstance(value, Mapping):
            return value
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            return dumped if isinstance(dumped, Mapping) else None
        return None

    metadata = as_mapping(getattr(response, "usage_metadata", None))
    if metadata is None:
        response_metadata = as_mapping(getattr(response, "response_metadata", None))
        if response_metadata is not None:
            metadata = as_mapping(
                response_metadata.get("token_usage") or response_metadata.get("usage")
            )

    if metadata is None:
        return None

    input_tokens = metadata.get("input_tokens", metadata.get("prompt_tokens"))
    output_tokens = metadata.get("output_tokens", metadata.get("completion_tokens"))
    total_tokens = metadata.get("total_tokens")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    prompt_tokens = int(input_tokens or 0)
    completion_tokens = int(output_tokens or 0)
    total = int(
        total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
    )

    reasoning_tokens = 0
    for details_key in ("output_token_details", "output_tokens_details"):
        details = as_mapping(metadata.get(details_key))
        if details is not None:
            reasoning_tokens = int(
                details.get("reasoning", details.get("reasoning_tokens", 0)) or 0
            )
            break
    if reasoning_tokens == 0:
        details = as_mapping(metadata.get("completion_tokens_details"))
        if details is not None:
            reasoning_tokens = int(details.get("reasoning_tokens", 0) or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
        "reasoning_tokens": reasoning_tokens,
    }


# Web search is not a provider feature here — the gateway serves it as its own
# Exa-backed endpoint (/v1/web_search, see web_search.py) that the client's tool
# loop calls, billed at one flat per-search rate.
