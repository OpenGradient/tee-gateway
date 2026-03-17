"""
LLM routing backend for TEE-LLM-Router.

Provides LangChain model instantiation, provider routing, message conversion,
and shared HTTP clients. Replaces src/server.py as the direct LLM backend
called from the Flask/connexion controllers.
"""

import json
import logging
from typing import List, Dict, Optional, Any
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI

from tee_gateway.config import ProviderConfig
from tee_gateway.model_registry import get_model_config

logger = logging.getLogger(__name__)

# HTTP Client Configuration
ANTHROPIC_TIMEOUT = 120.0

_TIMEOUT = httpx.Timeout(
    timeout=120.0,
    connect=15.0,
    read=15.0,
    write=30.0,
    pool=10.0,
)

_LIMITS = httpx.Limits(
    max_keepalive_connections=10,
    max_connections=50,
    keepalive_expiry=60 * 20,  # 20 minutes
)

# Shared synchronous HTTP clients for each provider.
# Initialized to None; built by set_provider_config() after key injection.
openai_http_client: Optional[httpx.Client] = None
xai_http_client: Optional[httpx.Client] = None


_provider_config: Optional[ProviderConfig] = None


def set_provider_config(config: ProviderConfig) -> None:
    """Store the provider config and rebuild HTTP clients. Called once after key injection."""
    global _provider_config, openai_http_client, xai_http_client

    old_openai = openai_http_client
    old_xai = xai_http_client

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

    get_chat_model_cached.cache_clear()
    _provider_config = config

    if old_openai is not None:
        old_openai.close()
    if old_xai is not None:
        old_xai.close()


def get_provider_config() -> Optional[ProviderConfig]:
    return _provider_config


def get_provider_from_model(model: str) -> str:
    """Infer provider from model name. Raises ValueError if model is unknown."""
    cfg = get_model_config(model)
    return cfg.provider


@lru_cache(maxsize=32)
def get_chat_model_cached(model: str, temperature: float, max_tokens: int):
    """Get cached chat model instance using the injected ProviderConfig.

    Models are cached by (model, temperature, max_tokens) tuple.
    Cache is cleared by set_provider_config() after key injection.
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

        return ChatOpenAI(
            model=api_name,
            temperature=effective_temp,
            max_tokens=max_tokens,
            http_client=openai_http_client,
            api_key=SecretStr(config.openai_api_key),
            streaming=True,
            stream_usage=True,
        )  # type: ignore [call-arg]

    elif provider == "anthropic":
        if not config.anthropic_api_key:
            raise ValueError("anthropic_api_key not set in ProviderConfig")

        return ChatAnthropic(
            model=api_name,
            api_key=SecretStr(config.anthropic_api_key),
            temperature=effective_temp,
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

        return ChatXAI(
            model=api_name,
            api_key=SecretStr(config.xai_api_key),
            temperature=effective_temp,
            max_tokens=max_tokens,
            http_client=xai_http_client,
            streaming=True,
            stream_usage=True,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def convert_messages(messages: list) -> List[Any]:
    """Convert OpenAI-format message objects or dicts to LangChain message objects."""
    langchain_messages: List[BaseMessage] = []

    for msg in messages:
        # Support both OpenAPI model objects and plain dicts
        if isinstance(msg, dict):
            role = msg.get("role", "").lower()
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
            name = msg.get("name")
        else:
            role = getattr(msg, "role", "").lower()
            content = getattr(msg, "content", "") or ""
            tool_calls = getattr(msg, "tool_calls", None)
            tool_call_id = getattr(msg, "tool_call_id", None)
            name = getattr(msg, "name", None)

        if role == "system":
            langchain_messages.append(SystemMessage(content=content))

        elif role == "user":
            # content may be a string or a list of content parts; handle both
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
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
        return {
            "prompt_tokens": meta.get("input_tokens", 0),
            "completion_tokens": meta.get("output_tokens", 0),
            "total_tokens": meta.get("total_tokens", 0),
        }
    return None
