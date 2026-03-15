"""
LLM routing backend for TEE-LLM-Router.

Provides LangChain model instantiation, provider routing, message conversion,
and shared HTTP clients. Replaces src/server.py as the direct LLM backend
called from the Flask/connexion controllers.
"""

import os
import json
import logging
import threading
from typing import List, Dict, Optional, Any
from functools import lru_cache

import httpx
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
# These are rebuilt by reinitialize_http_clients() after key injection.
openai_http_client = httpx.Client(
    base_url="https://api.openai.com/v1",
    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
    timeout=_TIMEOUT,
    limits=_LIMITS,
    http2=True,
    follow_redirects=False,
)

xai_http_client = httpx.Client(
    base_url="https://api.x.ai/v1",
    headers={"Authorization": f"Bearer {os.getenv('XAI_API_KEY', '')}"},
    timeout=_TIMEOUT,
    limits=_LIMITS,
    http2=True,
    follow_redirects=False,
)

_clients_lock = threading.Lock()


def reinitialize_http_clients():
    """Recreate shared HTTP clients with updated API keys and clear the model cache.

    Call this after injecting new provider API keys so that the new
    Authorization headers are picked up by subsequent requests.
    """
    global openai_http_client, xai_http_client

    with _clients_lock:
        old_openai = openai_http_client
        old_xai = xai_http_client

        openai_http_client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
            timeout=_TIMEOUT,
            limits=_LIMITS,
            http2=True,
            follow_redirects=False,
        )
        xai_http_client = httpx.Client(
            base_url="https://api.x.ai/v1",
            headers={"Authorization": f"Bearer {os.environ.get('XAI_API_KEY', '')}"},
            timeout=_TIMEOUT,
            limits=_LIMITS,
            http2=True,
            follow_redirects=False,
        )

        # Invalidate the model cache so the next request creates fresh
        # LangChain instances that read the newly set env-var keys.
        get_chat_model_cached.cache_clear()

        old_openai.close()
        old_xai.close()


def get_provider_from_model(model: str) -> str:
    """Infer provider from model name. Raises ValueError if model is unknown."""
    cfg = get_model_config(model)
    return cfg.provider


@lru_cache(maxsize=32)
def get_chat_model_cached(model: str, temperature: float, max_tokens: int):
    """Get cached chat model instance using environment API keys.

    Models are cached by (model, temperature, max_tokens) tuple.
    Call reinitialize_http_clients() to clear this cache after key injection.
    """
    cfg = get_model_config(model)
    provider = cfg.provider
    api_name = cfg.api_name
    effective_temp = (
        cfg.force_temperature if cfg.force_temperature is not None else temperature
    )

    logger.info(f"Creating cached chat model - Provider: {provider}, Model: {api_name}")

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        return ChatGoogleGenerativeAI(
            model=api_name,
            google_api_key=api_key,
            temperature=effective_temp,
            max_output_tokens=max_tokens,
            thinking_budget=cfg.thinking_budget,
            include_thoughts=False if cfg.thinking_budget is not None else None,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        return ChatOpenAI(
            model=api_name,
            api_key=api_key,
            temperature=effective_temp,
            max_tokens=max_tokens,
            http_client=openai_http_client,
            streaming=True,
            stream_usage=True,
        )

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        return ChatAnthropic(
            model=api_name,
            api_key=api_key,
            temperature=effective_temp,
            max_tokens=max_tokens,
            timeout=ANTHROPIC_TIMEOUT,
            streaming=True,
            stream_usage=True,
        )

    elif provider == "x-ai":
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not found in environment")

        return ChatXAI(
            model=api_name,
            api_key=api_key,
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
