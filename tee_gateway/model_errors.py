"""Helpers for turning provider model failures into client-facing errors."""

from __future__ import annotations

from typing import Any

from tee_gateway.model_registry import get_model_config


def exception_status_code(exc: Exception) -> int | None:
    """Extract an HTTP/gRPC-ish status code from common provider SDK exceptions."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status

    code_attr = getattr(exc, "code", None)
    code_value: Any
    if callable(code_attr):
        try:
            code_value = code_attr()
        except Exception:
            code_value = None
    else:
        code_value = code_attr

    if isinstance(code_value, int):
        return code_value
    if getattr(code_value, "name", None) == "NOT_FOUND":
        return 404

    return None


def is_upstream_model_not_served(exc: Exception) -> bool:
    """Return True when a provider says a registered model is not available."""
    if exception_status_code(exc) == 404:
        return True

    exc_type = type(exc).__name__.lower()
    if "notfound" in exc_type or "not_found" in exc_type:
        return True

    message = str(exc).lower()
    model_missing_phrases = (
        "model not found",
        "model is not found",
        "model does not exist",
        "model_not_found",
        "not found for model",
        "not served",
        "is not supported for",
    )
    return any(phrase in message for phrase in model_missing_phrases)


def upstream_model_not_served_payload(model: str, exc: Exception) -> dict[str, Any]:
    """Build an OpenAI-style-ish payload for a provider model availability miss."""
    payload: dict[str, Any] = {
        "error": (
            f"Upstream provider does not currently serve model '{model}'. "
            "The model may have been removed, renamed, or disabled upstream."
        ),
        "code": "model_not_served_upstream",
        "model": model,
        "exception_type": type(exc).__name__,
        "upstream_error": str(exc) or type(exc).__name__,
    }

    try:
        cfg = get_model_config(model)
    except Exception:
        return payload

    payload["provider"] = cfg.provider
    payload["api_name"] = cfg.api_name
    return payload
