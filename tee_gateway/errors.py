"""Self-describing error payloads for the gateway's inference endpoints.

Every failure that reaches a client used to be ``{"error": str(e),
"exception_type": ...}`` with HTTP 500 — whether the enclave had a bug, the
provider was overloaded, or the provider's TCP connection reset. From the
relay's and browser's side those are different incidents: a provider 529 is
retryable and not ours, a ``ValueError`` in our request shaping is a bug and
retrying it is pointless. This module classifies an exception once, so the
non-streaming error response, the stream-setup error response and the
in-band SSE error frame all report the same thing:

``error``
    Human-readable detail — the provider's message when there is one.
``exception_type``
    The exception class name (content-free bucket key; unchanged).
``source``
    ``"provider"`` when a model provider answered with an error or could not
    be reached; ``"gateway"`` for anything that failed inside the enclave.
``provider_status``
    The HTTP status the provider returned, when it returned one.
``retryable``
    Whether resending the same request has a real chance of succeeding
    (provider 5xx/429/overload, connection resets, timeouts).

:func:`http_status_for` maps the same classification onto the outer status:
provider failures are ``502``/``504`` (we are the gateway that got a bad or no
answer from upstream), gateway failures stay ``500``.

Nothing here quotes the request: provider messages describe *their* refusal
(model name, parameter names, rate limits), not the prompt.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import httpx

# Truncation guard for provider messages that echo a whole request body.
_MESSAGE_CHARS = 600
_WHITESPACE_RE = re.compile(r"\s+")

SOURCE_PROVIDER = "provider"
SOURCE_GATEWAY = "gateway"

# Provider statuses after which a resend of the identical request can succeed.
_RETRYABLE_PROVIDER_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


def _provider_status(exc: BaseException) -> Optional[int]:
    """The HTTP status a provider SDK exception carries, or ``None``.

    Duck-typed on purpose: openai/anthropic/xai expose ``status_code``,
    google-genai ``status`` (an int on APIError) or ``code``, and a raw httpx
    failure has ``response.status_code``. Importing every SDK's exception
    class here would couple this module to each provider package.
    """
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and 100 <= status <= 599:
        return status
    return None


def _is_provider_exception(exc: BaseException) -> bool:
    """Whether ``exc`` came out of a provider client rather than our code.

    Matched by module: every provider SDK we route through lives under one
    of these roots, and LangChain re-raises the SDK's own exception types.
    httpx errors are treated as provider failures because the only outbound
    httpx traffic from an inference request is to a provider.
    """
    if isinstance(exc, httpx.HTTPError):
        return True
    module = type(exc).__module__ or ""
    return module.split(".", 1)[0] in {
        "openai",
        "anthropic",
        "google",
        "xai_sdk",
        "langchain_google_genai",
        "grpc",
        "aiohttp",
    }


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return True
    return "timeout" in type(exc).__name__.lower()


def _is_connection_failure(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return True
    return "connection" in type(exc).__name__.lower()


def _message(exc: BaseException, fallback: str) -> str:
    text = _WHITESPACE_RE.sub(" ", str(exc)).strip()
    if not text:
        return fallback
    if len(text) > _MESSAGE_CHARS:
        text = text[:_MESSAGE_CHARS] + "…"
    return text


def describe_exception(
    exc: BaseException, *, fallback: str = "Request processing failed"
) -> dict[str, Any]:
    """Build the error payload for ``exc`` (see module docstring)."""
    provider_status = _provider_status(exc) if _is_provider_exception(exc) else None
    from_provider = _is_provider_exception(exc)

    payload: dict[str, Any] = {
        "error": _message(exc, fallback),
        "exception_type": type(exc).__name__,
        "source": SOURCE_PROVIDER if from_provider else SOURCE_GATEWAY,
    }
    if provider_status is not None:
        payload["provider_status"] = provider_status

    if from_provider:
        payload["retryable"] = (
            provider_status in _RETRYABLE_PROVIDER_STATUSES
            if provider_status is not None
            else _is_timeout(exc) or _is_connection_failure(exc)
        )
    else:
        payload["retryable"] = False
    return payload


# Provider 4xx that are about *this gateway's* standing with the provider
# (its key, its plan, its quota), not about the request. They must not reach
# the client as-is: a 401/403 would read as the client's own credentials
# failing, and a 402 would be taken by the relay's x402 client for a payment
# challenge from this gateway.
_GATEWAY_ACCOUNT_STATUSES = frozenset({401, 402, 403, 407})


def http_status_for(exc: BaseException) -> int:
    """Outer HTTP status for a failed (non-streaming) inference request.

    * anything raised by the gateway's own code → ``500``;
    * a provider could not be reached → ``502``, or ``504`` for a timeout;
    * a provider answered 5xx → ``502``; ``429`` → ``503``; ``408`` → ``504``;
    * a provider refused the gateway's credentials or account → ``502``;
    * any other provider 4xx is about the request itself (too long, bad
      parameter, unknown model, oversize image) and is passed through
      unchanged, so the relay and browser do not retry it as a gateway
      failure — resending an invalid request cannot make it valid.
    """
    if not _is_provider_exception(exc):
        return 500
    status = _provider_status(exc)
    if status is None:
        return 504 if _is_timeout(exc) else 502
    if status == 429:
        return 503
    if status == 408:
        return 504
    if status >= 500 or status in _GATEWAY_ACCOUNT_STATUSES:
        return 502
    return status if status >= 400 else 502


def error_response(
    exc: BaseException, *, fallback: str = "Request processing failed"
) -> tuple[dict[str, Any], int]:
    """``(payload, status)`` for a Flask handler's error return."""
    return describe_exception(exc, fallback=fallback), http_status_for(exc)
