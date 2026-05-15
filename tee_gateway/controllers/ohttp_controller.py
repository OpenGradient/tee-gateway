"""
Oblivious HTTP endpoint for anonymous inference.

This handler is intentionally minimal: it does HPKE decapsulation, dispatches
the inner request to the existing chat-completions handler in-process (no
network hop), and HPKE-encapsulates the response. The inner JSON request is
identical to the standard /v1/chat/completions body.

Threat model nuances:
  * The relay in front of this endpoint sees the encapsulated ciphertext and
    the client IP, but no request content.
  * The enclave sees plaintext and the relay's IP, never the client's.
  * If the client's payload contains identifiers (cookies, ``user`` field,
    custom request IDs), unlinkability is broken at the application layer —
    we strip the obvious ones below.
  * Streaming is intentionally not supported on this endpoint. SSE would
    create per-chunk side channels (timing, length) that defeat the point of
    bundling everything into a single sealed response.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import connexion
from flask import Response

from tee_gateway import ohttp
from tee_gateway.tee_manager import get_tee_keys

logger = logging.getLogger(__name__)

# Per-thread plaintext of the in-flight OHTTP request, used to bridge the
# encrypted request/response to the x402 token-based cost calculator. The
# calculator runs from WSGI ``close()`` on the same worker thread (gunicorn
# sync workers), so a thread-local is the simplest reliable handoff.
_inner_local = threading.local()


def _stash_inner_context(request_json: dict, response_json: dict) -> None:
    _inner_local.value = {"request": request_json, "response": response_json}


def consume_inner_context() -> dict[str, Any] | None:
    """Pop the inner OHTTP plaintext for the current request. Called from the
    x402 cost calculator. Returns None if no OHTTP request was processed on
    this thread (i.e. the calculator was invoked for a non-OHTTP path)."""
    value = getattr(_inner_local, "value", None)
    if value is not None:
        # One-shot: clearing prevents accidental reuse across requests if the
        # same thread serves a non-OHTTP request next.
        _inner_local.value = None
    return value


OHTTP_MEDIA_TYPE = "message/ohttp-req"
OHTTP_RESPONSE_MEDIA_TYPE = "message/ohttp-res"

# Fields that can re-identify a client and have no role in inference. We drop
# them before forwarding to the inner handler — keeping them inside the
# encrypted envelope would only protect them from the relay, not from us or
# the upstream LLM provider.
_IDENTIFYING_FIELDS = ("user", "metadata", "x-request-id", "request_id")


def _scrub(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in _IDENTIFYING_FIELDS}


def create_anonymous_chat_completion():
    """POST /v1/ohttp — decrypt, dispatch, re-encrypt.

    Body: raw bytes (OHTTP-encapsulated request).
    Returns: raw bytes (OHTTP-encapsulated response) with Content-Type
    ``message/ohttp-res``.
    """
    req = connexion.request
    # Tolerate both Connexion's Flask request and a bare Flask request.
    raw_body: bytes = req.get_data(cache=False)
    if not raw_body:
        return _error(400, "empty body")

    tee = get_tee_keys()
    if tee.hpke_private_key is None:
        return _error(503, "anonymous inference not initialized")

    try:
        decap = ohttp.decapsulate_request(tee.hpke_private_key, raw_body)
    except Exception as exc:
        # Don't leak which step failed — clients can retry with a fresh
        # encapsulation, all observable failures look identical.
        logger.warning("OHTTP decapsulation failed: %s", type(exc).__name__)
        return _error(400, "malformed encapsulated request")

    try:
        inner_body = json.loads(decap.plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _error(400, "inner payload is not valid JSON")

    if not isinstance(inner_body, dict):
        return _error(400, "inner payload must be a JSON object")

    if inner_body.get("stream"):
        # Streaming is rejected on principle (see module docstring). Clients
        # who want low TTFT under anonymity should use a shorter max_tokens.
        return _error(400, "stream=true is not supported over OHTTP")

    inner_body = _scrub(inner_body)

    # Late import to avoid a circular dependency at module load (the chat
    # controller pulls in models that import this package).
    from tee_gateway.controllers.chat_controller import (
        _create_non_streaming_response,
        _parse_chat_request,
    )

    try:
        chat_request = _parse_chat_request(inner_body)
        inner_result = _create_non_streaming_response(chat_request)
    except Exception as exc:
        logger.error("inner inference failed under OHTTP: %s", exc, exc_info=True)
        inner_result = ({"error": "inference failed"}, 500)

    # _create_non_streaming_response returns either a dict or (body, status)
    if isinstance(inner_result, tuple):
        body_obj, status = inner_result
    else:
        body_obj, status = inner_result, 200

    # Hand the plaintext request + response to the x402 cost calculator so
    # the OHTTP request is priced by tokens like every other inference call.
    # Only stash on successful 2xx outcomes — error responses aren't billed.
    if 200 <= status < 300 and isinstance(body_obj, dict):
        _stash_inner_context(inner_body, body_obj)

    inner_json = json.dumps(
        {"status": status, "body": body_obj},
        separators=(",", ":"),
    ).encode("utf-8")

    sealed = ohttp.encapsulate_response(decap.response_key, decap.enc, inner_json)
    return Response(
        sealed,
        status=200,
        mimetype=OHTTP_RESPONSE_MEDIA_TYPE,
    )


def get_hpke_config():
    """GET /v1/ohttp/config — return the HPKE key configuration.

    Returns both an OHTTP-compliant binary key_config (base64) and the
    individual fields for clients that prefer to parse JSON. The same data is
    embedded inside the attestation document at /signing-key for clients that
    want to verify the binding to the enclave's PCRs in one step.
    """
    try:
        tee = get_tee_keys()
        return tee.get_hpke_config(), 200
    except Exception as exc:
        logger.error("HPKE config error: %s", exc, exc_info=True)
        return {"error": str(exc)}, 500


def _error(status: int, message: str) -> tuple[dict, int]:
    return {"error": message}, status
