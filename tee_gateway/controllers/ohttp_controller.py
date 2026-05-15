"""
Oblivious HTTP endpoint for anonymous inference.

This handler is intentionally a thin shell: it does HPKE decapsulation and
then re-issues the inner request as a real WSGI sub-request against the
enclave's own ``/v1/chat/completions``. That means x402 payment handling,
the pre-inference pricing gate, LangChain routing, the post-inference cost
calculator and TEE response signing all execute via the same code paths as
the public chat endpoint — no duplicate routing tables, no thread-local
side channels, no parallel pricing logic. ``/v1/ohttp`` itself is NOT
gated by x402: payment travels inside the sealed envelope as an
``x-payment`` header on the inner request, and the gating happens
naturally when the sub-request hits the chat endpoint.

Wire format of the (HPKE-decrypted) inner payload — a JSON object:
    {
      "x-payment": "<base64 x402 payment payload, optional>",
      "body":      { ... standard /v1/chat/completions JSON body ... }
    }

Wire format of the (pre-HPKE) inner response:
    {
      "status":  <int>,
      "headers": { "x-payment-response": "...", "x-upto-session": "..." },
      "body":    <parsed JSON object or string>
    }

Threat model nuances:
  * The relay in front of this endpoint sees the encapsulated ciphertext
    and the client IP, but no request content or payment header.
  * The enclave sees plaintext and the relay's IP, never the client's.
  * If the inner JSON body contains identifiers (``user``, cookies,
    custom request IDs), unlinkability is broken at the application
    layer — we strip the obvious ones below.
  * Streaming is intentionally rejected; the inner sub-request must
    return a single sealed response.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

from flask import Response, current_app, request as flask_request

from tee_gateway import ohttp
from tee_gateway.tee_manager import get_tee_keys

logger = logging.getLogger(__name__)

OHTTP_MEDIA_TYPE = "message/ohttp-req"
OHTTP_RESPONSE_MEDIA_TYPE = "message/ohttp-res"

# Fields that can re-identify a client and have no role in inference. We drop
# them before forwarding to the inner handler — keeping them inside the
# encrypted envelope would only protect them from the relay, not from us or
# the upstream LLM provider.
_IDENTIFYING_FIELDS = ("user", "metadata", "x-request-id", "request_id")

# Response headers we propagate from the inner /v1/chat/completions response
# back to the client (encrypted). Includes x402 settlement metadata and
# anything the standard chat endpoint exposes via the TEE-signed response.
_FORWARDED_HEADER_PREFIXES = ("x-payment", "x-upto", "x-settlement", "x-tee")
_FORWARDED_HEADER_NAMES = ("www-authenticate",)


def _scrub(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in _IDENTIFYING_FIELDS}


def _should_forward_header(name: str) -> bool:
    lower = name.lower()
    return lower in _FORWARDED_HEADER_NAMES or any(
        lower.startswith(p) for p in _FORWARDED_HEADER_PREFIXES
    )


def create_anonymous_chat_completion():
    """POST /v1/ohttp — decrypt, sub-dispatch to /v1/chat/completions, re-encrypt."""
    raw_body: bytes = flask_request.get_data(cache=False)
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
        envelope = json.loads(decap.plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _seal_inner(decap, 400, {}, {"error": "inner payload is not valid JSON"})

    if not isinstance(envelope, dict):
        return _seal_inner(
            decap, 400, {}, {"error": "inner payload must be a JSON object"}
        )

    body_obj = envelope.get("body")
    if not isinstance(body_obj, dict):
        return _seal_inner(
            decap, 400, {}, {"error": "inner 'body' must be a JSON object"}
        )

    if body_obj.get("stream"):
        # Streaming is rejected on principle: SSE re-introduces per-chunk
        # timing/length side channels that defeat the point of sealing
        # everything into one response.
        return _seal_inner(
            decap, 400, {}, {"error": "stream=true is not supported over OHTTP"}
        )

    body_obj = _scrub(body_obj)
    body_bytes = json.dumps(body_obj, separators=(",", ":")).encode("utf-8")

    payment_header = envelope.get("x-payment")
    if payment_header is not None and not isinstance(payment_header, str):
        return _seal_inner(
            decap, 400, {}, {"error": "'x-payment' must be a string if present"}
        )

    status_code, response_headers, response_body = _wsgi_subrequest(
        path="/v1/chat/completions",
        body_bytes=body_bytes,
        payment_header=payment_header,
    )

    return _seal_inner(decap, status_code, response_headers, response_body)


def _wsgi_subrequest(
    path: str,
    body_bytes: bytes,
    payment_header: str | None,
) -> tuple[int, dict[str, str], Any]:
    """Issue an in-process WSGI request through the app's full middleware stack.

    Returns ``(status_code, forwarded_headers, parsed_body_or_text)``. The
    parsed body is the decoded JSON object on JSON responses, otherwise the
    raw response text. We invoke ``current_app.wsgi_app`` directly so the
    x402 payment middleware (which wraps ``wsgi_app`` at injection time)
    runs the same way it would for an external HTTP request to the same
    path — including the pre-inference pricing gate, payment verification,
    cost settlement and TEE response signing.
    """
    outer_env = flask_request.environ
    sub_env: dict[str, Any] = {
        k: v
        for k, v in outer_env.items()
        if k.startswith("wsgi.")
        or k in ("SERVER_NAME", "SERVER_PORT", "SERVER_PROTOCOL", "HTTP_HOST")
    }
    sub_env.update(
        {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": path,
            "RAW_URI": path,
            "REQUEST_URI": path,
            "SCRIPT_NAME": "",
            "QUERY_STRING": "",
            "CONTENT_TYPE": "application/json",
            "CONTENT_LENGTH": str(len(body_bytes)),
            "wsgi.input": io.BytesIO(body_bytes),
        }
    )
    if payment_header:
        sub_env["HTTP_X_PAYMENT"] = payment_header

    captured: dict[str, Any] = {"status": "500 Internal Server Error", "headers": []}

    def _start_response(status: str, headers: list, exc_info: Any = None):
        captured["status"] = status
        captured["headers"] = headers
        return lambda _chunk: None

    iterator = current_app.wsgi_app(sub_env, _start_response)
    body_chunks: list[bytes] = []
    try:
        for chunk in iterator:
            if chunk:
                body_chunks.append(chunk)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            # Triggers x402's post-response settlement (StreamingSessionResponse.close).
            close()

    status_code = int(captured["status"].split(" ", 1)[0])
    forwarded_headers = {
        name: value
        for name, value in captured["headers"]
        if _should_forward_header(name)
    }

    raw_body = b"".join(body_chunks)
    parsed_body: Any
    if not raw_body:
        parsed_body = ""
    else:
        try:
            parsed_body = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed_body = raw_body.decode("utf-8", errors="replace")

    return status_code, forwarded_headers, parsed_body


def _seal_inner(
    decap: ohttp.DecapsulatedRequest,
    status_code: int,
    headers: dict[str, str],
    body: Any,
) -> Response:
    """Encapsulate a ``{status, headers, body}`` triple as an OHTTP response."""
    plaintext = json.dumps(
        {"status": status_code, "headers": headers, "body": body},
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    sealed = ohttp.encapsulate_response(decap.response_key, decap.enc, plaintext)
    return Response(sealed, status=200, mimetype=OHTTP_RESPONSE_MEDIA_TYPE)


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
    """Plaintext error response (not sealed) — only used before HPKE decap
    succeeds. Once we have a recipient context we always seal errors so the
    relay can't distinguish them from real failures."""
    return {"error": message}, status
