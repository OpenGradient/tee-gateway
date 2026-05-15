"""
Oblivious HTTP endpoint for anonymous inference (relay-pays model).

This handler is a thin shell: it HPKE-decapsulates the inner request, re-issues
it as an in-process WSGI sub-request against the enclave's own
``/v1/chat/completions``, then encapsulates the response. All x402 payment,
LangChain routing, cost settlement and TEE response signing reuse the public
chat code paths — there is no duplicated routing or pricing logic here.

Trust / payment model:
  * The CLIENT encrypts only an LLM chat-completion request. It does not see,
    sign, or carry x402 payment material.
  * A RELAY sits between the client and the enclave. The relay holds the
    x402 wallet and forwards the (still-encrypted) inner request to the
    enclave, attaching its own ``x-payment`` header on the OUTER HTTP
    request.
  * The ENCLAVE decrypts the inner payload, forwards the request to its own
    chat endpoint with the relay's ``x-payment`` header, and returns. The
    relay sees status, settlement headers, and token-usage headers, but
    never sees the inner prompt or completion.

Wire format — outer HTTP request to /v1/ohttp:
    Headers:
        x-payment: <base64 x402 payload, supplied by the relay>
        content-type: message/ohttp-req
    Body: HPKE-encapsulated chat-completion JSON (just the inner body,
          no JSON envelope — the inner payload IS the chat request).

Wire format — outer HTTP response from /v1/ohttp:
    On 2xx (inference succeeded):
        Headers:
            content-type: message/ohttp-res
            x-payment-response, x-upto-session, ...   (forwarded from x402)
            x-usage-prompt-tokens, x-usage-completion-tokens,
            x-usage-total-tokens, x-usage-model       (so the relay can bill)
        Body: HPKE-encapsulated chat-completion response JSON. The relay
              cannot decrypt; only the client (who has the HPKE response
              key from its sender context) can read prompts/completions.
    On non-2xx (402 payment required, validation errors, etc.):
        Body forwarded as plaintext so the relay can act on it (read x402
        payment requirements, retry with a larger payment, surface errors
        to the user). These bodies never contain user prompts/completions.

Privacy properties:
  * Relay sees ciphertext + relay-side wallet + token usage + relay's IP.
    Never sees prompts, completions, or the client's IP.
  * Enclave sees plaintext prompts/completions + relay's IP. Never sees
    the client's IP.
  * Unlinkability holds unless the relay and the enclave collude.
  * Streaming is rejected; SSE re-introduces per-chunk timing/length side
    channels that would defeat sealing.
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
# back through the relay to the client.
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
        chat_body = json.loads(decap.plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _error(400, "inner payload is not valid JSON")

    if not isinstance(chat_body, dict):
        return _error(400, "inner payload must be a JSON object")

    if chat_body.get("stream"):
        return _error(400, "stream=true is not supported over OHTTP")

    chat_body = _scrub(chat_body)
    body_bytes = json.dumps(chat_body, separators=(",", ":")).encode("utf-8")

    # The relay pays — x-payment is a standard outer-request header, not
    # inside the encrypted envelope. Pass it through to the inner endpoint
    # so x402 verifies and settles exactly as it does for a normal call.
    payment_header = flask_request.headers.get("X-Payment")

    sub_status, sub_headers, sub_body = _wsgi_subrequest(
        path="/v1/chat/completions",
        body_bytes=body_bytes,
        payment_header=payment_header,
    )

    return _build_outer_response(decap, sub_status, sub_headers, sub_body)


def _build_outer_response(
    decap: ohttp.DecapsulatedRequest,
    status: int,
    headers: list[tuple[str, str]],
    body_bytes: bytes,
) -> Response:
    """Translate the inner sub-response into the outer OHTTP response.

    On 2xx we seal the body (which contains user prompts/completions) and
    surface token-usage headers so the relay can bill. On non-2xx we pass
    through the inner body verbatim — those responses carry x402 payment
    requirements or error messages that the relay needs to read, and never
    contain user content.
    """
    forwarded = {name: value for name, value in headers if _should_forward_header(name)}
    inner_content_type = next(
        (v for k, v in headers if k.lower() == "content-type"),
        "application/json",
    )

    if not (200 <= status < 300):
        return Response(
            body_bytes,
            status=status,
            headers=forwarded,
            content_type=inner_content_type,
        )

    forwarded.update(_extract_usage_headers(body_bytes))
    sealed = ohttp.encapsulate_response(decap.response_key, decap.enc, body_bytes)
    return Response(
        sealed,
        status=status,
        headers=forwarded,
        mimetype=OHTTP_RESPONSE_MEDIA_TYPE,
    )


def _extract_usage_headers(body_bytes: bytes) -> dict[str, str]:
    """Pull token-usage + model name out of a chat-completion response and
    project them onto outer HTTP headers for the relay's billing pipeline.
    These are the ONLY pieces of metadata the relay needs to charge; the
    prompt and completion themselves stay sealed."""
    try:
        body = json.loads(body_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(body, dict):
        return {}

    headers: dict[str, str] = {}
    usage = body.get("usage")
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        total_tokens = usage.get("total_tokens")
        if prompt_tokens is not None:
            headers["X-Usage-Prompt-Tokens"] = str(prompt_tokens)
        if completion_tokens is not None:
            headers["X-Usage-Completion-Tokens"] = str(completion_tokens)
        if total_tokens is not None:
            headers["X-Usage-Total-Tokens"] = str(total_tokens)

    model = body.get("model")
    if isinstance(model, str):
        headers["X-Usage-Model"] = model
    return headers


def _wsgi_subrequest(
    path: str,
    body_bytes: bytes,
    payment_header: str | None,
) -> tuple[int, list[tuple[str, str]], bytes]:
    """Issue an in-process WSGI request through the app's full middleware stack.

    Returns ``(status_code, headers, body_bytes)``. We invoke
    ``current_app.wsgi_app`` directly so the x402 payment middleware (which
    wraps ``wsgi_app`` at injection time) runs the same way it would for an
    external HTTP request to the same path — including the pre-inference
    pricing gate, payment verification, cost settlement and TEE response
    signing.
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
    return status_code, captured["headers"], b"".join(body_chunks)


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
        return {"error": "Failed to retrieve HPKE config"}, 500


def _error(status: int, message: str) -> tuple[dict, int]:
    """Plaintext error for cases where we never have a recipient context
    (empty body, malformed encapsulation). Post-decap input errors are
    also returned plaintext so the relay can surface them to the client —
    they never contain user prompts."""
    return {"error": message}, status
