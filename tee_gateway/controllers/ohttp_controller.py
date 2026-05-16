"""
Oblivious HTTP endpoint for anonymous inference (relay-pays model).

This handler is a thin shell: it HPKE-decapsulates the inner request, re-issues
it as an in-process WSGI sub-request against the enclave's own
``/v1/chat/completions``, then encapsulates the response. All x402 payment,
LangChain routing, cost settlement and TEE response signing reuse the public
chat code paths — there is no duplicated routing or pricing logic here.

Two response modes are supported, dispatched by the inner ``stream`` flag:
  * stream=false → single-shot OHTTP response (RFC 9458 §4.5),
    content-type ``message/ohttp-res``. Usage stats surface in outer headers.
  * stream=true  → chunked OHTTP response (draft-ietf-ohai-chunked-ohttp-08),
    content-type ``message/ohttp-chunked-res``. Each SSE event from the
    inner /v1/chat/completions stream becomes one sealed OHTTP chunk; the
    final chunk uses AAD=b"final" so truncation is detectable. Usage stats
    can't appear in outer headers (sent before body) — clients read them
    from the final SSE event inside the decrypted stream; the relay relies
    on x402 settlement metadata (X-Upto-Session) for billing.

Trust / payment model:
  * The CLIENT encrypts only an LLM chat-completion request. It does not see,
    sign, or carry x402 payment material.
  * A RELAY sits between the client and the enclave. The relay holds the
    x402 wallet and forwards the (still-encrypted) inner request to the
    enclave, attaching its own ``x-payment`` header on the OUTER HTTP
    request.
  * The ENCLAVE decrypts the inner payload, forwards the request to its own
    chat endpoint with the relay's ``x-payment`` header, and returns. The
    relay sees status, settlement headers, and (for non-stream) token-usage
    headers, but never sees the inner prompt or completion.

Privacy properties:
  * Relay (network position): terminates the client's TCP/TLS connection,
    so the relay DOES see the client's IP at the network layer — that's
    unavoidable. What the relay does NOT see is the request/response
    content: it observes only the OHTTP-encapsulated ciphertext, its own
    wallet's x-payment material, and (single-shot only) the token-usage
    outer headers it needs to bill.
  * Enclave (compute position): sees the plaintext prompt and completion
    (they are decrypted inside the enclave to run the LLM call), but at
    the network layer it only sees the RELAY's IP — never the client's.
    That's the unlinkability property: the enclave cannot tie a request's
    plaintext back to a specific end user.
  * Unlinkability between a specific client identity and a specific
    plaintext request holds unless the relay and the enclave collude
    (the relay would have to disclose its client-IP log alongside the
    enclave's per-request plaintext log).
  * Streaming leaks per-chunk timing and length (the relay sees the
    cadence of varint-framed sealed chunks). This is an inherent cost of
    server-sent events — clients who can't accept that signal should
    use stream=false.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Iterator

from flask import Response, current_app, request as flask_request

from tee_gateway import ohttp
from tee_gateway.tee_manager import get_tee_keys

logger = logging.getLogger(__name__)

OHTTP_MEDIA_TYPE = "message/ohttp-req"
OHTTP_RESPONSE_MEDIA_TYPE = "message/ohttp-res"
OHTTP_CHUNKED_RESPONSE_MEDIA_TYPE = "message/ohttp-chunked-res"
_SSE_CONTENT_TYPE = "text/event-stream"

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

    chat_body = _scrub(chat_body)
    body_bytes = json.dumps(chat_body, separators=(",", ":")).encode("utf-8")

    # The relay pays — x-payment is a standard outer-request header, not
    # inside the encrypted envelope. Pass it through to the inner endpoint
    # so x402 verifies and settles exactly as it does for a normal call.
    payment_header = flask_request.headers.get("X-Payment")

    sub_status, sub_headers, sub_iter = _wsgi_subrequest(
        path="/v1/chat/completions",
        body_bytes=body_bytes,
        payment_header=payment_header,
    )

    inner_content_type = next(
        (v for k, v in sub_headers if k.lower() == "content-type"),
        "application/json",
    )
    is_streaming = (
        200 <= sub_status < 300
        and inner_content_type.split(";", 1)[0].strip().lower() == _SSE_CONTENT_TYPE
    )

    if is_streaming:
        return _build_streaming_response(decap, sub_status, sub_headers, sub_iter)

    # Non-streaming: drain into bytes (this also triggers x402's
    # post-response settlement via the WSGI iterator's close()).
    body_bytes_out = _drain(sub_iter)
    return _build_outer_response(
        decap, sub_status, sub_headers, body_bytes_out, inner_content_type
    )


def _build_outer_response(
    decap: ohttp.DecapsulatedRequest,
    status: int,
    headers: list[tuple[str, str]],
    body_bytes: bytes,
    inner_content_type: str,
) -> Response:
    """Single-shot OHTTP response. Seals the body on 2xx (contains user
    prompts/completions) and surfaces token usage as outer headers so the
    relay can bill. Non-2xx bodies (x402 payment requirements, validation
    errors) are forwarded as plaintext so the relay can act on them."""
    forwarded = {name: value for name, value in headers if _should_forward_header(name)}

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


def _build_streaming_response(
    decap: ohttp.DecapsulatedRequest,
    status: int,
    headers: list[tuple[str, str]],
    sub_iter: Iterator[bytes],
) -> Response:
    """Chunked OHTTP response (draft-ietf-ohai-chunked-ohttp-08).

    Each SSE event from the inner /v1/chat/completions stream is sealed as
    one OHTTP chunk and yielded immediately. The final chunk uses
    AAD=b"final" with a zero-length varint prefix; emitting it requires
    look-ahead by one chunk so we know which one is last, hence the
    ``pending`` buffer below.

    Usage stats can't be exposed as outer headers (those are already sent
    before the body); the relay bills via x402 settlement metadata
    (X-Upto-Session header, set up-front). The client reads usage from the
    final SSE event inside the decrypted stream.
    """
    forwarded = {name: value for name, value in headers if _should_forward_header(name)}

    def _stream() -> Iterator[bytes]:
        encrypter = ohttp.ChunkedResponseEncrypter(
            decap.response_key_chunked, decap.enc
        )
        yield encrypter.header()

        pending: bytes | None = None
        try:
            for chunk in sub_iter:
                if not chunk:
                    continue
                if pending is not None:
                    yield encrypter.encrypt_chunk(pending, is_final=False)
                pending = chunk
            # Always emit exactly one final chunk so the AAD=b"final"
            # marker is present — that's what protects clients from
            # undetected truncation.
            yield encrypter.encrypt_chunk(pending or b"", is_final=True)
        finally:
            close = getattr(sub_iter, "close", None)
            if callable(close):
                # Triggers x402's streaming-session settlement.
                close()

    return Response(
        _stream(),
        status=status,
        headers=forwarded,
        mimetype=OHTTP_CHUNKED_RESPONSE_MEDIA_TYPE,
    )


def _drain(sub_iter: Iterator[bytes]) -> bytes:
    chunks: list[bytes] = []
    try:
        for chunk in sub_iter:
            if chunk:
                chunks.append(chunk)
    finally:
        close = getattr(sub_iter, "close", None)
        if callable(close):
            close()
    return b"".join(chunks)


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
) -> tuple[int, list[tuple[str, str]], Iterator[bytes]]:
    """Issue an in-process WSGI request through the app's full middleware stack.

    Returns ``(status_code, headers, body_iterator)``. The caller is
    responsible for draining and closing the iterator (close() triggers
    x402's post-response settlement). We invoke ``current_app.wsgi_app``
    directly so the x402 payment middleware (which wraps ``wsgi_app`` at
    injection time) runs the same way it would for an external HTTP
    request to the same path — including the pre-inference pricing gate,
    payment verification, cost settlement and TEE response signing.
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

    # The OpenAPI spec declares a global ApiKeyAuth requirement and connexion
    # enforces it before our handler runs (returns 401 "No authorization
    # token provided"). The security function (security_controller.py) is an
    # intentional passthrough — x402 is the real access control — so any
    # value satisfies the schema check. We deliberately do NOT forward the
    # outer Authorization header: anything the relay attached there could
    # re-identify the client (API keys, JWT subjects, bearer tokens) and
    # defeat the whole point of OHTTP. A fixed constant keeps every OHTTP
    # request indistinguishable to the chat backend at this layer.
    sub_env["HTTP_AUTHORIZATION"] = "Bearer ohttp"

    captured: dict[str, Any] = {"status": "500 Internal Server Error", "headers": []}

    def _start_response(status: str, headers: list, exc_info: Any = None):
        captured["status"] = status
        captured["headers"] = headers
        return lambda _chunk: None

    iterator = current_app.wsgi_app(sub_env, _start_response)
    status_code = int(captured["status"].split(" ", 1)[0])
    # Don't wrap in iter() — that would strip the iterable's close() method,
    # which the caller relies on to trigger x402's post-response settlement.
    return status_code, captured["headers"], iterator  # type: ignore[return-value]


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
