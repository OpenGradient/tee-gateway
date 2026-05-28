"""
Oblivious HTTP endpoint for anonymous inference (relay-pays model).

This handler is a thin shell: it HPKE-decapsulates the inner request, re-issues
it as an in-process WSGI sub-request against the enclave's own
``/v1/chat/completions``, then encapsulates the response. All x402 payment,
LangChain routing, cost settlement and TEE response signing reuse the public
chat code paths — there is no duplicated routing or pricing logic here.

Two response modes are supported, dispatched by the inner ``stream`` flag:
  * stream=false → single-shot OHTTP response (RFC 9458 §4.5),
    content-type ``message/ohttp-res``.
  * stream=true  → chunked OHTTP response (draft-ietf-ohai-chunked-ohttp-08),
    content-type ``message/ohttp-chunked-res``. Each SSE event from the
    inner /v1/chat/completions stream becomes one sealed OHTTP chunk; the
    final chunk uses AAD=b"final" so truncation is detectable.

Billing channel for the relay (both modes settle the real cost via x402
against the relay's x-payment; the gateway is the source of truth for the
amount):
  * stream=false: outer headers expose ONLY the settled cost —
    X-Inference-Cost-OPG (smallest units, the integer x402 actually charged)
    and X-Inference-Cost-USD (the equivalent USD at the price used). No
    model name, no token counts: those would be a fingerprint of the
    inner request and have no role in billing.
  * stream=true: outer response headers are flushed before any body chunk
    is yielded, so cost isn't known at header-write time. The relay reads
    the settled amount from x402: either by querying the facilitator with
    its X-Upto-Session, or via X-Payment-Response on its next request. The
    client still sees cost in the final SSE event inside the encrypted
    stream (the ``opengradient`` block written by the chat controller).

Trust / payment model:
  * The CLIENT encrypts only an LLM chat-completion request. It does not see,
    sign, or carry x402 payment material.
  * A RELAY sits between the client and the enclave. The relay holds the
    x402 wallet and forwards the (still-encrypted) inner request to the
    enclave, attaching its own ``x-payment`` header on the OUTER HTTP
    request.
  * The ENCLAVE decrypts the inner payload, forwards the request to its own
    chat endpoint with the relay's ``x-payment`` header, and returns. The
    relay sees status and settlement headers and, for non-stream, the outer
    ``X-Inference-Cost-OPG`` / ``X-Inference-Cost-USD`` billing headers, but
    never sees the inner prompt or completion.

Privacy properties:
  * Relay (network position): terminates the client's TCP/TLS connection,
    so the relay DOES see the client's IP at the network layer — that's
    unavoidable. What the relay does NOT see is the request/response
    content: it observes only the OHTTP-encapsulated ciphertext, its own
    wallet's x-payment material, and (single-shot only) the settled-cost
    outer headers it needs to bill its own customer. Model name and token
    counts are deliberately NOT surfaced — they'd fingerprint the inner
    request without adding anything billing-relevant.
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
from tee_gateway.pricing import SessionCost

logger = logging.getLogger(__name__)

OHTTP_RESPONSE_MEDIA_TYPE = "message/ohttp-res"
OHTTP_CHUNKED_RESPONSE_MEDIA_TYPE = "message/ohttp-chunked-res"
OHTTP_BILLING_FRAME_MAGIC = b"\n--opengradient-ohttp-billing-v1--\n"
_SSE_CONTENT_TYPE = "text/event-stream"

# Cap on the encapsulated request size. The inner payload is a chat-completion
# JSON body; even with long conversation history this comfortably fits in a few
# hundred KB. Rejecting larger bodies up-front prevents a malicious relay from
# forcing the enclave to allocate and attempt HPKE decapsulation on arbitrarily
# large blobs.
_MAX_ENCAPSULATED_REQUEST_BYTES = 512 * 1024

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
    declared_length = flask_request.content_length
    if (
        declared_length is not None
        and declared_length > _MAX_ENCAPSULATED_REQUEST_BYTES
    ):
        return _error(413, "encapsulated request too large")

    raw_body: bytes = flask_request.get_data(cache=False, parse_form_data=False)
    if not raw_body:
        return _error(400, "empty body")
    # Re-check after read in case Content-Length was absent or lied about the
    # actual stream length (chunked transfer, malicious client).
    if len(raw_body) > _MAX_ENCAPSULATED_REQUEST_BYTES:
        return _error(413, "encapsulated request too large")

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
    _set_inner_cost_context(flask_request, request_json=chat_body)
    body_bytes = json.dumps(chat_body, separators=(",", ":")).encode("utf-8")

    sub_status, sub_headers, sub_iter = _wsgi_subrequest(
        path="/v1/chat/completions",
        body_bytes=body_bytes,
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
        cost_context = flask_request.environ.get("x402.cost_context")
        return _build_streaming_response(
            decap,
            sub_status,
            sub_headers,
            sub_iter,
            cost_context if isinstance(cost_context, dict) else None,
        )

    # Non-streaming: drain into bytes, record the inner plaintext cost block
    # for outer /v1/ohttp x402 settlement, then seal the response.
    body_bytes_out = _drain(sub_iter)
    _set_inner_response_cost_context(
        flask_request, body_bytes_out, status_code=sub_status
    )
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
    prompts/completions) and surfaces only extracted cost/price headers as
    outer headers for relay billing (including `X-Inference-Price-OPG-USD`);
    usage details remain sealed in the encapsulated body. Non-2xx bodies
    (x402 payment requirements, validation errors) are forwarded as
    plaintext so the relay can act on them."""
    # Keep headers as a list of (name, value) tuples — WSGI gives us a list
    # specifically because HTTP allows multi-valued headers (RFC 7230 §3.2.2;
    # WWW-Authenticate in particular per RFC 7235 §4.1 can repeat, one per
    # challenge scheme). Collapsing to a dict would drop duplicates silently
    # and could lose an x402 challenge if future versions emit more than one.
    forwarded: list[tuple[str, str]] = [
        (name, value) for name, value in headers if _should_forward_header(name)
    ]

    if not (200 <= status < 300):
        return Response(
            body_bytes,
            status=status,
            headers=forwarded,
            content_type=inner_content_type,
        )

    # Cost headers come from a JSON parse — single-valued by construction —
    # so a dict is fine internally; just project to tuples at merge time.
    forwarded.extend(_extract_cost_headers(body_bytes).items())
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
    cost_context: dict[str, Any] | None,
) -> Response:
    """Chunked OHTTP response (draft-ietf-ohai-chunked-ohttp-08).

    Each SSE event from the inner /v1/chat/completions stream is sealed as
    one OHTTP chunk and yielded immediately. The final chunk uses
    AAD=b"final" with a zero-length varint prefix; emitting it requires
    look-ahead by one chunk so we know which one is last, hence the
    ``pending`` buffer below.

    Cost can't be exposed as outer headers (those are already flushed
    before the body), so the gateway emits a private plaintext billing
    frame for the relay to strip and process before forwarding bytes to
    the browser.
    """
    # See _build_outer_response: keep as a list so duplicate HTTP header
    # values (e.g. multiple WWW-Authenticate challenges) survive forwarding.
    forwarded: list[tuple[str, str]] = [
        (name, value) for name, value in headers if _should_forward_header(name)
    ]

    def _stream() -> Iterator[bytes]:
        encrypter = ohttp.ChunkedResponseEncrypter(
            decap.response_key_chunked, decap.enc
        )
        yield encrypter.header()

        pending: bytes | None = None
        plaintext_chunks: list[bytes] = []
        try:
            for chunk in sub_iter:
                if not chunk:
                    continue
                plaintext_chunks.append(chunk)
                if pending is not None:
                    yield encrypter.encrypt_chunk(pending, is_final=False)
                pending = chunk
            response_json = _parse_final_sse_json(b"".join(plaintext_chunks))
            _set_inner_cost_context(
                cost_context,
                response_json=response_json,
                status_code=status,
            )
            billing_frame = _build_billing_frame(response_json)
            if billing_frame:
                yield billing_frame

            # Always emit exactly one final chunk so the AAD=b"final"
            # marker is present — that's what protects clients from
            # undetected truncation.
            yield encrypter.encrypt_chunk(pending or b"", is_final=True)
        finally:
            _set_inner_stream_cost_context(
                cost_context, plaintext_chunks, status_code=status
            )
            close = getattr(sub_iter, "close", None)
            if callable(close):
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


def _extract_cost_headers(body_bytes: bytes) -> dict[str, str]:
    """Project the response's ``opengradient`` cost block onto outer headers
    for the relay's billing pipeline. Cost is the only metadata the relay
    needs — model name and token counts stay sealed (they'd fingerprint the
    inner request without being billing-relevant). The price is surfaced too
    so the relay (and downstream auditors) can verify
    ``cost_usd == cost_opg / 10^decimals * opg_price_usd`` without trusting us.
    """
    try:
        body = json.loads(body_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(body, dict):
        return {}
    try:
        cost = SessionCost.model_validate(body.get("opengradient"))
    except Exception:
        return {}
    wire = cost.model_dump(mode="json")
    return {
        "X-Inference-Cost-OPG": wire["cost_opg"],
        "X-Inference-Cost-USD": wire["cost_usd"],
        "X-Inference-Price-OPG-USD": wire["opg_price_usd"],
    }


def _build_billing_frame(response_json: dict[str, Any] | None) -> bytes:
    """Build the private gateway-to-relay billing frame for streaming OHTTP.

    This plaintext frame carries only the same billing fields projected as
    outer headers for non-streaming OHTTP. The relay strips it before forwarding
    bytes to the browser.
    """
    if not isinstance(response_json, dict):
        return b""
    try:
        cost = SessionCost.model_validate(response_json.get("opengradient"))
    except Exception:
        return b""
    payload = json.dumps(
        cost.model_dump(mode="json"),
        separators=(",", ":"),
    ).encode("utf-8")
    return OHTTP_BILLING_FRAME_MAGIC + len(payload).to_bytes(4, "big") + payload


def _wsgi_subrequest(
    path: str,
    body_bytes: bytes,
) -> tuple[int, list[tuple[str, str]], Iterator[bytes]]:
    """Issue an in-process WSGI request through the app's full middleware stack.

    Returns ``(status_code, headers, body_iterator)``. The caller is
    responsible for draining and closing the iterator. The outer /v1/ohttp
    request is the x402-paid boundary, so this inner chat dispatch uses the
    pre-x402 WSGI app saved at middleware installation time. That avoids
    charging/verifying the same relay payment twice while still running
    connexion routing, validation, TEE signing, and provider inference.
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

    inner_wsgi = current_app.config.get("OHTTP_INNER_WSGI_APP") or current_app.wsgi_app
    iterator = inner_wsgi(sub_env, _start_response)
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


def _set_inner_cost_context(
    req_or_context,
    *,
    request_json: dict[str, Any] | None = None,
    response_json: dict[str, Any] | None = None,
    status_code: int | None = None,
) -> None:
    if isinstance(req_or_context, dict):
        cost_context = req_or_context
    elif req_or_context is None:
        return
    else:
        cost_context = req_or_context.environ.get("x402.cost_context")
    if not isinstance(cost_context, dict):
        return
    if request_json is not None:
        cost_context["inner_request_json"] = request_json
    if response_json is not None:
        cost_context["inner_response_json"] = response_json
    if status_code is not None:
        cost_context["inner_status_code"] = status_code


def _set_inner_response_cost_context(
    req,
    body_bytes: bytes,
    *,
    status_code: int,
) -> None:
    try:
        response_json = json.loads(body_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        response_json = None
    _set_inner_cost_context(
        req,
        response_json=response_json if isinstance(response_json, dict) else None,
        status_code=status_code,
    )


def _set_inner_stream_cost_context(
    req,
    chunks: list[bytes],
    *,
    status_code: int,
) -> None:
    body = b"".join(chunks)
    response_json = _parse_final_sse_json(body)
    _set_inner_cost_context(
        req,
        response_json=response_json,
        status_code=status_code,
    )


def _parse_final_sse_json(body: bytes) -> dict[str, Any] | None:
    last_json: dict[str, Any] | None = None
    for line in body.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            last_json = parsed
    return last_json


def _sealed_error(req, decap: ohttp.DecapsulatedRequest, status: int, message: str):
    body = {"error": message}
    _set_inner_cost_context(req, response_json=body, status_code=status)
    return _sealed_json_response(decap, status, body)


def _sealed_json_response(
    decap: ohttp.DecapsulatedRequest,
    status: int,
    body_obj: Any,
) -> Response:
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
