"""Unit tests for the /v1/ohttp handler in
``tee_gateway.controllers.ohttp_controller``.

The controller is a relatively thin shell that sits between HPKE
decapsulation and an in-process WSGI sub-request to /v1/chat/completions.
These tests pin down the shell's behaviour without bringing up the real
chat backend by stubbing ``get_tee_keys`` (to plant a known HPKE keypair)
and ``current_app.wsgi_app`` (to fake the inner /v1/chat/completions
response).

What we verify:
  * 413 on an oversized encapsulated body
  * 400 on a malformed encapsulation
  * 400 when the decrypted inner payload is not valid JSON
  * Non-2xx inner responses are forwarded as plaintext (no HPKE seal),
    with the inner content-type and the x402-related headers preserved
  * 2xx inner responses are sealed and the ``opengradient`` cost block is
    projected onto outer X-Inference-Cost-* headers
  * Streaming inner responses emit exactly one AAD=b"final" chunk and
    close the inner WSGI iterator (settles x402)
"""

from __future__ import annotations

import json
import struct
from decimal import Decimal
from typing import Iterator

import pytest
from flask import Flask

from tee_gateway import ohttp
from tee_gateway.controllers import ohttp_controller


# ---------------------------------------------------------------------------
# helpers


def _encapsulate(plaintext: bytes):
    """Build a real HPKE-encapsulated request and return ``(sk, wire, sender, enc)``.

    The sender context is returned so individual tests that need to decrypt
    the outer response can re-derive the response key the same way an SDK
    client would.
    """
    sk, pk_raw = ohttp.generate_keypair()
    hdr = bytes([ohttp.KEY_CONFIG_ID]) + struct.pack(
        ">HHH",
        ohttp.KEM_ID_X25519,
        ohttp.KDF_ID_HKDF_SHA256,
        ohttp.AEAD_ID_CHACHA20_POLY1305,
    )
    info = b"message/bhttp request" + b"\x00" + hdr
    pkr = ohttp._SUITE.kem.deserialize_public_key(pk_raw)
    enc, sender = ohttp._SUITE.create_sender_context(pkr, info=info)
    wire = hdr + enc + sender.seal(plaintext, aad=b"")
    return sk, wire, sender, enc


class _FakeTee:
    def __init__(self, sk):
        self.hpke_private_key = sk


class _CloseTrackingIter:
    """Iterable that records when ``close()`` is called.

    Mirrors how the real WSGI iterator from the chat handler behaves —
    closing it is what triggers x402's post-response settlement, so we
    need the controller to call it on both the streaming and non-streaming
    paths.
    """

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for c in self._chunks:
            yield c

    def close(self):
        self.closed = True


def _make_app(inner_responder, sk):
    """Build a tiny Flask app with the controller mounted at /v1/ohttp.

    ``inner_responder`` is invoked when the controller does its WSGI
    sub-dispatch to ``/v1/chat/completions``; it must return
    ``(status_line, headers, body_iter)``. Anything else passes through
    to the real Flask routing so the outer POST still lands on our
    handler.
    """
    app = Flask(__name__)
    app.add_url_rule(
        "/v1/ohttp",
        view_func=ohttp_controller.create_anonymous_chat_completion,
        methods=["POST"],
    )

    original_wsgi = app.wsgi_app
    captured = {"env": None, "called": 0, "iter": None}

    def fake_wsgi(env, start_response):
        if env.get("PATH_INFO") == "/v1/chat/completions":
            captured["env"] = env
            captured["called"] += 1
            status, headers, body_iter = inner_responder()
            captured["iter"] = body_iter
            start_response(status, headers)
            return body_iter
        return original_wsgi(env, start_response)

    app.wsgi_app = fake_wsgi

    def fake_get_tee_keys():
        return _FakeTee(sk)

    return app, captured, fake_get_tee_keys


# ---------------------------------------------------------------------------
# 413 / 400 cases — no decap happens, so we don't even need a key


def test_oversized_body_returns_413(monkeypatch):
    sk, _ = ohttp.generate_keypair()
    # Body is well past _MAX_ENCAPSULATED_REQUEST_BYTES (512 KiB). Werkzeug
    # will set Content-Length from the data length, so the up-front check
    # fires before any HPKE work.
    app, captured, fake_keys = _make_app(lambda: ("200 OK", [], iter([])), sk)
    monkeypatch.setattr(ohttp_controller, "get_tee_keys", fake_keys)

    too_big = b"x" * (ohttp_controller._MAX_ENCAPSULATED_REQUEST_BYTES + 1)
    client = app.test_client()
    resp = client.post("/v1/ohttp", data=too_big, content_type="message/ohttp-req")

    assert resp.status_code == 413
    assert captured["called"] == 0


def test_empty_body_returns_400(monkeypatch):
    sk, _ = ohttp.generate_keypair()
    app, captured, fake_keys = _make_app(lambda: ("200 OK", [], iter([])), sk)
    monkeypatch.setattr(ohttp_controller, "get_tee_keys", fake_keys)

    client = app.test_client()
    resp = client.post("/v1/ohttp", data=b"", content_type="message/ohttp-req")
    assert resp.status_code == 400
    assert captured["called"] == 0


def test_malformed_encapsulation_returns_400(monkeypatch):
    sk, _ = ohttp.generate_keypair()
    app, captured, fake_keys = _make_app(lambda: ("200 OK", [], iter([])), sk)
    monkeypatch.setattr(ohttp_controller, "get_tee_keys", fake_keys)

    # Random short payload — passes the size gate but ohttp.decapsulate_request
    # will reject it as malformed. The controller MUST normalise that into a
    # generic 400 so it doesn't expose an oracle on which decap step failed.
    client = app.test_client()
    resp = client.post(
        "/v1/ohttp", data=b"\x00" * 64, content_type="message/ohttp-req"
    )
    assert resp.status_code == 400
    assert captured["called"] == 0


def test_invalid_inner_json_returns_400(monkeypatch):
    """Plaintext decapsulates fine but isn't valid JSON — controller must
    surface that as a 400 rather than fall through to the chat handler."""
    sk, wire, _, _ = _encapsulate(b"not-json{{{")
    app, captured, fake_keys = _make_app(lambda: ("200 OK", [], iter([])), sk)
    monkeypatch.setattr(ohttp_controller, "get_tee_keys", fake_keys)

    client = app.test_client()
    resp = client.post("/v1/ohttp", data=wire, content_type="message/ohttp-req")
    assert resp.status_code == 400
    assert captured["called"] == 0


def test_non_object_inner_payload_returns_400(monkeypatch):
    sk, wire, _, _ = _encapsulate(b"[1, 2, 3]")
    app, captured, fake_keys = _make_app(lambda: ("200 OK", [], iter([])), sk)
    monkeypatch.setattr(ohttp_controller, "get_tee_keys", fake_keys)

    client = app.test_client()
    resp = client.post("/v1/ohttp", data=wire, content_type="message/ohttp-req")
    assert resp.status_code == 400
    assert captured["called"] == 0


# ---------------------------------------------------------------------------
# Non-2xx inner: plaintext passthrough, forwarded headers


def test_non_2xx_inner_response_is_plaintext_passthrough(monkeypatch):
    """An x402 402 (or any non-2xx) from the chat handler must NOT be
    HPKE-sealed — the relay needs to read the payment challenge and act
    on it. The inner content-type and x402 control headers must survive."""
    sk, wire, _, _ = _encapsulate(b'{"model":"gpt-4.1","messages":[]}')

    inner_body = b'{"error":"payment required"}'

    def inner():
        return (
            "402 Payment Required",
            [
                ("Content-Type", "application/json"),
                ("WWW-Authenticate", "x402 ..."),
                ("X-Payment-Required", "true"),
                ("X-Tee-Signature", "sig"),
                ("Set-Cookie", "tracker=abc"),  # must NOT be forwarded
            ],
            _CloseTrackingIter([inner_body]),
        )

    app, captured, fake_keys = _make_app(inner, sk)
    monkeypatch.setattr(ohttp_controller, "get_tee_keys", fake_keys)

    client = app.test_client()
    resp = client.post(
        "/v1/ohttp",
        data=wire,
        content_type="message/ohttp-req",
        headers={"X-Payment": "client-payment-blob"},
    )

    assert resp.status_code == 402
    # Body must be plaintext, NOT message/ohttp-res.
    assert resp.content_type.startswith("application/json")
    assert resp.data == inner_body
    # x402 / WWW-Authenticate forwarded; arbitrary headers not.
    assert resp.headers.get("WWW-Authenticate") == "x402 ..."
    assert resp.headers.get("X-Payment-Required") == "true"
    assert resp.headers.get("X-Tee-Signature") == "sig"
    assert "Set-Cookie" not in resp.headers
    # The relay's X-Payment was forwarded into the inner env so the
    # x402 middleware can verify it.
    assert captured["env"]["HTTP_X_PAYMENT"] == "client-payment-blob"
    # WSGI iterator was drained AND closed (drains x402 settlement).
    assert captured["iter"].closed is True


# ---------------------------------------------------------------------------
# 2xx inner: response sealed, cost headers surfaced


def _client_decrypt_response(sealed: bytes, sender, enc: bytes) -> bytes:
    """Recover the plaintext from a single-shot OHTTP response, the same
    way an external client SDK would."""
    from cryptography.hazmat.primitives import hashes, hmac
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

    response_secret = sender.export(b"message/bhttp response", 32)
    response_nonce = sealed[:32]
    aead_ct = sealed[32:]
    salt = enc + response_nonce
    h = hmac.HMAC(salt, hashes.SHA256())
    h.update(response_secret)
    prk = h.finalize()
    key = HKDFExpand(algorithm=hashes.SHA256(), length=32, info=b"key").derive(prk)
    nonce = HKDFExpand(algorithm=hashes.SHA256(), length=12, info=b"nonce").derive(prk)
    return ChaCha20Poly1305(key).decrypt(nonce, aead_ct, b"")


def test_2xx_inner_response_is_sealed_with_cost_headers(monkeypatch):
    sk, wire, sender, enc = _encapsulate(b'{"model":"gpt-4.1","messages":[]}')

    inner_payload = {
        "id": "chatcmpl-xyz",
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "opengradient": {
            "cost_opg": 1234567890,
            "cost_usd": "0.001234",
            "opg_price_usd": "1.234",
        },
    }
    inner_body = json.dumps(inner_payload).encode()

    def inner():
        return (
            "200 OK",
            [
                ("Content-Type", "application/json"),
                ("X-Tee-Signature", "sig"),
                ("X-Payment-Response", "settled"),
            ],
            _CloseTrackingIter([inner_body]),
        )

    app, captured, fake_keys = _make_app(inner, sk)
    monkeypatch.setattr(ohttp_controller, "get_tee_keys", fake_keys)

    client = app.test_client()
    resp = client.post("/v1/ohttp", data=wire, content_type="message/ohttp-req")

    assert resp.status_code == 200
    assert resp.content_type.startswith(ohttp_controller.OHTTP_RESPONSE_MEDIA_TYPE)
    # Cost is projected onto outer headers — relay-billable, no model name
    # and no token counts leak.
    assert resp.headers["X-Inference-Cost-OPG"] == "1234567890"
    assert Decimal(resp.headers["X-Inference-Cost-USD"]) == Decimal("0.001234")
    assert Decimal(resp.headers["X-Inference-Price-OPG-USD"]) == Decimal("1.234")
    # x402 / TEE headers still pass through.
    assert resp.headers.get("X-Tee-Signature") == "sig"
    assert resp.headers.get("X-Payment-Response") == "settled"

    decrypted = _client_decrypt_response(resp.data, sender, enc)
    assert json.loads(decrypted) == inner_payload
    assert captured["iter"].closed is True


def test_2xx_inner_without_cost_block_omits_cost_headers(monkeypatch):
    """Missing or unparseable opengradient block must not 500 — we just
    skip the cost projection. Belt-and-braces guard around _extract_cost_headers."""
    sk, wire, sender, enc = _encapsulate(b'{"model":"gpt-4.1","messages":[]}')

    inner_body = b'{"id":"chatcmpl-xyz","choices":[]}'

    def inner():
        return (
            "200 OK",
            [("Content-Type", "application/json")],
            _CloseTrackingIter([inner_body]),
        )

    app, _, fake_keys = _make_app(inner, sk)
    monkeypatch.setattr(ohttp_controller, "get_tee_keys", fake_keys)

    client = app.test_client()
    resp = client.post("/v1/ohttp", data=wire, content_type="message/ohttp-req")

    assert resp.status_code == 200
    assert "X-Inference-Cost-OPG" not in resp.headers
    assert "X-Inference-Cost-USD" not in resp.headers


# ---------------------------------------------------------------------------
# Streaming path


def test_streaming_inner_response_emits_one_final_chunk_and_closes(monkeypatch):
    sk, wire, sender, enc = _encapsulate(b'{"model":"gpt-4.1","stream":true}')

    sse_chunks = [
        b"data: {\"a\":1}\n\n",
        b"data: {\"b\":2}\n\n",
        b"data: [DONE]\n\n",
    ]

    def inner():
        return (
            "200 OK",
            [("Content-Type", "text/event-stream")],
            _CloseTrackingIter(sse_chunks),
        )

    app, captured, fake_keys = _make_app(inner, sk)
    monkeypatch.setattr(ohttp_controller, "get_tee_keys", fake_keys)

    client = app.test_client()
    resp = client.post("/v1/ohttp", data=wire, content_type="message/ohttp-req")

    assert resp.status_code == 200
    assert resp.content_type.startswith(
        ohttp_controller.OHTTP_CHUNKED_RESPONSE_MEDIA_TYPE
    )

    # Decode the chunked-OHTTP wire frame the same way a client SDK would.
    body = resp.get_data()
    response_nonce = body[:32]
    off = 32

    response_secret = sender.export(b"message/bhttp chunked response", 32)
    aead_key, aead_nonce = ohttp._derive_response_keys(
        response_secret, enc, response_nonce
    )
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    aead = ChaCha20Poly1305(aead_key)

    recovered = []
    final_count = 0
    counter = 0
    while off < len(body):
        length, off = ohttp.decode_varint(body, off)
        is_final = length == 0
        seg_len = len(body) - off if is_final else length
        ct = body[off : off + seg_len]
        off += seg_len
        chunk_nonce = bytes(
            a ^ b for a, b in zip(aead_nonce, counter.to_bytes(12, "big"))
        )
        aad = b"final" if is_final else b""
        recovered.append(aead.decrypt(chunk_nonce, ct, aad))
        counter += 1
        if is_final:
            final_count += 1
            break

    # Every SSE event must be recovered, and the last chunk MUST be the
    # AAD=b"final" terminator — that's what protects clients from
    # undetected truncation. There must be exactly one.
    assert recovered == sse_chunks
    assert final_count == 1
    # And the inner iterator must have been closed so x402 streaming
    # settlement runs.
    assert captured["iter"].closed is True


def test_streaming_path_uses_chunked_when_only_one_inner_chunk(monkeypatch):
    """Even with a single SSE event, the controller must still emit
    exactly one AAD=b"final" chunk (the pending/lookahead logic in
    _build_streaming_response is the part being pinned down)."""
    sk, wire, sender, enc = _encapsulate(b'{"model":"gpt-4.1","stream":true}')

    only = b"data: [DONE]\n\n"

    def inner():
        return (
            "200 OK",
            [("Content-Type", "text/event-stream")],
            _CloseTrackingIter([only]),
        )

    app, captured, fake_keys = _make_app(inner, sk)
    monkeypatch.setattr(ohttp_controller, "get_tee_keys", fake_keys)

    client = app.test_client()
    resp = client.post("/v1/ohttp", data=wire, content_type="message/ohttp-req")

    body = resp.get_data()
    response_nonce = body[:32]
    off = 32
    response_secret = sender.export(b"message/bhttp chunked response", 32)
    aead_key, aead_nonce = ohttp._derive_response_keys(
        response_secret, enc, response_nonce
    )
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    aead = ChaCha20Poly1305(aead_key)

    length, off = ohttp.decode_varint(body, off)
    # With exactly one inner chunk, the controller buffers it and emits
    # it as the final marker — so the first (and only) framed chunk is
    # the zero-length-prefixed final.
    assert length == 0
    ct = body[off:]
    chunk_nonce = bytes(a ^ b for a, b in zip(aead_nonce, (0).to_bytes(12, "big")))
    assert aead.decrypt(chunk_nonce, ct, b"final") == only
    assert captured["iter"].closed is True


# ---------------------------------------------------------------------------
# 503 when HPKE key isn't initialized


def test_returns_503_when_hpke_key_missing(monkeypatch):
    app, captured, _ = _make_app(lambda: ("200 OK", [], iter([])), sk=None)

    class _Tee:
        hpke_private_key = None

    monkeypatch.setattr(ohttp_controller, "get_tee_keys", lambda: _Tee())

    client = app.test_client()
    resp = client.post("/v1/ohttp", data=b"abc", content_type="message/ohttp-req")
    assert resp.status_code == 503
    assert captured["called"] == 0


# ---------------------------------------------------------------------------
# Identifying field scrubbing


def test_identifying_fields_are_scrubbed_before_inner_dispatch(monkeypatch):
    payload = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "hi"}],
        "user": "alice@example.com",
        "metadata": {"req": "123"},
        "x-request-id": "abc",
        "request_id": "def",
    }
    sk, wire, _, _ = _encapsulate(json.dumps(payload).encode())

    inner_body = b'{"id":"chatcmpl-1","choices":[]}'

    def inner():
        return (
            "200 OK",
            [("Content-Type", "application/json")],
            _CloseTrackingIter([inner_body]),
        )

    app, captured, fake_keys = _make_app(inner, sk)
    monkeypatch.setattr(ohttp_controller, "get_tee_keys", fake_keys)

    client = app.test_client()
    resp = client.post("/v1/ohttp", data=wire, content_type="message/ohttp-req")
    assert resp.status_code == 200

    # Reconstruct what the inner handler received.
    inner_env = captured["env"]
    inner_body_received = inner_env["wsgi.input"].read(
        int(inner_env["CONTENT_LENGTH"])
    )
    forwarded = json.loads(inner_body_received)
    assert forwarded == {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "hi"}],
    }
    # Authorization is overwritten with the OHTTP-fixed value so it can't
    # re-identify the client.
    assert inner_env["HTTP_AUTHORIZATION"] == "Bearer ohttp"
