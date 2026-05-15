"""Tests for the OHTTP encapsulation module."""

from __future__ import annotations

import json

import pytest

from tee_gateway import ohttp


def test_round_trip_request_and_response():
    sk, pk_raw = ohttp.generate_keypair()

    plaintext = json.dumps({"model": "gpt-4.1", "n": 1}).encode()
    # Encapsulate using the same code paths a client would, since pyhpke is
    # symmetric — we wire the request manually.
    config = ohttp.key_config(pk_raw)
    assert config[0] == ohttp.KEY_CONFIG_ID

    # Build a wire payload exactly as the SDK does.
    import struct

    hdr = bytes([ohttp.KEY_CONFIG_ID]) + struct.pack(
        ">HHH",
        ohttp.KEM_ID_X25519,
        ohttp.KDF_ID_HKDF_SHA256,
        ohttp.AEAD_ID_CHACHA20_POLY1305,
    )
    info = b"message/bhttp request" + b"\x00" + hdr
    pkr = ohttp._SUITE.kem.deserialize_public_key(pk_raw)
    enc, sender = ohttp._SUITE.create_sender_context(pkr, info=info)
    ct = sender.seal(plaintext, aad=b"")
    wire = hdr + enc + ct

    decap = ohttp.decapsulate_request(sk, wire)
    assert decap.plaintext == plaintext

    response_secret = sender.export(b"message/bhttp response", 32)
    assert decap.response_key == response_secret
    assert decap.enc == enc

    response = b'{"ok":true}'
    sealed = ohttp.encapsulate_response(decap.response_key, decap.enc, response)

    from cryptography.hazmat.primitives import hashes, hmac
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

    response_nonce = sealed[:32]
    aead_ct = sealed[32:]
    salt = enc + response_nonce
    h = hmac.HMAC(salt, hashes.SHA256())
    h.update(response_secret)
    prk = h.finalize()
    key = HKDFExpand(algorithm=hashes.SHA256(), length=32, info=b"key").derive(prk)
    nonce = HKDFExpand(algorithm=hashes.SHA256(), length=12, info=b"nonce").derive(prk)
    assert ChaCha20Poly1305(key).decrypt(nonce, aead_ct, b"") == response


def test_rejects_wrong_suite():
    sk, pk_raw = ohttp.generate_keypair()
    # Build a wire with the wrong AEAD ID
    import struct

    hdr = bytes([ohttp.KEY_CONFIG_ID]) + struct.pack(
        ">HHH",
        ohttp.KEM_ID_X25519,
        ohttp.KDF_ID_HKDF_SHA256,
        0x0001,  # AES-128-GCM
    )
    fake_wire = hdr + b"\x00" * 32 + b"\x00" * 16
    with pytest.raises(ValueError, match="unsupported"):
        ohttp.decapsulate_request(sk, fake_wire)


def test_rejects_short_input():
    sk, _ = ohttp.generate_keypair()
    with pytest.raises(ValueError, match="too short"):
        ohttp.decapsulate_request(sk, b"\x01")


def test_generate_keypair_is_independent():
    """Each invocation must produce an independent keypair — the HPKE key
    is intentionally not derived from any shared seed."""
    _, pk_a = ohttp.generate_keypair()
    _, pk_b = ohttp.generate_keypair()
    assert pk_a != pk_b


def test_rejects_tampered_ciphertext():
    sk, pk_raw = ohttp.generate_keypair()
    import struct

    hdr = bytes([ohttp.KEY_CONFIG_ID]) + struct.pack(
        ">HHH",
        ohttp.KEM_ID_X25519,
        ohttp.KDF_ID_HKDF_SHA256,
        ohttp.AEAD_ID_CHACHA20_POLY1305,
    )
    info = b"message/bhttp request" + b"\x00" + hdr
    pkr = ohttp._SUITE.kem.deserialize_public_key(pk_raw)
    enc, sender = ohttp._SUITE.create_sender_context(pkr, info=info)
    ct = sender.seal(b"hello", aad=b"")
    wire = bytearray(hdr + enc + ct)
    wire[-1] ^= 0xFF
    with pytest.raises(Exception):
        ohttp.decapsulate_request(sk, bytes(wire))
