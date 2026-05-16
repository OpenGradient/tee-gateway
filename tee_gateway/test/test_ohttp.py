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


def test_varint_round_trip():
    """QUIC varint encoder/decoder matches across all 4 length classes."""
    for value in (0, 63, 64, 16383, 16384, 1073741823, 1073741824, (1 << 62) - 1):
        encoded = ohttp.encode_varint(value)
        decoded, off = ohttp.decode_varint(encoded)
        assert decoded == value, value
        assert off == len(encoded)

    with pytest.raises(ValueError):
        ohttp.encode_varint(-1)
    with pytest.raises(ValueError):
        ohttp.encode_varint(1 << 62)


def test_chunked_response_round_trip():
    """Client encrypts a chunked response stream; client-side decrypter
    must recover every chunk and detect the AAD=b'final' terminator."""
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    sk, pk_raw = ohttp.generate_keypair()
    import struct as _struct

    hdr = bytes([ohttp.KEY_CONFIG_ID]) + _struct.pack(
        ">HHH",
        ohttp.KEM_ID_X25519,
        ohttp.KDF_ID_HKDF_SHA256,
        ohttp.AEAD_ID_CHACHA20_POLY1305,
    )
    info = b"message/bhttp request" + b"\x00" + hdr
    pkr = ohttp._SUITE.kem.deserialize_public_key(pk_raw)
    enc, sender = ohttp._SUITE.create_sender_context(pkr, info=info)
    wire = hdr + enc + sender.seal(b'{"stream": true}', aad=b"")

    decap = ohttp.decapsulate_request(sk, wire)
    assert decap.response_key_chunked == sender.export(
        b"message/bhttp chunked response", 32
    )

    # Server side: stream three chunks plus an empty final marker.
    plaintexts = [b"data: {chunk1}\n\n", b"data: {chunk2}\n\n", b"data: [DONE]\n\n"]
    encrypter = ohttp.ChunkedResponseEncrypter(decap.response_key_chunked, decap.enc)
    wire_chunks = [encrypter.header()]
    for pt in plaintexts[:-1]:
        wire_chunks.append(encrypter.encrypt_chunk(pt, is_final=False))
    wire_chunks.append(encrypter.encrypt_chunk(plaintexts[-1], is_final=True))
    stream = b"".join(wire_chunks)

    # Client side: re-derive keys, walk the varint-framed chunks.
    response_nonce = stream[:32]
    off = 32
    response_secret = sender.export(b"message/bhttp chunked response", 32)
    aead_key, aead_nonce = ohttp._derive_response_keys(
        response_secret, enc, response_nonce
    )
    aead = ChaCha20Poly1305(aead_key)

    recovered: list[bytes] = []
    counter = 0
    while off < len(stream):
        length, off = ohttp.decode_varint(stream, off)
        is_final = length == 0
        # On the final chunk the prefix is zero; the actual sealed length is
        # plaintext_len + 16 (Poly1305 tag). The chunk consumes the rest.
        seg_len = len(stream) - off if is_final else length
        ct = stream[off : off + seg_len]
        off += seg_len
        chunk_nonce = bytes(
            a ^ b for a, b in zip(aead_nonce, counter.to_bytes(12, "big"))
        )
        aad = b"final" if is_final else b""
        recovered.append(aead.decrypt(chunk_nonce, ct, aad))
        counter += 1
        if is_final:
            break

    assert recovered == plaintexts
    # The decrypter MUST reject the same stream with the final-AAD swapped
    # — protects against undetected truncation at the boundary.
    with pytest.raises(Exception):
        aead.decrypt(
            bytes(a ^ b for a, b in zip(aead_nonce, (counter - 1).to_bytes(12, "big"))),
            stream[-len(ct) :],
            b"",
        )


def test_chunked_encrypter_rejects_double_finalize():
    sk, pk_raw = ohttp.generate_keypair()
    import struct as _struct

    hdr = bytes([ohttp.KEY_CONFIG_ID]) + _struct.pack(
        ">HHH",
        ohttp.KEM_ID_X25519,
        ohttp.KDF_ID_HKDF_SHA256,
        ohttp.AEAD_ID_CHACHA20_POLY1305,
    )
    info = b"message/bhttp request" + b"\x00" + hdr
    pkr = ohttp._SUITE.kem.deserialize_public_key(pk_raw)
    enc, sender = ohttp._SUITE.create_sender_context(pkr, info=info)
    decap = ohttp.decapsulate_request(sk, hdr + enc + sender.seal(b"hi", aad=b""))

    encrypter = ohttp.ChunkedResponseEncrypter(decap.response_key_chunked, decap.enc)
    encrypter.header()
    encrypter.encrypt_chunk(b"only", is_final=True)
    with pytest.raises(RuntimeError, match="already finalized"):
        encrypter.encrypt_chunk(b"extra", is_final=False)


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
