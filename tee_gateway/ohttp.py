"""
Oblivious HTTP encapsulation for anonymous inference.

Implements request/response encapsulation per RFC 9458 (Oblivious HTTP)
with a fixed HPKE ciphersuite:
  - KEM:  DHKEM(X25519, HKDF-SHA256) (0x0020)
  - KDF:  HKDF-SHA256                (0x0001)
  - AEAD: ChaCha20-Poly1305          (0x0003)

Also implements the chunked-response extension from
draft-ietf-ohai-chunked-ohttp-08 for streaming responses (SSE inference):
  - Same HPKE context, separate export label "message/bhttp chunked response".
  - Wire: response_nonce || (varint(sealed_len) || sealed_ct)+
                          || varint(0) || sealed_final_ct
  - AEAD AAD: "" for non-final chunks, "final" for the last chunk.
  - Per-chunk nonce: aead_nonce XOR encode_be(counter), counter from 0.

The inner payload is application/json (or text/event-stream for chunked) —
we do not BHTTP-wrap the inference request, since the enclave is the terminal
endpoint and not a generic HTTP proxy. This is a documented divergence from
strict RFC 9458; the cryptographic construction is identical.

Trust model: the relay sees ciphertext + client IP; the enclave sees plaintext
+ relay IP. Unlinkability holds unless relay and enclave collude.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from pyhpke import AEADId, CipherSuite, KDFId, KEMId
from pyhpke.kem_key_interface import KEMKeyInterface


# RFC 9180 / 9458 algorithm identifiers
KEM_ID_X25519 = 0x0020
KDF_ID_HKDF_SHA256 = 0x0001
AEAD_ID_CHACHA20_POLY1305 = 0x0003

# Single, stable key configuration ID. Bump when the keypair or suite changes
# so clients can refuse stale configs.
KEY_CONFIG_ID = 0x01

# AEAD parameters for ChaCha20-Poly1305
_NK = 32  # key length
_NN = 12  # nonce length

# Per RFC 9458 §4.1/4.5 and draft-ietf-ohai-chunked-ohttp §3.1 — "info" labels.
_LABEL_REQUEST = b"message/bhttp request"
_LABEL_RESPONSE = b"message/bhttp response"
_LABEL_CHUNKED_RESPONSE = b"message/bhttp chunked response"

_SUITE = CipherSuite.new(
    KEMId.DHKEM_X25519_HKDF_SHA256,
    KDFId.HKDF_SHA256,
    AEADId.CHACHA20_POLY1305,
)


def encode_varint(value: int) -> bytes:
    """QUIC variable-length integer encoding (RFC 9000 §16).

    The top two bits of the first byte encode the length (00=1B, 01=2B,
    10=4B, 11=8B); the remaining bits hold the big-endian value. Used by
    draft-ietf-ohai-chunked-ohttp to frame each response chunk on the wire.
    """
    if value < 0:
        raise ValueError("varint must be non-negative")
    if value < (1 << 6):
        return bytes([value])
    if value < (1 << 14):
        return bytes([0x40 | (value >> 8), value & 0xFF])
    if value < (1 << 30):
        return struct.pack(">I", 0x80000000 | value)
    if value < (1 << 62):
        return struct.pack(">Q", 0xC000000000000000 | value)
    raise ValueError("varint value exceeds 2^62-1")


def decode_varint(buf: bytes, offset: int = 0) -> tuple[int, int]:
    """Parse one QUIC varint from ``buf`` starting at ``offset``. Returns
    ``(value, new_offset)``. Used by clients and tests."""
    if offset >= len(buf):
        raise ValueError("varint truncated")
    first = buf[offset]
    length_bits = first >> 6
    if length_bits == 0:
        return first, offset + 1
    if length_bits == 1:
        if offset + 2 > len(buf):
            raise ValueError("varint truncated")
        return ((first & 0x3F) << 8) | buf[offset + 1], offset + 2
    if length_bits == 2:
        if offset + 4 > len(buf):
            raise ValueError("varint truncated")
        head = bytes([first & 0x3F]) + buf[offset + 1 : offset + 4]
        return struct.unpack(">I", head)[0], offset + 4
    if offset + 8 > len(buf):
        raise ValueError("varint truncated")
    head = bytes([first & 0x3F]) + buf[offset + 1 : offset + 8]
    return struct.unpack(">Q", head)[0], offset + 8


def _header_bytes(key_id: int = KEY_CONFIG_ID) -> bytes:
    return bytes([key_id]) + struct.pack(
        ">HHH",
        KEM_ID_X25519,
        KDF_ID_HKDF_SHA256,
        AEAD_ID_CHACHA20_POLY1305,
    )


def key_config(public_key_raw: bytes, key_id: int = KEY_CONFIG_ID) -> bytes:
    """Build an OHTTP key configuration blob (RFC 9458 §3).

    Format:
        key_id(1) || kem_id(2) || public_key(Npk=32) ||
        symmetric_algorithms_length(2) || (kdf_id(2) || aead_id(2))+
    """
    if len(public_key_raw) != 32:
        raise ValueError("X25519 public key must be 32 bytes")
    symmetric = struct.pack(">HH", KDF_ID_HKDF_SHA256, AEAD_ID_CHACHA20_POLY1305)
    return (
        bytes([key_id])
        + struct.pack(">H", KEM_ID_X25519)
        + public_key_raw
        + struct.pack(">H", len(symmetric))
        + symmetric
    )


@dataclass
class DecapsulatedRequest:
    """Result of decapsulating an OHTTP-wrapped request.

    Two response secrets are exported up-front so the caller can switch
    between single-shot (``response_key``) and chunked
    (``response_key_chunked``) encapsulation after inspecting the inner
    body — the recipient context can only be created once per request.
    """

    plaintext: bytes
    response_key: bytes  # exported with label "message/bhttp response"
    response_key_chunked: bytes  # exported with label "message/bhttp chunked response"
    enc: bytes  # client's ephemeral public key, used as salt for response keying


def decapsulate_request(
    private_key: KEMKeyInterface, encapsulated_request: bytes
) -> DecapsulatedRequest:
    """Decrypt an HPKE-wrapped request inside the enclave.

    Raises ValueError on malformed input or unsupported ciphersuite. We
    never echo the underlying exception text to clients — it can leak
    timing/oracle info.
    """
    if len(encapsulated_request) < 7 + 32:
        raise ValueError("encapsulated request too short")

    key_id = encapsulated_request[0]
    kem_id, kdf_id, aead_id = struct.unpack(">HHH", encapsulated_request[1:7])
    if (key_id, kem_id, kdf_id, aead_id) != (
        KEY_CONFIG_ID,
        KEM_ID_X25519,
        KDF_ID_HKDF_SHA256,
        AEAD_ID_CHACHA20_POLY1305,
    ):
        raise ValueError("unsupported HPKE configuration")

    enc = encapsulated_request[7 : 7 + 32]
    aead_ct = encapsulated_request[7 + 32 :]

    info = _LABEL_REQUEST + b"\x00" + _header_bytes(key_id)
    recipient = _SUITE.create_recipient_context(enc, private_key, info=info)
    plaintext = recipient.open(aead_ct, aad=b"")

    # Two exports, one per response mode. RFC 9458 §4.5 and the chunked
    # draft §3.1 specify max(Nn, Nk) as the export length.
    export_len = max(_NN, _NK)
    response_secret = recipient.export(_LABEL_RESPONSE, export_len)
    response_secret_chunked = recipient.export(_LABEL_CHUNKED_RESPONSE, export_len)
    return DecapsulatedRequest(
        plaintext=plaintext,
        response_key=response_secret,
        response_key_chunked=response_secret_chunked,
        enc=enc,
    )


def encapsulate_response(response_secret: bytes, enc: bytes, plaintext: bytes) -> bytes:
    """Seal a response under the per-request derived key (RFC 9458 §4.5).

    Wire format: response_nonce(max(Nn, Nk)=Nk=32) || AEAD ciphertext
    """
    response_nonce = os.urandom(max(_NN, _NK))
    aead_key, aead_nonce = _derive_response_keys(response_secret, enc, response_nonce)
    ct = ChaCha20Poly1305(aead_key).encrypt(aead_nonce, plaintext, b"")
    return response_nonce + ct


def _derive_response_keys(
    response_secret: bytes, enc: bytes, response_nonce: bytes
) -> tuple[bytes, bytes]:
    """HKDF-Extract(enc || response_nonce, response_secret) then Expand for
    ``aead_key`` (Nk bytes, info=b"key") and ``aead_nonce`` (Nn bytes,
    info=b"nonce"). Shared by single-shot and chunked response paths."""
    h = hmac.HMAC(enc + response_nonce, hashes.SHA256())
    h.update(response_secret)
    prk = h.finalize()
    aead_key = HKDFExpand(algorithm=hashes.SHA256(), length=_NK, info=b"key").derive(
        prk
    )
    aead_nonce = HKDFExpand(
        algorithm=hashes.SHA256(), length=_NN, info=b"nonce"
    ).derive(prk)
    return aead_key, aead_nonce


class ChunkedResponseEncrypter:
    """Stream a chunked OHTTP response per draft-ietf-ohai-chunked-ohttp-08.

    Usage:
        enc = ChunkedResponseEncrypter(decap.response_key_chunked, decap.enc)
        yield enc.header()                        # response_nonce
        for plaintext in non_final_chunks:
            yield enc.encrypt_chunk(plaintext, is_final=False)
        yield enc.encrypt_chunk(last_plaintext, is_final=True)

    The final chunk uses AAD=b"final" with a zero-length varint prefix —
    that pair is what prevents an attacker from truncating the stream
    undetectably, so callers MUST always emit exactly one is_final=True
    chunk to terminate the response (even if its plaintext is empty).
    """

    def __init__(self, response_secret: bytes, enc: bytes):
        self._response_nonce = os.urandom(max(_NN, _NK))
        self._aead_key, self._aead_nonce = _derive_response_keys(
            response_secret, enc, self._response_nonce
        )
        self._aead = ChaCha20Poly1305(self._aead_key)
        self._counter = 0
        self._finalized = False

    def header(self) -> bytes:
        """Wire bytes that prefix the chunk stream."""
        return self._response_nonce

    def encrypt_chunk(self, plaintext: bytes, is_final: bool) -> bytes:
        if self._finalized:
            raise RuntimeError("ChunkedResponseEncrypter already finalized")
        ctr_bytes = self._counter.to_bytes(_NN, "big")
        chunk_nonce = bytes(a ^ b for a, b in zip(self._aead_nonce, ctr_bytes))
        aad = b"final" if is_final else b""
        sealed = self._aead.encrypt(chunk_nonce, plaintext, aad)
        self._counter += 1
        length_prefix = encode_varint(0) if is_final else encode_varint(len(sealed))
        if is_final:
            self._finalized = True
        return length_prefix + sealed


def generate_keypair() -> tuple[KEMKeyInterface, bytes]:
    """Generate a fresh X25519 keypair for HPKE.

    The HPKE keypair is intentionally independent of the RSA TEE signing
    key: deriving one from the other would create a single point of
    compromise (a leak of the RSA private key would also leak the OHTTP
    private key, and vice versa). Both public keys are still covered by
    the same nitriding attestation transcript, so verifiers get binding
    without sharing key material.

    pyhpke 0.6 derives the keypair from random IKM via
    ``kem.derive_key_pair(ikm)``; we feed it ``os.urandom(32)`` so each
    enclave boot produces an independent keypair.
    """
    pair = _SUITE.kem.derive_key_pair(os.urandom(32))
    return pair.private_key, pair.public_key.to_public_bytes()
