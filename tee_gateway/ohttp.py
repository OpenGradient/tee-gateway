"""
Oblivious HTTP encapsulation for anonymous inference.

Implements request/response encapsulation per RFC 9458 (Oblivious HTTP)
with a fixed HPKE ciphersuite:
  - KEM:  DHKEM(X25519, HKDF-SHA256) (0x0020)
  - KDF:  HKDF-SHA256                (0x0001)
  - AEAD: ChaCha20-Poly1305          (0x0003)

The inner payload is application/json — we do not BHTTP-wrap the inference
request, since the enclave is the terminal endpoint and not a generic HTTP
proxy. This is a documented divergence from strict RFC 9458; the cryptographic
construction (HPKE base + exported response keying) is identical.

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

# Per RFC 9458 §4.1/4.2 — "info" labels for the HPKE context.
_LABEL_REQUEST = b"message/bhttp request"
_LABEL_RESPONSE = b"message/bhttp response"

_SUITE = CipherSuite.new(
    KEMId.DHKEM_X25519_HKDF_SHA256,
    KDFId.HKDF_SHA256,
    AEADId.CHACHA20_POLY1305,
)


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
    """Result of decapsulating an OHTTP-wrapped request."""

    plaintext: bytes
    response_key: bytes  # 32 bytes exported from the HPKE context
    enc: bytes  # client's ephemeral public key, used as salt for the response


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

    # Export a fresh secret bound to this HPKE context, used to derive the
    # response AEAD key. This is the OHTTP-defined separation between the
    # request and response halves of the same exchange.
    response_secret = recipient.export(_LABEL_RESPONSE, _NK)
    return DecapsulatedRequest(
        plaintext=plaintext, response_key=response_secret, enc=enc
    )


def encapsulate_response(response_secret: bytes, enc: bytes, plaintext: bytes) -> bytes:
    """Seal a response under the per-request derived key (RFC 9458 §4.2).

    Wire format: response_nonce(max(Nn, Nk)=Nk=32) || AEAD ciphertext
    """
    response_nonce = os.urandom(max(_NN, _NK))
    salt = enc + response_nonce

    h = hmac.HMAC(salt, hashes.SHA256())
    h.update(response_secret)
    prk = h.finalize()

    aead_key = HKDFExpand(algorithm=hashes.SHA256(), length=_NK, info=b"key").derive(
        prk
    )
    aead_nonce = HKDFExpand(
        algorithm=hashes.SHA256(), length=_NN, info=b"nonce"
    ).derive(prk)

    ct = ChaCha20Poly1305(aead_key).encrypt(aead_nonce, plaintext, b"")
    return response_nonce + ct


def generate_keypair() -> tuple[KEMKeyInterface, bytes]:
    """Generate an X25519 keypair for HPKE. Returns (private_key, public_key_raw).

    pyhpke 0.6 derives keys from random IKM via ``kem.derive_key_pair(ikm)``,
    which returns a ``KEMKeyPair`` wrapper. We hold onto the private side for
    decapsulation and serialize the public side to raw 32-byte form for the
    key configuration blob.
    """
    pair = _SUITE.kem.derive_key_pair(os.urandom(32))
    pk_raw = pair.public_key.to_public_bytes()
    return pair.private_key, pk_raw
