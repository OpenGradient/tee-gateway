"""
Smoke test: exercise the OHTTP anonymous-inference endpoints end-to-end against
a running gateway. Mirrors test_bytedance.py — point it at a local server and
verify a request round-trips through HPKE encap/decap + the chat backend.

Usage:
    # Non-streaming round-trip (default):
    uv run python scripts/test_ohttp.py
    uv run python scripts/test_ohttp.py --model gpt-4.1 --prompt "what model are you?"

    # Streaming via chunked OHTTP:
    uv run python scripts/test_ohttp.py --stream --model claude-haiku-4-5

    # Against a remote gateway (anything reachable via HTTP/S):
    uv run python scripts/test_ohttp.py --url https://my-enclave.example

The gateway must already have provider keys injected via POST /v1/keys for the
inner chat request to succeed. Run scripts/test_bytedance.py first if unsure —
it shares the same backend.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import IO, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402
from cryptography.hazmat.primitives import hashes, hmac  # noqa: E402
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305  # noqa: E402
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand  # noqa: E402
from pyhpke import AEADId, CipherSuite, KDFId, KEMId  # noqa: E402

# Fixed ciphersuite — the gateway only accepts this combination.
_SUITE = CipherSuite.new(
    KEMId.DHKEM_X25519_HKDF_SHA256,
    KDFId.HKDF_SHA256,
    AEADId.CHACHA20_POLY1305,
)
_KEM_ID = 0x0020
_KDF_ID = 0x0001
_AEAD_ID = 0x0003
_NK = 32
_NN = 12
_LABEL_REQ = b"message/bhttp request"
_LABEL_RESP = b"message/bhttp response"
_LABEL_RESP_CHUNKED = b"message/bhttp chunked response"


# ---------------------------------------------------------------------------
# HPKE encapsulation (client side)
# ---------------------------------------------------------------------------


def encapsulate_request(
    public_key_raw: bytes, key_id: int, plaintext: bytes
) -> tuple[bytes, object, bytes]:
    """Build an OHTTP-encapsulated request. Returns ``(wire, sender, enc)``.

    The sender context is kept by the caller so it can later export the
    response secret (single-shot or chunked) and decrypt the reply."""
    hdr = bytes([key_id]) + struct.pack(">HHH", _KEM_ID, _KDF_ID, _AEAD_ID)
    info = _LABEL_REQ + b"\x00" + hdr
    pkr = _SUITE.kem.deserialize_public_key(public_key_raw)
    enc, sender = _SUITE.create_sender_context(pkr, info=info)
    ct = sender.seal(plaintext, aad=b"")
    return hdr + enc + ct, sender, enc


# ---------------------------------------------------------------------------
# Response decryption (single-shot + chunked)
# ---------------------------------------------------------------------------


def _derive_response_keys(
    response_secret: bytes, enc: bytes, response_nonce: bytes
) -> tuple[bytes, bytes]:
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


def decrypt_single_shot(sealed: bytes, sender, enc: bytes) -> bytes:
    response_secret = sender.export(_LABEL_RESP, max(_NN, _NK))
    nonce_len = max(_NN, _NK)
    response_nonce = sealed[:nonce_len]
    aead_ct = sealed[nonce_len:]
    aead_key, aead_nonce = _derive_response_keys(response_secret, enc, response_nonce)
    return ChaCha20Poly1305(aead_key).decrypt(aead_nonce, aead_ct, b"")


def _read_varint(stream: IO[bytes]) -> int | None:
    """Read one QUIC varint from a byte stream. Returns None at clean EOF."""
    first = stream.read(1)
    if not first:
        return None
    b = first[0]
    nbytes = 1 << (b >> 6)  # 1, 2, 4, or 8
    rest = stream.read(nbytes - 1) if nbytes > 1 else b""
    if len(rest) != nbytes - 1:
        raise ValueError("truncated varint")
    head = bytes([b & 0x3F]) + rest
    if nbytes == 1:
        return b & 0x3F
    if nbytes == 2:
        return (head[0] << 8) | head[1]
    if nbytes == 4:
        return struct.unpack(">I", head)[0]
    return struct.unpack(">Q", head)[0]


def decrypt_chunked_stream(
    raw_stream: IO[bytes], sender, enc: bytes
) -> Iterator[bytes]:
    """Decrypt a chunked OHTTP response (draft-ietf-ohai-chunked-ohttp-08)
    incrementally, yielding plaintext as each sealed chunk arrives.

    Wire: response_nonce || (varint(len) || ct)+ || varint(0) || final_ct
    Per-chunk nonce = aead_nonce XOR encode_be(counter), AAD=""/b"final".
    """
    response_secret = sender.export(_LABEL_RESP_CHUNKED, max(_NN, _NK))
    response_nonce = raw_stream.read(max(_NN, _NK))
    if len(response_nonce) != max(_NN, _NK):
        raise ValueError("truncated response_nonce")
    aead_key, aead_nonce = _derive_response_keys(response_secret, enc, response_nonce)
    aead = ChaCha20Poly1305(aead_key)
    counter = 0

    while True:
        length = _read_varint(raw_stream)
        if length is None:
            raise ValueError("stream ended without AAD=final chunk")

        chunk_nonce = bytes(
            a ^ b for a, b in zip(aead_nonce, counter.to_bytes(_NN, "big"))
        )
        counter += 1

        if length == 0:
            final_ct = raw_stream.read()  # rest of stream
            yield aead.decrypt(chunk_nonce, final_ct, b"final")
            return

        ct = raw_stream.read(length)
        if len(ct) != length:
            raise ValueError("truncated chunk ciphertext")
        yield aead.decrypt(chunk_nonce, ct, b"")


# ---------------------------------------------------------------------------
# Wire-payload dump (for visual confirmation that nothing plaintext leaves)
# ---------------------------------------------------------------------------


def _hexdump(data: bytes, max_bytes: int = 96) -> str:
    """xxd-style hex dump, capped at ``max_bytes`` so output stays readable."""
    out: list[str] = []
    for i in range(0, min(len(data), max_bytes), 16):
        row = data[i : i + 16]
        hex_part = " ".join(f"{b:02x}" for b in row)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        out.append(f"  {i:04x}  {hex_part:<48}  |{ascii_part}|")
    if len(data) > max_bytes:
        out.append(f"  ...   ({len(data) - max_bytes} more bytes of ciphertext)")
    return "\n".join(out)


def dump_outgoing(
    url: str,
    headers: dict[str, str],
    inner_plaintext: bytes,
    wire: bytes,
    enc: bytes,
    key_id: int,
) -> None:
    """Print everything that's about to go on the wire so you can eyeball
    that the relay only sees opaque ciphertext."""
    print("\n================ OUTGOING REQUEST ================")
    print(f"POST {url}")
    for name, value in headers.items():
        print(f"  {name}: {value}")
    print(
        f"\n  inner plaintext ({len(inner_plaintext)} bytes, "
        f"NEVER goes on the wire — sealed under HPKE):"
    )
    try:
        print(
            "    "
            + json.dumps(json.loads(inner_plaintext), indent=4).replace("\n", "\n    ")
        )
    except json.JSONDecodeError:
        print(f"    {inner_plaintext!r}")

    # The wire body decomposes into:
    #   header (7 bytes): key_id || kem_id || kdf_id || aead_id
    #   enc    (32 bytes): client's ephemeral X25519 public key
    #   ct     (rest):     AEAD ciphertext + 16-byte tag — opaque to the relay
    print(f"\n  encapsulated body ({len(wire)} bytes total):")
    print(
        f"    OHTTP header   = {wire[:7].hex()}  "
        f"(key_id=0x{key_id:02x}, suite=0x0020/0x0001/0x0003)"
    )
    print(f"    enc (ephemeral)= {enc.hex()}")
    print(
        f"    ciphertext     = {len(wire) - 7 - 32} bytes (HPKE-sealed, no plaintext leaks):"
    )
    print(_hexdump(wire[7 + 32 :]))
    print("==================================================\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def fetch_config(base_url: str) -> tuple[bytes, int]:
    """GET /v1/ohttp/config; return (public_key_raw_bytes, key_id)."""
    r = requests.get(f"{base_url}/v1/ohttp/config", timeout=10)
    r.raise_for_status()
    cfg = r.json()
    print(
        f"HPKE config: key_id={cfg['key_id']} suite={cfg['kem_id']}/{cfg['kdf_id']}/{cfg['aead_id']}"
    )
    print(f"  public_key = {cfg['public_key']}")
    print(f"  key_config = {cfg['key_config']}")
    pk_raw = bytes.fromhex(cfg["public_key"])
    if len(pk_raw) != 32:
        raise ValueError(f"expected 32-byte X25519 pubkey, got {len(pk_raw)}")
    return pk_raw, cfg["key_id"]


def verify_attestation_binding(base_url: str, hpke_pubkey_hex: str) -> None:
    """Cross-check that the HPKE pubkey we just got matches what the
    attestation document at /signing-key reports. A network attacker
    between us and the enclave could otherwise swap in their own
    pubkey and decrypt our prompt."""
    r = requests.get(f"{base_url}/signing-key", timeout=10)
    r.raise_for_status()
    doc = r.json()
    attested = (doc.get("hpke") or {}).get("public_key")
    if attested is None:
        print(
            "WARN: /signing-key did not include hpke.public_key; skipping binding check"
        )
        return
    if attested.lower() != hpke_pubkey_hex.lower():
        raise ValueError(
            f"HPKE pubkey mismatch! config={hpke_pubkey_hex} attestation={attested}"
        )
    print("HPKE pubkey matches the attestation document")


def run_non_streaming(
    base_url: str, model: str, prompt: str, payment: str | None
) -> int:
    pk_raw, key_id = fetch_config(base_url)
    verify_attestation_binding(base_url, pk_raw.hex())

    inner = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    }
    inner_bytes = json.dumps(inner, separators=(",", ":")).encode("utf-8")
    wire, sender, enc = encapsulate_request(pk_raw, key_id, inner_bytes)

    headers = {"Content-Type": "message/ohttp-req"}
    if payment:
        headers["X-Payment"] = payment

    dump_outgoing(f"{base_url}/v1/ohttp", headers, inner_bytes, wire, enc, key_id)
    r = requests.post(f"{base_url}/v1/ohttp", data=wire, headers=headers, timeout=120)
    print(f"HTTP {r.status_code}  content-type={r.headers.get('content-type')}")

    if r.status_code >= 400 or "ohttp-res" not in (r.headers.get("content-type") or ""):
        # Plaintext error pass-through (e.g. 402 with x402 payment requirements).
        print("---- plaintext error body ----")
        try:
            print(json.dumps(r.json(), indent=2))
        except ValueError:
            print(r.text)
        return 1

    print("---- forwarded headers ----")
    for k, v in r.headers.items():
        if k.lower().startswith(
            ("x-payment", "x-upto", "x-settlement", "x-tee", "x-usage")
        ):
            print(f"  {k}: {v}")

    plaintext = decrypt_single_shot(r.content, sender, enc)
    print("---- decrypted response ----")
    try:
        parsed = json.loads(plaintext)
        choices = parsed.get("choices") or []
        if choices:
            print(choices[0].get("message", {}).get("content", "<empty>"))
        print("---- usage ----")
        print(json.dumps(parsed.get("usage"), indent=2))
        for field in ("tee_signature", "tee_request_hash", "tee_output_hash"):
            if field in parsed:
                print(f"  {field}: {parsed[field][:40]}...")
    except json.JSONDecodeError:
        print(plaintext.decode("utf-8", errors="replace"))
    return 0


def run_streaming(base_url: str, model: str, prompt: str, payment: str | None) -> int:
    pk_raw, key_id = fetch_config(base_url)
    verify_attestation_binding(base_url, pk_raw.hex())

    inner = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "stream": True,
    }
    inner_bytes = json.dumps(inner, separators=(",", ":")).encode("utf-8")
    wire, sender, enc = encapsulate_request(pk_raw, key_id, inner_bytes)

    headers = {"Content-Type": "message/ohttp-req"}
    if payment:
        headers["X-Payment"] = payment

    dump_outgoing(f"{base_url}/v1/ohttp", headers, inner_bytes, wire, enc, key_id)
    r = requests.post(
        f"{base_url}/v1/ohttp",
        data=wire,
        headers=headers,
        timeout=120,
        stream=True,
    )
    print(f"HTTP {r.status_code}  content-type={r.headers.get('content-type')}")

    ct = r.headers.get("content-type", "")
    if r.status_code >= 400 or "chunked-res" not in ct:
        print("---- non-streaming response body ----")
        print(r.text)
        return 1

    print("---- decrypted SSE events ----")
    # urllib3's HTTPResponse — supports .read(n) without re-decoding chunked transfer.
    for plaintext in decrypt_chunked_stream(r.raw, sender, enc):
        text = plaintext.decode("utf-8", errors="replace")
        sys.stdout.write(text)
        sys.stdout.flush()
    print("\n---- end of stream ----")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=os.environ.get("OHTTP_URL", "http://127.0.0.1:8000")
    )
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument(
        "--prompt", default="What model are you? Reply in one short sentence."
    )
    parser.add_argument(
        "--stream", action="store_true", help="Use chunked OHTTP streaming"
    )
    parser.add_argument(
        "--payment",
        default=os.environ.get("X_PAYMENT"),
        help="Optional x402 payment payload (base64). Send via the outer X-Payment header.",
    )
    args = parser.parse_args()

    try:
        if args.stream:
            return run_streaming(args.url, args.model, args.prompt, args.payment)
        return run_non_streaming(args.url, args.model, args.prompt, args.payment)
    except requests.RequestException as exc:
        print(f"\nERROR: HTTP failure — {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
