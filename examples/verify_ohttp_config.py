"""Verify a signed OHTTP/HPKE key configuration against the attested TEE key.

Unlike the response signature (which clients already verify per-inference), the
HPKE key used for anonymous inference is **not** part of the Nitro attestation
transcript. Instead the enclave binds it to its attested RSA signing key with an
RSA-PSS signature over a keccak256 config hash (the same hash
``TEERegistryV2.computeOHTTPConfigHash`` computes on-chain).

So before encrypting anything to the HPKE public key, a client/relay MUST:

  1. Obtain the enclave's signing key via the Nitro attestation document
     (``/enclave/attestation`` → verified against the AWS Nitro root CA, see
     ``verify_attestation.py``). This is the only trust anchor.
  2. Fetch the signed config from ``GET /v1/ohttp/config`` (or read the ``hpke``
     field embedded in ``/signing-key``).
  3. Confirm the config's ``tee_id`` equals keccak256(DER(signing key)) — this
     ties the config to *this* attested key, not some other enclave's.
  4. Recompute the config hash from the config fields and check it matches
     ``signature_hash``.
  5. Verify the RSA-PSS signature over that hash with the attested public key.

If any step fails, the HPKE public key is NOT attested — do not use it.

The trust chain is: PCRs → attested RSA signing key → RSA-PSS signature → HPKE
config. Skipping step 5 leaves you with an unauthenticated key that any
network attacker (including the relay) could swap.
"""

import base64

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from eth_hash.auto import keccak

# ---------------------------------------------------------------------------
# INPUTS — replace with values from your deployment.
# ---------------------------------------------------------------------------

# Attested signing key, recovered from the verified Nitro attestation document.
# (Here we hardcode a PEM only as a placeholder; in practice this MUST come from
# attestation, never from the same /v1/ohttp/config response you are verifying.)
public_key_pem = """-----BEGIN PUBLIC KEY-----
PASTE_THE_ATTESTED_PUBLIC_KEY_PEM_HERE
-----END PUBLIC KEY-----"""

# The JSON object returned by GET /v1/ohttp/config (== the "hpke" field of
# /signing-key). Field names match TEEKeyManager.get_signed_hpke_config().
signed_config = {
    "tee_id": "0x...",  # keccak256(DER(signing key))
    "key_id": 0x01,
    "kem_id": 0x0020,  # DHKEM(X25519, HKDF-SHA256)
    "kdf_id": 0x0001,  # HKDF-SHA256
    "aead_id": 0x0003,  # ChaCha20-Poly1305
    "public_key": "...",  # hex, raw X25519 public key (32 bytes)
    "key_config": "...",  # base64, RFC 9458 §3 key-config blob
    "signature": "...",  # base64, RSA-PSS-SHA256 over the config hash
    "signature_hash": "0x...",  # hex, keccak256 config hash (for convenience)
}


# ---------------------------------------------------------------------------
# This must byte-match tee_manager.compute_ohttp_config_hash and the on-chain
# TEERegistryV2.computeOHTTPConfigHash. Layout (every fixed field a 32-byte
# word, dynamic fields pre-hashed):
#   keccak256(
#       keccak256("OPENGRADIENT_TEE_OHTTP_CONFIG_V1")  // domain
#       || tee_id                                       // bytes32
#       || uint256(key_id) || uint256(kem_id)
#       || uint256(kdf_id) || uint256(aead_id)
#       || keccak256(public_key)                        // bytes32
#       || keccak256(key_config)                        // bytes32
#   )
# ---------------------------------------------------------------------------
def compute_ohttp_config_hash(
    tee_id: bytes,
    key_id: int,
    kem_id: int,
    kdf_id: int,
    aead_id: int,
    ohttp_public_key: bytes,
    ohttp_key_config: bytes,
) -> bytes:
    if len(tee_id) != 32:
        raise ValueError("tee_id must be 32 bytes")

    def word(value: int) -> bytes:
        return value.to_bytes(32, "big")

    domain = keccak(b"OPENGRADIENT_TEE_OHTTP_CONFIG_V1")
    return keccak(
        domain
        + tee_id
        + word(key_id)
        + word(kem_id)
        + word(kdf_id)
        + word(aead_id)
        + keccak(ohttp_public_key)
        + keccak(ohttp_key_config)
    )


def main() -> None:
    print("=" * 70)
    print("OHTTP CONFIG VERIFICATION")
    print("=" * 70)

    public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))

    # --- Step 1: config is bound to THIS attested signing key ---------------
    # tee_id = keccak256(DER(SubjectPublicKeyInfo)) — recompute it from the
    # attested key and require an exact match. This is what prevents a
    # signature made by enclave A's key from being presented as a binding for
    # enclave B.
    public_key_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    expected_tee_id = keccak(public_key_der)
    claimed_tee_id = bytes.fromhex(signed_config["tee_id"].removeprefix("0x"))

    print("\n[1] Binding config to attested signing key")
    print(f"    tee_id (from attested key): 0x{expected_tee_id.hex()}")
    print(f"    tee_id (claimed in config): 0x{claimed_tee_id.hex()}")
    if claimed_tee_id != expected_tee_id:
        raise SystemExit("✗ tee_id does not match the attested key — REJECT")
    print("    ✓ config is bound to the attested signing key")

    # --- Step 2: recompute the signed hash ----------------------------------
    config_hash = compute_ohttp_config_hash(
        expected_tee_id,
        signed_config["key_id"],
        signed_config["kem_id"],
        signed_config["kdf_id"],
        signed_config["aead_id"],
        bytes.fromhex(signed_config["public_key"]),
        base64.b64decode(signed_config["key_config"]),
    )

    print("\n[2] Recomputing config hash")
    print(f"    computed:  0x{config_hash.hex()}")
    print(f"    reported:  {signed_config['signature_hash']}")
    reported_hash = bytes.fromhex(signed_config["signature_hash"].removeprefix("0x"))
    if config_hash != reported_hash:
        raise SystemExit("✗ recomputed hash != signature_hash — REJECT")
    print("    ✓ hash matches signature_hash")

    # --- Step 3: verify the RSA-PSS signature over the hash ------------------
    # salt_length=32 (SHA256 digest size) matches TEEKeyManager.sign_data.
    print("\n[3] Verifying RSA-PSS-SHA256 signature")
    try:
        public_key.verify(
            base64.b64decode(signed_config["signature"]),
            config_hash,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )
    except Exception as e:  # noqa: BLE001 — surface any verification failure
        raise SystemExit(f"✗ signature verification FAILED: {e}")

    print("    ✓ signature verifies against the attested key")
    print("\n" + "=" * 70)
    print("✓✓✓ HPKE CONFIG VERIFIED ✓✓✓")
    print("=" * 70)
    print(
        "\nThe HPKE public key below is bound to the attested enclave; it is\n"
        "safe to HPKE-encapsulate your request to it:\n"
        f"  {signed_config['public_key']}"
    )


if __name__ == "__main__":
    main()
