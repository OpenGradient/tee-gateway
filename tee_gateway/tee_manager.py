"""
TEE Key Manager
Handles RSA key pair generation, nitriding registration, and request/response signing.
Single source of truth for the enclave's signing key — used by all controllers.
"""

import logging
import hashlib
import base64
import secrets
import urllib.request
from datetime import datetime, UTC

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from eth_account import Account
from eth_hash.auto import keccak

logger = logging.getLogger(__name__)

NITRIDING_BASE_URL = "http://127.0.0.1:8080"


class TEEKeyManager:
    """Manages private/public key pair for TEE attestation and signing.

    A single instance of this class is created at startup and shared across
    all controllers. It registers its public key with nitriding so that the
    attestation document and the signing key are always the same key pair.
    """

    def __init__(self, register=True):
        self.private_key = None
        self.public_key = None
        self.public_key_pem = None
        self.tee_id = None
        self.wallet_address = None
        self._generate_keys()
        if register:
            self.register_with_nitriding()

    def _generate_keys(self):
        """Generate RSA key pair and derive the tee_id."""
        logger.info("Generating TEE RSA key pair...")
        self.private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

        self.public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        # tee_id = keccak256(abi.encodePacked(signingKey))
        # signingKey is the DER-encoded public key (canonical binary form, no
        # encoding ambiguity). DER bytes are exactly the base64 body of the PEM
        # decoded, so external verifiers can strip the PEM headers, base64-decode
        # the body, and keccak256-hash the resulting bytes.
        public_key_der = self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.tee_id = keccak(public_key_der).hex()

        # Generate Ethereum wallet key pair inside the TEE so it is part of
        # the attestation trust boundary and never leaves the enclave.
        wallet_key_bytes = secrets.token_bytes(32)
        wallet_account = Account.from_key(wallet_key_bytes)
        self.wallet_address = wallet_account.address

        logger.info("TEE key pair generated successfully")
        logger.info(f"tee_id: 0x{self.tee_id}")
        logger.info(f"wallet_address: {self.wallet_address}")

    def register_with_nitriding(self):
        """Register public key hash with nitriding."""
        try:
            public_key_der = self.public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

            key_hash = hashlib.sha256(public_key_der).digest()
            key_hash_b64 = base64.b64encode(key_hash).decode("utf-8")

            logger.info(f"Public key DER length: {len(public_key_der)} bytes")
            logger.info(f"Public key SHA256 hash (hex): {key_hash.hex()}")
            logger.info(f"Public key SHA256 hash (base64): {key_hash_b64}")

            url = f"{NITRIDING_BASE_URL}/enclave/hash"
            req = urllib.request.Request(
                url, data=key_hash_b64.encode("utf-8"), method="POST"
            )

            response = urllib.request.urlopen(req, timeout=5)
            response_body = response.read().decode("utf-8")

            if response.getcode() == 200:
                logger.info("Successfully registered public key hash with nitriding")
                logger.info(f"Response: {response_body}")
                return True
            else:
                logger.error(
                    f"Failed to register public key hash: HTTP {response.getcode()}"
                )
                return False

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else "No error body"
            logger.error(f"HTTP Error {e.code}: {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.warning(
                f"Could not register with nitriding (may not be in TEE): {e}"
            )
            return False

    def sign_data(self, data: bytes) -> str:
        """Sign msg_hash bytes with RSA-PSS-SHA256, return base64 signature.

        Expects pre-computed bytes (e.g. the keccak256 msg_hash from
        compute_tee_msg_hash). RSA-PSS hashes the input again with SHA256
        internally, matching the double-hash the on-chain verifier uses.

        Salt length is fixed at 32 bytes (SHA256 digest size) to match the
        on-chain verifier's expectations.
        """
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size,  # 32 bytes
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def get_public_key(self) -> str:
        """Return public key in PEM format."""
        return self.public_key_pem

    def get_tee_id(self) -> str:
        """Return the tee_id: keccak256(abi.encodePacked(public_key_der))."""
        return self.tee_id

    def get_wallet_address(self) -> str:
        """Return the TEE-generated Ethereum wallet address (checksum)."""
        return self.wallet_address

    def get_attestation_document(self) -> dict:
        """Return TEE attestation document."""
        return {
            "public_key": self.public_key_pem,
            "tee_id": f"0x{self.tee_id}",
            "wallet_address": self.wallet_address,
            "timestamp": datetime.now(UTC).isoformat(),
            "enclave_info": {
                "platform": "aws-nitro",
                "instance_type": "tee-enabled",
                "version": "1.0.0",
            },
            "measurements": None,  # Would contain PCR values in real deployment
        }


def signal_ready():
    """Signal to nitriding that enclave is ready to accept traffic."""
    try:
        url = f"{NITRIDING_BASE_URL}/enclave/ready"
        r = urllib.request.urlopen(url)
        if r.getcode() != 200:
            raise Exception(f"Expected status code 200 but got {r.getcode()}")
        logger.info("Successfully signaled ready to nitriding")
    except Exception as e:
        logger.warning(f"Could not signal nitriding (may not be in TEE): {e}")


def compute_tee_msg_hash(
    request_bytes: bytes,
    response_content: str,
    timestamp: int,
) -> tuple:
    """Compute msg_hash matching the on-chain verifier:
      keccak256(abi.encodePacked(inputHash, outputHash, timestamp))
    where inputHash and outputHash are each keccak256 bytes32 values
    and timestamp is uint256 (big-endian 32 bytes).

    Returns (msg_hash_bytes, input_hash_hex, output_hash_hex).
    """
    input_hash = keccak(request_bytes)
    output_hash = keccak(response_content.encode("utf-8"))
    msg_hash = keccak(input_hash + output_hash + timestamp.to_bytes(32, "big"))

    return msg_hash, input_hash.hex(), output_hash.hex()


# Singleton instance — initialized lazily or eagerly depending on environment
_tee_keys: TEEKeyManager | None = None


def get_tee_keys() -> TEEKeyManager:
    """Get or create the singleton TEE key manager."""
    global _tee_keys
    if _tee_keys is None:
        _tee_keys = TEEKeyManager(register=True)
    return _tee_keys


def initialize_tee():
    """
    Initialize TEE: generate keys, register with nitriding, signal ready.
    Call this during application startup.
    """
    logger.info("Initializing TEE...")
    keys = get_tee_keys()
    signal_ready()
    logger.info(f"TEE initialized. tee_id: 0x{keys.get_tee_id()}")
    logger.info(f"Public key (first 80 chars): {keys.get_public_key()[:80]}...")
    return keys
