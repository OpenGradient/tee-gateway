"""
TEE Key Manager
Handles RSA key pair generation, nitriding registration, and request/response signing.
Extracted from server.py for use in the Flask/connexion middleware.
"""

import os
import logging
import hashlib
import base64
import json
import urllib.request
from datetime import datetime, UTC

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

NITRIDING_BASE_URL = "http://127.0.0.1:8080"


class TEEKeyManager:
    """Manages private/public key pair for TEE attestation and signing"""

    def __init__(self, register=True):
        self.private_key = None
        self.public_key = None
        self.public_key_pem = None
        self._generate_keys()
        if register:
            self.register_with_nitriding()

    def _generate_keys(self):
        """Generate RSA key pair for signing inference results"""
        logger.info("Generating TEE RSA key pair...")
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

        self.public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

        logger.info("TEE key pair generated successfully")

    def register_with_nitriding(self):
        """Register public key hash with nitriding"""
        try:
            public_key_der = self.public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            key_hash = hashlib.sha256(public_key_der).digest()
            key_hash_b64 = base64.b64encode(key_hash).decode('utf-8')

            logger.info(f"Public key DER length: {len(public_key_der)} bytes")
            logger.info(f"Public key SHA256 hash (hex): {key_hash.hex()}")
            logger.info(f"Public key SHA256 hash (base64): {key_hash_b64}")

            url = f"{NITRIDING_BASE_URL}/enclave/hash"
            req = urllib.request.Request(
                url,
                data=key_hash_b64.encode('utf-8'),
                method='POST'
            )

            response = urllib.request.urlopen(req, timeout=5)
            response_body = response.read().decode('utf-8')

            if response.getcode() == 200:
                logger.info("Successfully registered public key hash with nitriding")
                logger.info(f"Response: {response_body}")
                return True
            else:
                logger.error(f"Failed to register public key hash: HTTP {response.getcode()}")
                return False

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else "No error body"
            logger.error(f"HTTP Error {e.code}: {e.reason} - {error_body}")
            return False
        except Exception as e:
            logger.warning(f"Could not register with nitriding (may not be in TEE): {e}")
            return False

    def sign_data(self, data: str) -> str:
        """Sign data with private key and return base64 signature"""
        data_bytes = data.encode('utf-8')
        signature = self.private_key.sign(
            data_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    def get_public_key(self) -> str:
        """Return public key in PEM format"""
        return self.public_key_pem

    def get_attestation_document(self) -> dict:
        """Return TEE attestation document"""
        return {
            "public_key": self.public_key_pem,
            "timestamp": datetime.now(UTC).isoformat(),
            "enclave_info": {
                "platform": "aws-nitro",
                "instance_type": "tee-enabled",
                "version": "1.0.0"
            },
            "measurements": None  # Would contain PCR values in real deployment
        }


def signal_ready():
    """Signal to nitriding that enclave is ready to accept traffic"""
    try:
        url = f"{NITRIDING_BASE_URL}/enclave/ready"
        r = urllib.request.urlopen(url)
        if r.getcode() != 200:
            raise Exception(f"Expected status code 200 but got {r.getcode()}")
        logger.info("Successfully signaled ready to nitriding")
    except Exception as e:
        logger.warning(f"Could not signal nitriding (may not be in TEE): {e}")


def compute_request_hash(request_data: dict) -> str:
    """Compute SHA256 hash of request data"""
    request_json = json.dumps(request_data, sort_keys=True)
    return hashlib.sha256(request_json.encode('utf-8')).hexdigest()


# Singleton instance - initialized lazily or eagerly depending on environment
_tee_keys: TEEKeyManager | None = None


def get_tee_keys() -> TEEKeyManager:
    """Get or create the singleton TEE key manager"""
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
    logger.info(f"TEE initialized. Public key (first 80 chars): {keys.get_public_key()[:80]}...")
    return keys
