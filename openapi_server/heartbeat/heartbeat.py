"""
Blockchain heartbeat service for the TEERegistry contract.

Periodically signs a heartbeat payload and relays it to the facilitator
`POST /heartbeat` endpoint. The facilitator submits
TEERegistry.heartbeat(teeId, timestamp, signature) on-chain using a funded
relayer wallet, so each TEE wallet does not need gas.

The TEE must already be registered via registerTEEWithAttestation() before
the heartbeat loop starts.

Signed message:  keccak256(abi.encodePacked(teeId, timestamp))
TEE ID:          keccak256(publicKeyDER)

The TEE wallet private key is still generated inside the enclave and never
leaves it. It is retained for identity continuity but is no longer required
to pay gas for heartbeat transactions.

Enabled via environment variables:
    HEARTBEAT_CONTRACT_ADDRESS — TEERegistry contract address
    HEARTBEAT_FACILITATOR_URL  — Facilitator base URL (must expose POST /heartbeat)
    TEE_HEARTBEAT_INTERVAL     — Seconds between pings (default 900 = 15 min)
    HEARTBEAT_FACILITATOR_TIMEOUT — HTTP timeout seconds (default 20)

Also requires the TEEKeyManager (tee_keys) from server.py for RSA signing.
"""

import os
import logging
import asyncio
import base64
import time
from typing import Optional

from eth_hash.auto import keccak
from cryptography.hazmat.primitives import serialization
import requests

logger = logging.getLogger("heartbeat")

DEFAULT_HEARTBEAT_INTERVAL = 900  # 15 minutes
DEFAULT_FACILITATOR_TIMEOUT = 20  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds


class HeartbeatService:
    """Sends TEE-signed heartbeats to the TEERegistry contract."""

    def __init__(
        self,
        contract_address: str,
        facilitator_url: str,
        tee_keys,
        interval: int = DEFAULT_HEARTBEAT_INTERVAL,
        facilitator_timeout: int = DEFAULT_FACILITATOR_TIMEOUT,
    ):
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self.tee_keys = tee_keys
        self.facilitator_timeout = facilitator_timeout

        # Derive TEE identity from the RSA public key
        self.public_key_der = tee_keys.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.tee_id = keccak(self.public_key_der)  # bytes32

        self.contract_address = contract_address
        self.facilitator_url = facilitator_url.rstrip("/")
        self.heartbeat_endpoint = f"{self.facilitator_url}/heartbeat"
        self.wallet_address = tee_keys.get_wallet_address()

        # Status tracking
        self.last_success: Optional[float] = None
        self.last_error: Optional[str] = None
        self.total_sent: int = 0
        self.total_errors: int = 0
        self.last_tx_hash: Optional[str] = None
        self._running: bool = False

    def _sign_heartbeat(self, timestamp: int) -> bytes:
        """Sign keccak256(teeId ‖ timestamp) with the TEE's RSA key.

        Returns raw signature bytes (not base64).
        Uses the same RSA-PSS-SHA256 path as tee_keys.sign_data().
        """
        msg_hash = keccak(self.tee_id + timestamp.to_bytes(32, "big"))
        sig_b64 = self.tee_keys.sign_data(msg_hash)
        return base64.b64decode(sig_b64)

    def _relay_heartbeat(self, timestamp: int, signature: bytes) -> str:
        """Relay a signed heartbeat to facilitator and return the tx hash."""
        payload = {
            "teeId": "0x" + self.tee_id.hex(),
            "timestamp": str(timestamp),
            "signature": base64.b64encode(signature).decode("ascii"),
            "contractAddress": self.contract_address,
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            self.heartbeat_endpoint,
            json=payload,
            headers=headers,
            timeout=self.facilitator_timeout,
        )

        response_body: Optional[dict] = None
        try:
            response_body = response.json()
        except ValueError:
            response_body = None

        if response.status_code >= 400:
            error_msg = (
                response_body.get("error")
                if isinstance(response_body, dict)
                else response.text[:500]
            )
            raise Exception(f"Facilitator heartbeat relay failed ({response.status_code}): {error_msg}")

        tx_hash = response_body.get("txHash") if isinstance(response_body, dict) else None
        if not tx_hash or not isinstance(tx_hash, str):
            raise Exception("Facilitator heartbeat relay response missing txHash")

        return tx_hash

    async def _send_heartbeat(self) -> bool:
        """Sign and relay heartbeat transaction via facilitator. Returns True on success."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                timestamp = int(time.time())
                signature = self._sign_heartbeat(timestamp)
                tx_hash = await asyncio.to_thread(self._relay_heartbeat, timestamp, signature)

                self.last_success = time.time()
                self.total_sent += 1
                self.last_error = None
                self.last_tx_hash = tx_hash
                logger.info(
                    "Heartbeat relayed tx=%s count=%d", tx_hash, self.total_sent
                )
                return True

            except Exception as e:
                self.last_error = str(e)
                self.total_errors += 1
                logger.warning(
                    "Heartbeat attempt %d/%d failed: %s", attempt, MAX_RETRIES, e
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)

        logger.error("Heartbeat failed after %d attempts", MAX_RETRIES)
        return False

    async def _run_loop(self):
        """Main heartbeat loop. Runs until cancelled."""
        logger.info(
            "Heartbeat started (teeId=%s registry=%s interval=%ds teeWallet=%s relay=%s)",
            self.tee_id.hex(),
            self.contract_address,
            self.interval,
            self.wallet_address,
            self.heartbeat_endpoint,
        )
        self._running = True
        try:
            while True:
                await self._send_heartbeat()
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            logger.info("Heartbeat loop cancelled")
        except Exception as e:
            logger.error("Heartbeat loop crashed: %s", e, exc_info=True)
        finally:
            self._running = False

    def start(self) -> asyncio.Task:
        """Start the heartbeat background task."""
        self._task = asyncio.create_task(self._run_loop())
        return self._task

    async def stop(self):
        """Cancel the background task and wait for clean shutdown."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._running = False
        logger.info("Heartbeat service stopped")

    def status(self) -> dict:
        """Return current service status."""
        return {
            "running": self._running,
            "tee_id": "0x" + self.tee_id.hex(),
            "registry": self.contract_address,
            "wallet": self.wallet_address,
            "relay_mode": "facilitator",
            "facilitator_url": self.facilitator_url,
            "heartbeat_endpoint": self.heartbeat_endpoint,
            "interval_seconds": self.interval,
            "total_sent": self.total_sent,
            "total_errors": self.total_errors,
            "last_success_timestamp": self.last_success,
            "last_tx_hash": self.last_tx_hash,
            "last_error": self.last_error,
        }


def create_heartbeat_service(tee_keys) -> Optional["HeartbeatService"]:
    """Create a HeartbeatService from env vars + TEE keys, or None if not configured.

    Env vars are read at call time (not import time) because they are
    injected into the enclave at runtime via POST /v1/keys.
    """
    contract_address = os.getenv("HEARTBEAT_CONTRACT_ADDRESS")
    facilitator_url = os.getenv("HEARTBEAT_FACILITATOR_URL") or os.getenv("FACILITATOR_URL")

    if not all([contract_address, facilitator_url]):
        logger.info(
            "Heartbeat disabled (set HEARTBEAT_CONTRACT_ADDRESS and "
            "HEARTBEAT_FACILITATOR_URL or FACILITATOR_URL to enable)"
        )
        return None

    # Parse interval only after confirming heartbeat is enabled
    try:
        interval = int(os.getenv("TEE_HEARTBEAT_INTERVAL", str(DEFAULT_HEARTBEAT_INTERVAL)))
    except (ValueError, TypeError):
        logger.warning(
            "Invalid TEE_HEARTBEAT_INTERVAL=%r, falling back to default %ds",
            os.getenv("TEE_HEARTBEAT_INTERVAL"),
            DEFAULT_HEARTBEAT_INTERVAL,
        )
        interval = DEFAULT_HEARTBEAT_INTERVAL

    try:
        facilitator_timeout = int(
            os.getenv("HEARTBEAT_FACILITATOR_TIMEOUT", str(DEFAULT_FACILITATOR_TIMEOUT))
        )
    except (ValueError, TypeError):
        logger.warning(
            "Invalid HEARTBEAT_FACILITATOR_TIMEOUT=%r, falling back to default %ds",
            os.getenv("HEARTBEAT_FACILITATOR_TIMEOUT"),
            DEFAULT_FACILITATOR_TIMEOUT,
        )
        facilitator_timeout = DEFAULT_FACILITATOR_TIMEOUT

    return HeartbeatService(
        contract_address=contract_address,
        facilitator_url=facilitator_url,
        tee_keys=tee_keys,
        interval=interval,
        facilitator_timeout=facilitator_timeout,
    )
