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

Enabled via environment variables:
    HEARTBEAT_CONTRACT_ADDRESS    — TEERegistry contract address
    HEARTBEAT_FACILITATOR_URL     — Facilitator base URL (must expose POST /heartbeat)
    TEE_HEARTBEAT_INTERVAL        — Seconds between pings (default 900 = 15 min)
    HEARTBEAT_FACILITATOR_TIMEOUT — HTTP timeout seconds (default 20)

Also requires the TEEKeyManager (tee_keys) for RSA signing.
"""

import logging
import base64
import time
import threading
from typing import Optional

from eth_hash.auto import keccak
from cryptography.hazmat.primitives import serialization
import httpx

from tee_gateway.config import HeartbeatConfig

logger = logging.getLogger("heartbeat")

MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds


class HeartbeatService:
    """Sends TEE-signed heartbeats to the TEERegistry contract via a daemon thread."""

    def __init__(
        self,
        contract_address: str,
        facilitator_url: str,
        tee_keys,
        interval: int = DEFAULT_HEARTBEAT_INTERVAL,
        facilitator_timeout: int = DEFAULT_FACILITATOR_TIMEOUT,
    ):
        self.interval = interval
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
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sign_heartbeat(self, timestamp: int) -> bytes:
        """Sign keccak256(teeId || timestamp) with the TEE's RSA key.

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

        response = httpx.post(
            self.heartbeat_endpoint,
            json=payload,
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
            raise Exception(
                f"Facilitator heartbeat relay failed ({response.status_code}): {error_msg}"
            )

        tx_hash = (
            response_body.get("txHash") if isinstance(response_body, dict) else None
        )
        if not tx_hash or not isinstance(tx_hash, str):
            raise Exception("Facilitator heartbeat relay response missing txHash")

        return tx_hash

    def _send_heartbeat(self) -> bool:
        """Sign and relay heartbeat transaction via facilitator. Returns True on success."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                timestamp = int(time.time())
                signature = self._sign_heartbeat(timestamp)
                tx_hash = self._relay_heartbeat(timestamp, signature)

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
                    if self._stop_event.wait(timeout=RETRY_DELAY):
                        return False

        logger.error("Heartbeat failed after %d attempts", MAX_RETRIES)
        return False

    def _run_loop(self):
        """Main heartbeat loop. Runs until stop event is set."""
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
            while not self._stop_event.is_set():
                self._send_heartbeat()
                self._stop_event.wait(timeout=self.interval)
        except Exception as e:
            logger.error("Heartbeat loop crashed: %s", e, exc_info=True)
        finally:
            self._running = False
            logger.info("Heartbeat loop stopped")

    def start(self):
        """Start the heartbeat background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Heartbeat already running, skipping duplicate start")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
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


def create_heartbeat_service(
    tee_keys, config: Optional["HeartbeatConfig"]
) -> Optional["HeartbeatService"]:
    """Create a HeartbeatService from a HeartbeatConfig + TEE keys, or None if not configured."""
    if config is None:
        logger.info("Heartbeat disabled (no HeartbeatConfig provided)")
        return None

    return HeartbeatService(
        contract_address=config.contract_address,
        facilitator_url=config.facilitator_url,
        tee_keys=tee_keys,
        interval=config.interval,
        facilitator_timeout=config.facilitator_timeout,
    )
