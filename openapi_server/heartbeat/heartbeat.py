"""
Blockchain heartbeat service for the TEERegistry contract.

Periodically calls TEERegistry.heartbeat(teeId, timestamp, signature) to
prove this TEE node is alive. The registry verifies the RSA-PSS signature
via the precompile (0x0900) using the public key stored at registration.

The TEE must already be registered via registerTEEWithAttestation() before
the heartbeat loop starts.

Signed message:  keccak256(abi.encodePacked(teeId, timestamp))
TEE ID:          keccak256(publicKeyDER)

Enabled via environment variables:
    HEARTBEAT_RPC_URL          — JSON-RPC endpoint
    HEARTBEAT_CONTRACT_ADDRESS — TEERegistry contract address
    HEARTBEAT_PRIVATE_KEY      — Wallet private key for sending txs (0x-prefixed)
    TEE_HEARTBEAT_INTERVAL     — Seconds between pings (default 900 = 15 min)

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
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

logger = logging.getLogger("heartbeat")

# Minimal ABI — only the registry functions we call
REGISTRY_ABI = [
    {
        "inputs": [
            {"name": "teeId", "type": "bytes32"},
            {"name": "timestamp", "type": "uint256"},
            {"name": "signature", "type": "bytes"},
        ],
        "name": "heartbeat",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

TEE_HEARTBEAT_INTERVAL = int(os.getenv("TEE_HEARTBEAT_INTERVAL", "900"))
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds


class HeartbeatService:
    """Sends TEE-signed heartbeats to the TEERegistry contract."""

    def __init__(
        self,
        rpc_url: str,
        contract_address: str,
        private_key: str,
        tee_keys,
        interval: int = TEE_HEARTBEAT_INTERVAL,
    ):
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self.tee_keys = tee_keys

        # Derive TEE identity from the RSA public key
        self.public_key_der = tee_keys.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.tee_id = keccak(self.public_key_der)  # bytes32

        # Web3 setup — wallet key is only used for sending txs (gas)
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        self.account = self.w3.eth.account.from_key(private_key)
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=REGISTRY_ABI,
        )

        # Status tracking
        self.last_success: Optional[float] = None
        self.last_error: Optional[str] = None
        self.total_sent: int = 0
        self.total_errors: int = 0
        self._running: bool = False

    def _sign_heartbeat(self, timestamp: int) -> bytes:
        """Sign keccak256(teeId ‖ timestamp) with the TEE's RSA key.

        Returns raw signature bytes (not base64).
        Uses the same RSA-PSS-SHA256 path as tee_keys.sign_data().
        """
        msg_hash = keccak(self.tee_id + timestamp.to_bytes(32, "big"))
        sig_b64 = self.tee_keys.sign_data(msg_hash)
        return base64.b64decode(sig_b64)

    async def _send_heartbeat(self) -> bool:
        """Build, sign, and send a heartbeat transaction. Returns True on success."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                timestamp = int(time.time())
                signature = self._sign_heartbeat(timestamp)

                nonce = await asyncio.to_thread(
                    self.w3.eth.get_transaction_count, self.account.address
                )
                gas_price = await asyncio.to_thread(lambda: self.w3.eth.gas_price)

                tx = self.contract.functions.heartbeat(
                    self.tee_id,
                    timestamp,
                    signature,
                ).build_transaction(
                    {
                        "from": self.account.address,
                        "nonce": nonce,
                        "gas": 300_000,
                        "gasPrice": gas_price,
                    }
                )

                signed_tx = self.account.sign_transaction(tx)
                tx_hash = await asyncio.to_thread(
                    self.w3.eth.send_raw_transaction, signed_tx.raw_transaction
                )
                receipt = await asyncio.to_thread(
                    self.w3.eth.wait_for_transaction_receipt, tx_hash, timeout=60
                )

                if receipt.status == 1:
                    self.last_success = time.time()
                    self.total_sent += 1
                    self.last_error = None
                    logger.info(
                        "Heartbeat tx=%s count=%d", tx_hash.hex(), self.total_sent
                    )
                    return True

                raise Exception(f"Transaction reverted (tx={tx_hash.hex()})")

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
            "Heartbeat started (teeId=%s registry=%s interval=%ds wallet=%s)",
            self.tee_id.hex(),
            self.contract_address,
            self.interval,
            self.account.address,
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
            "wallet": self.account.address,
            "interval_seconds": self.interval,
            "total_sent": self.total_sent,
            "total_errors": self.total_errors,
            "last_success_timestamp": self.last_success,
            "last_error": self.last_error,
        }


def create_heartbeat_service(tee_keys) -> Optional["HeartbeatService"]:
    """Create a HeartbeatService from env vars + TEE keys, or None if not configured."""
    rpc_url = os.getenv("HEARTBEAT_RPC_URL")
    contract_address = os.getenv("HEARTBEAT_CONTRACT_ADDRESS")
    private_key = os.getenv("HEARTBEAT_PRIVATE_KEY")

    if not all([rpc_url, contract_address, private_key]):
        logger.info(
            "Heartbeat disabled (set HEARTBEAT_RPC_URL, "
            "HEARTBEAT_CONTRACT_ADDRESS, HEARTBEAT_PRIVATE_KEY to enable)"
        )
        return None

    return HeartbeatService(
        rpc_url=rpc_url,
        contract_address=contract_address,
        private_key=private_key,
        tee_keys=tee_keys,
        interval=TEE_HEARTBEAT_INTERVAL,
    )
