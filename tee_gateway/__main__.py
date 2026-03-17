"""
OpenAI-compatible API server with TEE integration and x402v2 payment middleware.
Runs inside a Nitro Enclave, proxied by nitriding on port 443.
"""

import logging
import sys
import os
import gc
import time
import threading
import atexit
import psutil

import connexion
from flask import jsonify, request
from tee_gateway import encoder
from tee_gateway.tee_manager import initialize_tee, get_tee_keys
from tee_gateway.config import (
    HeartbeatConfig,
    ProviderConfig,
    DEFAULT_HEARTBEAT_INTERVAL,
)
from tee_gateway.llm_backend import set_provider_config
from tee_gateway.heartbeat import create_heartbeat_service

from x402v2.http import FacilitatorConfig, HTTPFacilitatorClientSync, PaymentOption
from x402v2.http.middleware.flask import payment_middleware
from x402v2.http.types import RouteConfig
from x402v2.mechanisms.evm.exact import ExactEvmServerScheme
from x402v2.mechanisms.evm.upto import UptoEvmServerScheme
from x402v2.schemas import AssetAmount
from x402v2.server import x402ResourceServerSync
from x402v2.session import SessionStore
import x402v2.http.middleware.flask as x402_flask

from .util import dynamic_session_cost_calculator
from .definitions import (
    EVM_NETWORK,
    BASE_TESTNET_NETWORK,
    EVM_PAYMENT_ADDRESS,
    USDC_ADDRESS,
    BASE_OPG_ADDRESS,
    CHAT_COMPLETIONS_USDC_AMOUNT,
    CHAT_COMPLETIONS_OPG_AMOUNT,
    COMPLETIONS_USDC_AMOUNT,
    FACILITATOR_URL,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("x402.middleware").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heartbeat background service
# ---------------------------------------------------------------------------
_heartbeat_service = None


def _init_heartbeat(heartbeat_config: HeartbeatConfig | None):
    """Create and start the heartbeat service if a HeartbeatConfig is provided."""
    global _heartbeat_service

    if _heartbeat_service is not None:
        logger.info("Heartbeat already initialized, skipping")
        return

    tee_keys = get_tee_keys()
    _heartbeat_service = create_heartbeat_service(tee_keys, heartbeat_config)
    if _heartbeat_service is None:
        return

    _heartbeat_service.start()


def _shutdown_heartbeat():
    """Stop the heartbeat service on process exit."""
    if _heartbeat_service is not None:
        _heartbeat_service.stop()


atexit.register(_shutdown_heartbeat)

facilitator = HTTPFacilitatorClientSync(FacilitatorConfig(url=FACILITATOR_URL))
server = x402ResourceServerSync(facilitator)
store = SessionStore()

server.register(EVM_NETWORK, ExactEvmServerScheme())
server.register(BASE_TESTNET_NETWORK, ExactEvmServerScheme())
server.register(EVM_NETWORK, UptoEvmServerScheme())
server.register(BASE_TESTNET_NETWORK, UptoEvmServerScheme())

routes = {
    "POST /v1/chat/completions": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="upto",
                pay_to=EVM_PAYMENT_ADDRESS,
                price=AssetAmount(
                    amount=CHAT_COMPLETIONS_USDC_AMOUNT,
                    asset=USDC_ADDRESS,
                    extra={
                        "name": "OUSDC",
                        "version": "2",
                        "assetTransferMethod": "permit2",
                    },
                ),
                network=EVM_NETWORK,
            ),
            PaymentOption(
                scheme="upto",
                pay_to=EVM_PAYMENT_ADDRESS,
                price=AssetAmount(
                    amount=CHAT_COMPLETIONS_OPG_AMOUNT,
                    asset=BASE_OPG_ADDRESS,
                    extra={
                        "name": "OPG",
                        "version": "2",
                        "assetTransferMethod": "permit2",
                    },
                ),
                network=BASE_TESTNET_NETWORK,
            ),
        ],
        mime_type="application/json",
        description="Chat completion",
    ),
    "POST /v1/completions": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="upto",
                pay_to=EVM_PAYMENT_ADDRESS,
                price=AssetAmount(
                    amount=COMPLETIONS_USDC_AMOUNT,
                    asset=USDC_ADDRESS,
                    extra={"name": "USDC", "version": "2"},
                ),
                network=EVM_NETWORK,
            ),
        ],
        mime_type="application/json",
        description="Completion",
    ),
}


# ---------------------------------------------------------------------------
# One-time provider key injection
# ---------------------------------------------------------------------------
_keys_initialized: bool = False
_keys_lock = threading.Lock()


def set_provider_keys():
    """
    POST /v1/keys — inject LLM provider API keys into the enclave.
    Can only be called once; subsequent calls return HTTP 409.
    """
    global _keys_initialized

    with _keys_lock:
        if _keys_initialized:
            return jsonify(
                {"error": "Provider keys have already been initialized"}
            ), 409

        body = request.get_json(silent=True)
        if not body:
            return jsonify({"error": "JSON body required"}), 400

        # Build provider config from request body
        provider_config = ProviderConfig(
            openai_api_key=body.get("openai_api_key") or None,
            anthropic_api_key=body.get("anthropic_api_key") or None,
            google_api_key=body.get("google_api_key") or None,
            xai_api_key=body.get("xai_api_key") or None,
        )
        set_provider_config(provider_config)

        # Build heartbeat config from request body (optional)
        contract_address = body.get("heartbeat_contract_address")
        facilitator_url = (
            body.get("heartbeat_facilitator_url")
            or os.getenv("FACILITATOR_URL")
            or FACILITATOR_URL
        )
        heartbeat_config: HeartbeatConfig | None = None
        if contract_address and facilitator_url:
            interval_raw = body.get(
                "tee_heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL
            )
            try:
                interval = int(interval_raw)
            except (ValueError, TypeError):
                logger.error(
                    "Invalid tee_heartbeat_interval %r; falling back to default %s",
                    interval_raw,
                    DEFAULT_HEARTBEAT_INTERVAL,
                )
                interval = DEFAULT_HEARTBEAT_INTERVAL
            heartbeat_config = HeartbeatConfig(
                contract_address=contract_address,
                facilitator_url=facilitator_url,
                interval=interval,
            )

        def _set(val: str | None) -> str:
            return "set" if val else "NOT SET"

        logger.info("Config check after injection:")
        logger.info(
            "  openai_api_key              : %s", _set(provider_config.openai_api_key)
        )
        logger.info(
            "  google_api_key              : %s", _set(provider_config.google_api_key)
        )
        logger.info(
            "  anthropic_api_key           : %s",
            _set(provider_config.anthropic_api_key),
        )
        logger.info(
            "  xai_api_key                 : %s", _set(provider_config.xai_api_key)
        )
        logger.info(
            "  heartbeat_contract_address  : %s",
            _set(heartbeat_config.contract_address if heartbeat_config else None),
        )
        logger.info(
            "  heartbeat_facilitator_url   : %s",
            _set(heartbeat_config.facilitator_url if heartbeat_config else None),
        )
        logger.info(
            "  tee_heartbeat_interval      : %s",
            heartbeat_config.interval if heartbeat_config else "900 (default)",
        )
        logger.info(
            "  HEARTBEAT_WALLET (TEE-gen)  : %s", get_tee_keys().get_wallet_address()
        )

        # Start heartbeat service if configured
        try:
            _init_heartbeat(heartbeat_config)
        except Exception as e:
            logger.warning(f"Heartbeat initialization failed: {e}")

        _keys_initialized = True

    providers_set = [
        p
        for p, k in {
            "openai": provider_config.openai_api_key,
            "google": provider_config.google_api_key,
            "anthropic": provider_config.anthropic_api_key,
            "xai": provider_config.xai_api_key,
        }.items()
        if k
    ]

    return jsonify(
        {
            "status": "ok",
            "providers_initialized": providers_set,
            "heartbeat_enabled": heartbeat_config is not None,
        }
    ), 200


def health():
    process = psutil.Process()
    system_memory = psutil.virtual_memory()

    connections = process.connections()
    conn_states = {}
    for conn in connections:
        state = conn.status if conn.status else "NONE"
        conn_states[state] = conn_states.get(state, 0) + 1

    return {
        "status": "OK",
        "version": "1.0.0",
        "tee_enabled": True,
        "uptime_seconds": time.time() - process.create_time(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "process_memory_mb": process.memory_info().rss / 1024 / 1024,
        "process_memory_percent": process.memory_percent(),
        "system_total_memory_mb": system_memory.total / 1024 / 1024,
        "system_used_memory_mb": system_memory.used / 1024 / 1024,
        "system_available_memory_mb": system_memory.available / 1024 / 1024,
        "system_memory_percent": system_memory.percent,
        "threads": process.num_threads(),
        "open_files": len(process.open_files()),
        "num_fds": process.num_fds(),
        "connections": len(connections),
        "connection_states": conn_states,
        "gc_objects": len(gc.get_objects()),
    }, 200


def signing_key():
    """Return TEE attestation document with public key and tee_id."""
    try:
        tee_keys = get_tee_keys()
        return jsonify(tee_keys.get_attestation_document())
    except Exception as e:
        logger.error(f"Attestation error: {e}")
        return {"error": str(e)}, 500


def heartbeat_status():
    """GET /heartbeat/status — return heartbeat service status."""
    if _heartbeat_service is None:
        return jsonify({"enabled": False}), 200
    return jsonify({"enabled": True, **_heartbeat_service.status()}), 200


def create_app():
    app = connexion.App(__name__, specification_dir="./openapi/")
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api("openapi.yaml", arguments={"title": "OpenAI API"}, pythonic_params=True)

    app.app.add_url_rule("/health", "health", health, methods=["GET"])
    app.app.add_url_rule("/signing-key", "signing-key", signing_key, methods=["GET"])
    app.app.add_url_rule(
        "/v1/keys", "set-provider-keys", set_provider_keys, methods=["POST"]
    )
    app.app.add_url_rule(
        "/heartbeat/status", "heartbeat-status", heartbeat_status, methods=["GET"]
    )

    # Initialize TEE here so it runs under both Gunicorn and direct execution.
    # This is the single TEEKeyManager instance — the same key both registers
    # with nitriding and signs all LLM responses.
    try:
        initialize_tee()
        logger.info("TEE initialized successfully")
    except Exception as e:
        logger.warning(f"TEE initialization failed (may not be in enclave): {e}")

    return app.app


# ---------------------------------------------------------------------------
# WSGI application + x402v2 payment middleware
# ---------------------------------------------------------------------------

# Create the WSGI application
application = create_app()

# This patch ensures that non-payment 0-length requests can still bypass the middleware
_original_read_body_bytes = x402_flask._read_body_bytes


def _patched_read_body_bytes(environ):
    try:
        content_length = int(environ.get("CONTENT_LENGTH") or 0)
    except (ValueError, TypeError):
        content_length = 0

    if content_length <= 0:
        return b""

    return _original_read_body_bytes(environ)


x402_flask._read_body_bytes = _patched_read_body_bytes

payment_middleware(
    application,
    routes=routes,
    server=server,
    session_store=store,
    cost_per_request=100000000000000,  # static precheck/fallback estimate
    session_idle_timeout=100,
    session_cost_calculator=dynamic_session_cost_calculator,
)
logger.info("x402v2 payment middleware initialized")

if __name__ == "__main__":
    port = int(os.getenv("API_SERVER_PORT", "8000"))
    host = os.getenv("API_SERVER_HOST", "0.0.0.0")
    logger.info(f"Starting OpenAI-compatible API server on {host}:{port}")
    application.run(host=host, port=port)
