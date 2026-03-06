"""
OpenAI-compatible API server with TEE integration and x402v2 payment middleware.
Runs inside a Nitro Enclave, proxied by nitriding on port 443.
"""

import logging
import sys
import os
import json
import gc
import time
import threading
import asyncio
import atexit
import psutil

import connexion
from flask import jsonify, request
from openapi_server import encoder
from openapi_server.tee_manager import initialize_tee, get_tee_keys
from openapi_server.llm_backend import reinitialize_http_clients
from openapi_server.heartbeat import create_heartbeat_service

from x402v2.http import FacilitatorConfig, HTTPFacilitatorClientSync, PaymentOption
from x402v2.http.middleware.flask import payment_middleware
from x402v2.http.types import RouteConfig
from x402v2.mechanisms.evm.exact import ExactEvmServerScheme
from x402v2.mechanisms.evm.upto import UptoEvmServerScheme
from x402v2.schemas import AssetAmount, Network
from x402v2.server import x402ResourceServerSync
from x402v2.session import SessionStore
import x402v2.http.middleware.flask as x402_flask
from .util import BASE_OPG_ADDRESS, USDC_ADDRESS, dynamic_session_cost_calculator

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
# Heartbeat background service (asyncio loop in a daemon thread)
# ---------------------------------------------------------------------------
_heartbeat_service = None
_heartbeat_loop = None


def _start_heartbeat_loop(loop: asyncio.AbstractEventLoop):
    """Run an asyncio event loop in a background thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _init_heartbeat():
    """Create and start the heartbeat service if env vars are configured."""
    global _heartbeat_service, _heartbeat_loop

    tee_keys = get_tee_keys()
    _heartbeat_service = create_heartbeat_service(tee_keys)
    if _heartbeat_service is None:
        return

    _heartbeat_loop = asyncio.new_event_loop()
    t = threading.Thread(target=_start_heartbeat_loop, args=(_heartbeat_loop,), daemon=True)
    t.start()

    asyncio.run_coroutine_threadsafe(_async_start_heartbeat(), _heartbeat_loop)
    logger.info("Heartbeat service scheduled on background event loop")


async def _async_start_heartbeat():
    """Start the heartbeat task inside the background event loop."""
    _heartbeat_service.start()


def _shutdown_heartbeat():
    """Stop the heartbeat service and background loop on process exit."""
    global _heartbeat_service, _heartbeat_loop
    if _heartbeat_service and _heartbeat_loop and _heartbeat_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_heartbeat_service.stop(), _heartbeat_loop)
        try:
            future.result(timeout=10)
        except Exception:
            pass
        _heartbeat_loop.call_soon_threadsafe(_heartbeat_loop.stop)
    logger.info("Heartbeat shutdown complete")


atexit.register(_shutdown_heartbeat)


EVM_NETWORK: Network = "eip155:10740"
BASE_TESTNET_NETWORK: Network = "eip155:84532"
EVM_PAYMENT_ADDRESS = "0x40eFb45552EDfB2502D90A657a8ab41F03ec460d"
FACILITATOR_URL = os.getenv("FACILITATOR_URL", "https://facilitator.memchat.io")

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
                    amount="100000",  # $0.01 USDC
                    asset=USDC_ADDRESS,
                    extra={"name": "OUSDC", "version": "2", "assetTransferMethod": "permit2"},
                ),
                network=EVM_NETWORK,
            ),
            PaymentOption(
                scheme="upto",
                pay_to=EVM_PAYMENT_ADDRESS,
                price=AssetAmount(
                    amount="50000000000000000",  # 0.05 OPG
                    asset=BASE_OPG_ADDRESS,
                    extra={"name": "OPG", "version": "2", "assetTransferMethod": "permit2"},
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
                    amount="10000",  # $0.01 USDC
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
            return jsonify({"error": "Provider keys have already been initialized"}), 409

        body = request.get_json(silent=True)
        if not body:
            return jsonify({"error": "JSON body required"}), 400

        # Inject keys directly into the environment
        if body.get('openai_api_key'):
            os.environ["OPENAI_API_KEY"] = body['openai_api_key']
        if body.get('google_api_key'):
            os.environ["GOOGLE_API_KEY"] = body['google_api_key']
        if body.get('anthropic_api_key'):
            os.environ["ANTHROPIC_API_KEY"] = body['anthropic_api_key']
        if body.get('xai_api_key'):
            os.environ["XAI_API_KEY"] = body['xai_api_key']

        def _key_status(env_var: str) -> str:
            val = os.environ.get(env_var, "")
            if not val:
                return "NOT SET"
            return f"set ({val[:6]}...{val[-4:]})"

        logger.info("ENV check after injection:")
        logger.info("  OPENAI_API_KEY    : %s", _key_status("OPENAI_API_KEY"))
        logger.info("  GOOGLE_API_KEY    : %s", _key_status("GOOGLE_API_KEY"))
        logger.info("  ANTHROPIC_API_KEY : %s", _key_status("ANTHROPIC_API_KEY"))
        logger.info("  XAI_API_KEY       : %s", _key_status("XAI_API_KEY"))

        # Rebuild HTTP clients with the new Authorization headers and clear
        # the model cache so subsequent requests use fresh instances.
        reinitialize_http_clients()

        _keys_initialized = True

    providers_set = [
        p for p, k in {
            "openai": body.get('openai_api_key'),
            "google": body.get('google_api_key'),
            "anthropic": body.get('anthropic_api_key'),
            "xai": body.get('xai_api_key'),
        }.items() if k
    ]
    logger.info("Provider API keys initialized for: %s", ", ".join(providers_set))
    return jsonify({"status": "ok", "providers_initialized": providers_set}), 200


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
    app.add_api("openapi.yaml",
                arguments={"title": "OpenAI API"},
                pythonic_params=True)

    app.app.add_url_rule("/health", "health", health, methods=["GET"])
    app.app.add_url_rule("/signing-key", "signing-key", signing_key, methods=["GET"])
    app.app.add_url_rule("/v1/keys", "set-provider-keys", set_provider_keys, methods=["POST"])
    app.app.add_url_rule("/heartbeat/status", "heartbeat-status", heartbeat_status, methods=["GET"])

    # Initialize TEE here so it runs under both Gunicorn and direct execution.
    # This is the single TEEKeyManager instance — the same key both registers
    # with nitriding and signs all LLM responses.
    try:
        initialize_tee()
        logger.info("TEE initialized successfully")
    except Exception as e:
        logger.warning(f"TEE initialization failed (may not be in enclave): {e}")

    # Start blockchain heartbeat service (if configured via env vars).
    # Runs in a background asyncio loop on a daemon thread.
    try:
        _init_heartbeat()
    except Exception as e:
        logger.warning(f"Heartbeat initialization failed: {e}")

    return app.app


# Create the WSGI application and attach x402v2 payment middleware
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
