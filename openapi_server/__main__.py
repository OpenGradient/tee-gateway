"""
OpenAI-compatible API server with TEE integration and x402v2 payment middleware.
Runs inside a Nitro Enclave, proxied by nitriding on port 443.
"""

import logging
import sys
import os
import json
from datetime import datetime, UTC
import gc
import time
import psutil

import connexion
from flask import jsonify
from openapi_server import encoder
from openapi_server.tee_manager import initialize_tee, get_tee_keys

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
    """Return TEE attestation document with public key."""
    try:
        tee_keys = get_tee_keys()
        return jsonify(tee_keys.get_attestation_document())
    except Exception as e:
        logger.error(f"Attestation error: {e}")
        return {"error": str(e)}, 500



def create_app():
    app = connexion.App(__name__, specification_dir="./openapi/")
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api("openapi.yaml",
                arguments={"title": "OpenAI API"},
                pythonic_params=True)

    app.app.add_url_rule("/health", "health", health, methods=["GET"])
    app.app.add_url_rule("/signing-key", "signing-key", signing_key, methods=["GET"])

    # Initialize TEE here so it runs under both Gunicorn and direct execution
    try:
        initialize_tee()
        logger.info("TEE initialized successfully")
    except Exception as e:
        logger.warning(f"TEE initialization failed (may not be in enclave): {e}")

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
    logger.info(f"Server ready on {host}:{port}")
    application.run(host=host, port=port)

