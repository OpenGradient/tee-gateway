"""
OpenAI-compatible API server with TEE integration and x402v2 payment middleware.
Runs inside a Nitro Enclave, proxied by nitriding on port 443.
"""

import logging
import sys
import os
import json
from datetime import datetime, UTC

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
USDC_ADDRESS = "0x094E464A23B90A71a0894D5D1e5D470FfDD074e1"
BASE_OPG_ADDRESS = "0x240b09731D96979f50B2C649C9CE10FcF9C7987F"
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
    return {"status": "OK", "tee_enabled": True}, 200


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
payment_middleware(
    application,
    routes=routes,
    server=server,
    session_store=store,
    cost_per_request=100000000000000,
    session_idle_timeout=100,
)
logger.info("x402v2 payment middleware initialized")

# --- DEBUG: wrap AFTER payment_middleware so we intercept first ---
_payment_wsgi = application.wsgi_app

def _debug_wsgi(environ, start_response):
    method = environ.get("REQUEST_METHOD", "")
    path = environ.get("PATH_INFO", "")
    headers = {
        k[5:].replace("_", "-"): v
        for k, v in environ.items()
        if k.startswith("HTTP_")
    }
    logger.info("=== REQUEST %s %s ===", method, path)
    for name, value in sorted(headers.items()):
        display = value if len(value) < 120 else value[:120] + "...[truncated]"
        logger.info("  HEADER %s: %s", name, display)
    logger.info("=== END HEADERS ===")
    return _payment_wsgi(environ, start_response)

application.wsgi_app = _debug_wsgi
logger.info("Debug middleware enabled")
# --- END DEBUG ---

if __name__ == "__main__":
    port = int(os.getenv("API_SERVER_PORT", "8000"))
    host = os.getenv("API_SERVER_HOST", "0.0.0.0")
    logger.info(f"Starting OpenAI-compatible API server on {host}:{port}")
    logger.info(f"Server ready on {host}:{port}")
    application.run(host=host, port=port)

