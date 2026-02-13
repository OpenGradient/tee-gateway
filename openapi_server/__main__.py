"""
OpenAI-compatible API server with TEE integration and x402 payment middleware.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def health():
    return {"status": "OK", "tee_enabled": True}, 200


def attestation():
    """Return TEE attestation document with public key"""
    try:
        tee_keys = get_tee_keys()
        return jsonify(tee_keys.get_attestation_document())
    except Exception as e:
        logger.error(f"Attestation error: {e}")
        return {"error": str(e)}, 500


def create_app():
    app = connexion.App(__name__, specification_dir='./openapi/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('openapi.yaml',
                arguments={'title': 'OpenAI API'},
                pythonic_params=True)

    # Add utility endpoints
    app.app.add_url_rule('/health', 'health', health, methods=['GET'])
    app.app.add_url_rule('/attestation', 'attestation', attestation, methods=['GET'])

    # # Initialize x402 payment middleware
    try:
        from x402.flask.middleware import PaymentMiddleware
        from x402.facilitator import FacilitatorConfig

        payment_middleware = PaymentMiddleware(app.app)
        payment_middleware.add(
            path=["/v1/chat/completions", "/v1/completions"],
            price="0.1",
            pay_to_address="0xbcF7F5f8D7d8a0C03599Eb6d7aA4Bb44Bd84d3A1D",
            network="og-evm",
            facilitator_config=FacilitatorConfig(
                url="https://facilitatorogevm.opengradient.ai",
                config={
                    "network": "og-evm",
                    "price": {
                        "amount": "0.1",
                        "asset": {
                            "address": "0x094E464A23B90A71a0894D5D1e5D470FfDD074e1",
                            "decimals": 6,
                            "eip712": {
                                "name": "OUSDC",
                                "version": "2",
                            },
                        },
                    },
                }
            )
        )
        logger.info("x402 payment middleware initialized")
    except ImportError:
        logger.warning("x402 payment middleware not available - running without payments")
    except Exception as e:
        logger.warning(f"Failed to initialize x402 middleware: {e}")

    return app.app


# Create the WSGI application
application = create_app()


if __name__ == '__main__':
    port = int(os.getenv('API_SERVER_PORT', '8000'))
    host = os.getenv('API_SERVER_HOST', '0.0.0.0')

    logger.info(f"Starting OpenAI-compatible API server on {host}:{port}")

    # Initialize TEE (generate keys, register with nitriding, signal ready)
    try:
        initialize_tee()
    except Exception as e:
        logger.warning(f"TEE initialization failed (may not be in enclave): {e}")

    logger.info(f"Server ready on {host}:{port}")
    application.run(host=host, port=port)
