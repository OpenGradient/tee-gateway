"""
OpenAI-compatible API server with TEE integration and x402 payment middleware.
Runs inside a Nitro Enclave, proxied by nitriding on port 443.
"""

import logging
import sys
import os
import threading
import atexit

import connexion
from flask import jsonify, request
from tee_gateway import encoder
from tee_gateway.tee_manager import initialize_tee, get_tee_keys
from tee_gateway.config import (
    HeartbeatConfig,
    ProviderConfig,
    DEFAULT_HEARTBEAT_BUFFER,
    DEFAULT_HEARTBEAT_INTERVAL,
)
from tee_gateway.llm_backend import set_provider_config
from tee_gateway.heartbeat import create_heartbeat_service

from x402.http import FacilitatorConfig, HTTPFacilitatorClientSync, PaymentOption
from x402.http.middleware.flask import payment_middleware
from x402.http.types import RouteConfig
from x402.mechanisms.evm.exact import ExactEvmServerScheme
from x402.mechanisms.evm.upto import UptoEvmServerScheme
from x402.extensions.erc20_approval_gas_sponsoring import (
    declare_erc20_approval_gas_sponsoring_extension,
)
from x402.schemas import AssetAmount
from x402.server import x402ResourceServerSync
from x402.session import SessionStore
import x402.http.middleware.flask as x402_flask

from .util import calculate_session_cost
from .model_registry import get_model_config
from .price_feed import OPGPriceFeed
from .definitions import (
    EVM_PAYMENT_ADDRESS,
    BASE_MAINNET_NETWORK,
    BASE_MAINNET_OPG_ADDRESS,
    CHAT_COMPLETIONS_OPG_SESSION_MAX_SPEND,
    FACILITATOR_URL,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# Centralised format ensures every log line — including third-party libraries
# such as werkzeug and connexion — carries a UTC timestamp with millisecond
# precision, making production debugging and log correlation straightforward.

LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

# Force third-party loggers to propagate through the root logger so they
# inherit the timestamped format above.  Without this, libraries that attach
# their own StreamHandler (e.g. werkzeug, connexion) would emit lines with
# the default Python format — which has no timestamp.
for _lib_name in ("werkzeug", "connexion"):
    _lib_logger = logging.getLogger(_lib_name)
    _lib_logger.handlers.clear()
    _lib_logger.propagate = True

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

# ---------------------------------------------------------------------------
# OPG price feed — start before x402 middleware so the first request can be
# priced correctly.  Runs as a daemon thread; no cleanup needed on exit.
# ---------------------------------------------------------------------------
_price_feed = OPGPriceFeed()
_price_feed.start()

facilitator = HTTPFacilitatorClientSync(FacilitatorConfig(url=FACILITATOR_URL))
server = x402ResourceServerSync(facilitator)
store = SessionStore()

server.register(BASE_MAINNET_NETWORK, ExactEvmServerScheme())

# Upto scheme registrations (permit2-based, variable settlement)
server.register(BASE_MAINNET_NETWORK, UptoEvmServerScheme())

routes = {
    "POST /v1/chat/completions": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="upto",
                pay_to=EVM_PAYMENT_ADDRESS,
                price=AssetAmount(
                    amount=CHAT_COMPLETIONS_OPG_SESSION_MAX_SPEND,
                    asset=BASE_MAINNET_OPG_ADDRESS,
                    extra={
                        "name": "OpenGradient",
                        "version": "1",
                        "assetTransferMethod": "permit2",
                    },
                ),
                network=BASE_MAINNET_NETWORK,
            ),
        ],
        extensions={
            **declare_erc20_approval_gas_sponsoring_extension(),
        },
        mime_type="application/json",
        description="Chat completion",
    ),
    "POST /v1/completions": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="upto",
                pay_to=EVM_PAYMENT_ADDRESS,
                price=AssetAmount(
                    amount=CHAT_COMPLETIONS_OPG_SESSION_MAX_SPEND,
                    asset=BASE_MAINNET_OPG_ADDRESS,
                    extra={
                        "name": "OpenGradient",
                        "version": "1",
                        "assetTransferMethod": "permit2",
                    },
                ),
                network=BASE_MAINNET_NETWORK,
            ),
        ],
        extensions={
            **declare_erc20_approval_gas_sponsoring_extension(),
        },
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
            buffer_raw = body.get("tee_heartbeat_buffer", DEFAULT_HEARTBEAT_BUFFER)
            try:
                timestamp_buffer = int(buffer_raw)
            except (ValueError, TypeError):
                logger.error(
                    "Invalid tee_heartbeat_buffer %r; falling back to default %s",
                    buffer_raw,
                    DEFAULT_HEARTBEAT_BUFFER,
                )
                timestamp_buffer = DEFAULT_HEARTBEAT_BUFFER
            heartbeat_config = HeartbeatConfig(
                contract_address=contract_address,
                facilitator_url=facilitator_url,
                interval=interval,
                timestamp_buffer=timestamp_buffer,
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
    return {
        "status": "OK",
        "version": "1.0.0",
        "tee_enabled": True,
        "price_feed": _price_feed.get_status(),
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
# WSGI application + x402 payment middleware
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


def _session_cost_calculator(ctx: dict) -> int:
    # Post-inference cost calculation — response already sent to client.
    # Predictable failures (unknown price, unknown model) are blocked by the
    # pre-inference gate; any exception here indicates a provider-side error
    # (e.g. missing usage field in the LLM response).  The x402 middleware
    # swallows the exception in close(), so the client is not charged.
    # Log CRITICAL so provider errors are never silently missed.
    try:
        return calculate_session_cost(ctx, _price_feed.get_price)
    except Exception as exc:
        logger.critical(
            "Post-inference cost calculation failed (provider error) — "
            "client was NOT charged: %s",
            exc,
            exc_info=True,
        )
        raise


_payment_mw = payment_middleware(
    application,
    routes=routes,
    server=server,
    session_store=store,
    cost_per_request=100000000000000,  # static precheck/fallback estimate
    session_idle_timeout=100,
    session_cost_calculator=_session_cost_calculator,
)

# ---------------------------------------------------------------------------
# Pre-inference pricing gate
#
# In the upto session scheme the response is streamed to the client before
# cost is settled, so a post-inference pricing failure cannot be surfaced as
# an HTTP error.  Instead we validate everything that can be checked up-front
# and reject the request early if pricing would fail:
#   1. Price feed has a valid OPG/USD price (CoinGecko fetch succeeded).
#   2. The requested model is in the registry (has a known per-token price).
# ---------------------------------------------------------------------------


@application.before_request
def _check_pricing_ready():
    if request.path not in ("/v1/chat/completions", "/v1/completions"):
        return
    try:
        _price_feed.get_price()
    except ValueError as exc:
        logger.warning("Rejecting inference request — price feed unavailable: %s", exc)
        return jsonify({"error": f"Pricing unavailable: {exc}"}), 503

    body = request.get_json(silent=True, cache=True) or {}
    model = body.get("model")
    if model:
        try:
            get_model_config(model)
        except ValueError:
            return jsonify({"error": f"Model '{model}' is not supported"}), 400


logger.info("x402 payment middleware initialized")

if __name__ == "__main__":
    port = int(os.getenv("API_SERVER_PORT", "8000"))
    host = os.getenv("API_SERVER_HOST", "0.0.0.0")
    logger.info(f"Starting OpenAI-compatible API server on {host}:{port}")
    application.run(host=host, port=port)
