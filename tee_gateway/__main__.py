"""
OpenAI-compatible API server with TEE integration and x402 payment middleware.
Runs inside a Nitro Enclave, proxied by nitriding on port 443.
"""

import logging
import sys
import os
import threading
import time
import atexit
from importlib.metadata import PackageNotFoundError, version as pkg_version

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
from tee_gateway.llm_backend import get_provider_config, set_provider_config
from tee_gateway.heartbeat import create_heartbeat_service
from tee_gateway.controllers.ohttp_controller import (
    create_anonymous_chat_completion,
    get_hpke_config,
)

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

from .model_registry import get_model_config
from .price_feed import OPGPriceFeed, set_price_feed
from .definitions import (
    EVM_PAYMENT_ADDRESS,
    BASE_MAINNET_NETWORK,
    BASE_MAINNET_OPG_ADDRESS,
    CHAT_COMPLETIONS_OPG_SESSION_MAX_SPEND,
    COMPLETIONS_OPG_SESSION_MAX_SPEND,
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
set_price_feed(_price_feed)

_started_at = time.time()


def _gateway_version() -> str:
    try:
        return pkg_version("tee-gateway")
    except PackageNotFoundError:
        return "unknown"


_active_facilitator_url: str | None = None


# ---------------------------------------------------------------------------
# x402 read-body patch
#
# Ensures that non-payment 0-length requests can bypass the middleware without
# errors. Applied at module load so it is in place before the middleware
# instance is created at injection time.
# ---------------------------------------------------------------------------
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


def _patched_stream_session_response(
    self,
    environ,
    start_response,
    context,
    session_id,
    payment_payload,
    payment_requirements,
):
    """Expose x402's per-request cost context to Flask route handlers.

    OHTTP requests arrive as ciphertext and return ciphertext, so the x402
    middleware cannot parse request_json/response_json from the outer HTTP
    bodies. The OHTTP controller decrypts the inner request and plaintext
    response inside the enclave; this patch gives it a request-local dict where
    it can attach those inner JSON objects for dynamic settlement.
    """
    self._start_reaper()

    request_body_bytes = x402_flask._read_body_bytes(environ)
    request_json = x402_flask._try_parse_json(request_body_bytes)
    parsed_request_json = (
        request_json if isinstance(request_json, (dict, list)) else None
    )

    x402_flask.g.payment_payload = payment_payload
    x402_flask.g.payment_requirements = payment_requirements
    x402_flask.g.x402_session_id = session_id

    cost_context = {
        "method": context.method,
        "path": context.path,
        "request_body_bytes": request_body_bytes,
        "request_json": parsed_request_json,
        "payment_payload": payment_payload,
        "payment_requirements": payment_requirements,
    }
    environ["x402.cost_context"] = cost_context

    status_capture = x402_flask.StatusCapture(start_response)
    status_capture.add_header(x402_flask.UPTO_SESSION_HEADER, session_id)

    upstream_iter = self._original_wsgi(environ, status_capture)

    return x402_flask.StreamingSessionResponse(
        upstream_iter,
        middleware=self,
        session_id=session_id,
        cost_context=cost_context,
        status_ref=status_capture,
    )


x402_flask.PaymentMiddleware._stream_session_response = (
    _patched_stream_session_response
)


def _session_cost_calculator(ctx: dict) -> int:
    # The chat/completions controllers compute cost in-band and embed it on
    # the response as a SessionCost model BEFORE returning. We parse it back
    # here — single source of truth for cost lives in the controller, not
    # split between the controller (which serves clients) and x402's callback
    # (which charges them).
    #
    # If the controller couldn't compute cost (e.g. missing usage from the
    # provider), the block is absent — SessionCost.model_validate raises and
    # x402's close() swallows it so the client is not charged. The controller
    # has already logged CRITICAL in that case.
    from .pricing import SessionCost

    if ctx.get("path") == "/v1/ohttp":
        response_json = ctx.get("inner_response_json")
    else:
        response_json = ctx.get("response_json")
    if not isinstance(response_json, dict):
        raise ValueError("response_json missing or not a dict")
    cost_block = response_json.get("opengradient")
    if cost_block is None:
        # Should never happen on a successful paid response — the controller
        # always embeds this when it can compute it, and when it can't, the
        # request typically errored out before reaching x402's close hook.
        # If we see this, settlement silently skips and a real bug is hiding.
        logger.critical(
            "opengradient cost block missing on paid response — client will "
            "NOT be charged. response_id=%s",
            response_json.get("id"),
        )
    return SessionCost.model_validate(cost_block).cost_opg


# ---------------------------------------------------------------------------
# One-time runtime configuration injection
# ---------------------------------------------------------------------------
_keys_initialized: bool = False
_keys_lock = threading.Lock()


def _init_payment_middleware(facilitator_url: str) -> None:
    """Build and attach the x402 payment middleware to the running Flask app.

    Called once from set_provider_keys() after the facilitator URL is known.
    Swaps application.wsgi_app so all subsequent requests flow through it.
    """
    facilitator = HTTPFacilitatorClientSync(FacilitatorConfig(url=facilitator_url))
    server = x402ResourceServerSync(facilitator)  # type: ignore[arg-type]
    store = SessionStore()

    server.register(BASE_MAINNET_NETWORK, ExactEvmServerScheme())
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
                        amount=COMPLETIONS_OPG_SESSION_MAX_SPEND,
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
        "POST /v1/ohttp": RouteConfig(
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
            mime_type="message/ohttp-req",
            description="OHTTP-wrapped chat completion",
        ),
    }

    inner_wsgi_app = application.wsgi_app
    flask_app = getattr(application, "app", application)
    flask_app.config["OHTTP_INNER_WSGI_APP"] = inner_wsgi_app

    # Return value intentionally discarded — PaymentMiddleware.__init__ self-wires
    # by setting application.wsgi_app = self._wsgi_middleware internally.
    payment_middleware(
        application,
        routes=routes,
        server=server,
        session_store=store,
        cost_per_request=100000000000000,  # static precheck/fallback estimate
        session_idle_timeout=100,
        session_cost_calculator=_session_cost_calculator,
    )
    logger.info(
        "x402 payment middleware initialized with facilitator: %s", facilitator_url
    )


def set_provider_keys():
    """
    POST /v1/keys — inject runtime configuration into the enclave.

    Accepts LLM provider API keys, a shared facilitator_url (used for both
    x402 payment verification and the heartbeat relay), and optional heartbeat
    parameters. Can only be called once; subsequent calls return HTTP 409.
    """
    global _keys_initialized, _active_facilitator_url

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
            bytedance_api_key=body.get("bytedance_api_key") or None,
        )
        set_provider_config(provider_config)

        facilitator_url = body.get("facilitator_url") or FACILITATOR_URL

        # Build heartbeat config from request body (optional)
        contract_address = body.get("heartbeat_contract_address")
        heartbeat_config: HeartbeatConfig | None = None
        if contract_address:
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
            "  bytedance_api_key           : %s",
            _set(provider_config.bytedance_api_key),
        )
        logger.info("  facilitator_url             : %s", facilitator_url)
        logger.info(
            "  heartbeat_contract_address  : %s",
            _set(heartbeat_config.contract_address if heartbeat_config else None),
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

        _init_payment_middleware(facilitator_url)

        _active_facilitator_url = facilitator_url
        _keys_initialized = True

    providers_set = [
        p
        for p, k in {
            "openai": provider_config.openai_api_key,
            "google": provider_config.google_api_key,
            "anthropic": provider_config.anthropic_api_key,
            "xai": provider_config.xai_api_key,
            "bytedance": provider_config.bytedance_api_key,
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
    cfg = get_provider_config()
    providers = cfg.initialized_providers() if cfg else []

    return {
        "status": "OK",
        "version": _gateway_version(),
        "tee_enabled": True,
        "uptime_seconds": int(time.time() - _started_at),
        "providers": providers,
        "facilitator_url": _active_facilitator_url,
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

    # Anonymous inference (OHTTP-wrapped chat completions). Deliberately
    # mounted via add_url_rule rather than the OpenAPI spec because the body
    # is raw binary and connexion's request-validation pipeline would reject
    # it as malformed JSON.
    app.app.add_url_rule(
        "/v1/ohttp",
        "anonymous-chat",
        create_anonymous_chat_completion,
        methods=["POST"],
    )
    app.app.add_url_rule(
        "/v1/ohttp/config", "ohttp-config", get_hpke_config, methods=["GET"]
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
# WSGI application
# ---------------------------------------------------------------------------

application = create_app()


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
    if request.path not in ("/v1/chat/completions", "/v1/completions", "/v1/ohttp"):
        return
    try:
        _price_feed.get_price()
    except ValueError as exc:
        logger.warning("Rejecting inference request — price feed unavailable: %s", exc)
        return jsonify({"error": f"Pricing unavailable: {exc}"}), 503

    if request.path == "/v1/ohttp":
        return

    body = request.get_json(silent=True, cache=True) or {}
    model = body.get("model")
    if model:
        try:
            get_model_config(model)
        except ValueError:
            return jsonify({"error": f"Model '{model}' is not supported"}), 400


if __name__ == "__main__":
    port = int(os.getenv("API_SERVER_PORT", "8000"))
    host = os.getenv("API_SERVER_HOST", "0.0.0.0")
    logger.info(f"Starting OpenAI-compatible API server on {host}:{port}")
    application.run(host=host, port=port)
