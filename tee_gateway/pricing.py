"""Per-request session-cost calculation for x402 settlement.

Converts realized token usage from an LLM response into an OPG smallest-units
integer (what x402 actually charges) plus the equivalent USD figure and the
OPG/USD price used for the conversion.  All three are bundled in
:class:`SessionCost`, embedded on the response under the ``opengradient`` key,
and read back by both the x402 settlement calculator and the OHTTP outer-header
extractor.  This module is the single source of truth for that math.
"""

import logging
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, field_serializer

from tee_gateway.definitions import (
    ASSET_DECIMALS_BY_ADDRESS,
    BASE_MAINNET_OPG_ADDRESS,
)
from tee_gateway.model_registry import get_model_config

logger = logging.getLogger("llm_server.dynamic_pricing")

_OPG_DECIMALS = ASSET_DECIMALS_BY_ADDRESS[BASE_MAINNET_OPG_ADDRESS.lower()]


def _as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(by_alias=True, exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return None


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _normalize_model_name(model: str | None) -> str | None:
    if not model:
        return None
    return str(model).strip().lower()


def _extract_usage_tokens(
    response_json: dict[str, Any] | None,
) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from response JSON.

    Raises ValueError if usage data is missing or malformed — no silent fallback.
    """
    if not isinstance(response_json, dict):
        raise ValueError("response_json is not a dict; cannot extract usage tokens")
    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        raise ValueError(
            "response_json has no 'usage' dict; cannot extract usage tokens"
        )

    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if prompt_tokens is None or completion_tokens is None:
        raise ValueError(f"usage dict is missing token counts: {usage!r}")

    try:
        return max(0, int(prompt_tokens)), max(0, int(completion_tokens))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse token counts from usage: {usage!r}") from exc


def _extract_model_from_context(
    request_json: dict[str, Any] | None,
    response_json: dict[str, Any] | None,
) -> str:
    """Extract and normalize model name from request JSON.

    Uses only the request model name — the response model field is ignored
    because providers may return a versioned alias that differs from the
    user-facing name.  Raises ValueError if the model name is absent.
    """
    if not isinstance(request_json, dict):
        raise ValueError("request_json is not a dict; cannot extract model name")
    req_model = request_json.get("model")
    if not req_model:
        raise ValueError("request_json has no 'model' field")
    normalized = _normalize_model_name(req_model)
    if not normalized:
        raise ValueError(f"model name normalizes to empty string: {req_model!r}")
    return normalized


def _extract_asset_decimals_from_requirements(payment_requirements: Any) -> int:
    req = _as_dict(payment_requirements) or {}

    asset = req.get("asset")
    if not asset and isinstance(req.get("price"), dict):
        asset = req["price"].get("asset")

    if not isinstance(asset, str) or not asset:
        raise ValueError(
            f"payment_requirements has no recognizable asset address; "
            f"cannot determine token decimals: {req!r}"
        )

    asset_lower = asset.lower()
    if asset_lower not in ASSET_DECIMALS_BY_ADDRESS:
        raise ValueError(
            f"Unknown asset address {asset!r}; not in ASSET_DECIMALS_BY_ADDRESS. "
            f"Add it to definitions.py before accepting payments with this token."
        )
    return ASSET_DECIMALS_BY_ADDRESS[asset_lower]


class SessionCost(BaseModel):
    """Settled per-request cost. ``cost_opg`` is what x402 actually charges;
    ``cost_usd`` and ``opg_price_usd`` are reported so clients and relays can
    audit the conversion without re-fetching the price feed.

    All three fields serialize to JSON strings: the OPG integer can exceed JS
    safe-int (2^53) for any non-trivial cost at 18 decimals, and the Decimals
    would lose precision through a float round-trip.
    """

    model_config = ConfigDict(frozen=True)

    cost_opg: int
    cost_usd: Decimal
    opg_price_usd: Decimal

    @field_serializer("cost_opg")
    def _serialize_opg(self, value: int) -> str:
        return str(value)

    @field_serializer("cost_usd", "opg_price_usd")
    def _serialize_decimal(self, value: Decimal) -> str:
        return format(value, "f")


def calculate_session_cost(
    request_json: dict[str, Any],
    response_json: dict[str, Any],
    asset_decimals: int,
    get_price: Callable[[], Decimal],
) -> SessionCost:
    """Compute the settled cost for a completed inference request.

    ``get_price`` is called on every invocation to fetch the current OPG/USD
    price — pass ``price_feed.get_price`` so the latest cached value is used.
    Raises ``ValueError`` on any missing/invalid data.  Predictable failures
    (unavailable price, unknown model) are blocked before inference by the
    pre-inference gate in ``__main__.py``; post-inference failures are logged
    as CRITICAL by the caller and the client is not charged.

    Returns both the OPG integer (what x402 charges) and the equivalent USD —
    derived from the SAME rounded OPG value, not the raw USD math — so the two
    numbers always reconcile via ``cost_usd = cost_opg / 10^decimals * price``.
    """
    if not isinstance(request_json, dict) or not isinstance(response_json, dict):
        raise ValueError(
            "calculate_session_cost requires both request_json and response_json"
        )

    model = _extract_model_from_context(request_json, response_json)
    cfg = get_model_config(model)
    input_tokens, output_tokens = _extract_usage_tokens(response_json)

    raw_usd = (Decimal(input_tokens) * cfg.input_price_usd) + (
        Decimal(output_tokens) * cfg.output_price_usd
    )
    token_price_usd = get_price()
    if token_price_usd <= 0:
        raise ValueError(f"Token price is non-positive: {token_price_usd}")

    scale = Decimal(10) ** asset_decimals
    cost_smallest_units = max(
        0,
        int(
            ((raw_usd / token_price_usd) * scale).to_integral_value(
                rounding=ROUND_CEILING
            )
        ),
    )
    # Reconcile USD from the rounded OPG integer so the two surfaced figures
    # are exactly consistent (clients verify: usd == opg / 10^decimals * price).
    settled_usd = (Decimal(cost_smallest_units) / scale) * token_price_usd

    logger.info(
        "DYNAMIC_SESSION_COST model=%s input_tokens=%d output_tokens=%d "
        "raw_usd=%s settled_usd=%s token_price_usd=%s decimals=%d cost=%d",
        model,
        input_tokens,
        output_tokens,
        str(raw_usd),
        str(settled_usd),
        str(token_price_usd),
        asset_decimals,
        cost_smallest_units,
    )
    return SessionCost(
        cost_opg=cost_smallest_units,
        cost_usd=settled_usd,
        opg_price_usd=token_price_usd,
    )


def compute_session_cost(
    request_json: dict[str, Any], response_with_usage: dict[str, Any]
) -> SessionCost | None:
    """Wrap calculate_session_cost for controllers: returns the SessionCost
    pydantic model, or ``None`` on failure. Predictable failures (unknown
    price/model) are blocked by the pre-inference gate, so anything reaching
    here is a provider-side error (e.g. missing usage). Logging as CRITICAL
    matches __main__._session_cost_calculator's contract: when this returns
    None, x402's downstream reader will skip settlement and the client is not
    charged.
    """
    # Imported lazily to keep this module free of process-level state at import
    # time (price_feed singleton is registered during app startup, after this
    # module has already been imported transitively via other paths).
    from tee_gateway.price_feed import get_price_feed

    try:
        return calculate_session_cost(
            request_json=request_json,
            response_json=response_with_usage,
            asset_decimals=_OPG_DECIMALS,
            get_price=get_price_feed().get_price,
        )
    except Exception as exc:
        logger.critical(
            "Post-inference cost calculation failed (provider error) — "
            "client will NOT be charged: %s",
            exc,
            exc_info=True,
        )
        return None
