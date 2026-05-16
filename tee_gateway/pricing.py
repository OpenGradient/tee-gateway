"""Per-request session-cost calculation for x402 settlement.

Converts realized token usage from an LLM response into an OPG smallest-units
integer (what x402 actually charges) plus the equivalent USD figure and the
OPG/USD price used for the conversion.  All three are bundled in
:class:`SessionCost`, embedded on the response under the ``opengradient`` key,
and read back by both the x402 settlement calculator and the OHTTP outer-header
extractor.  This module is the single source of truth for that math.
"""

import logging
from decimal import Decimal, ROUND_CEILING

from pydantic import BaseModel, ConfigDict, field_serializer

from tee_gateway.definitions import (
    ASSET_DECIMALS_BY_ADDRESS,
    BASE_MAINNET_OPG_ADDRESS,
)
from tee_gateway.model_registry import get_model_config

logger = logging.getLogger("llm_server.dynamic_pricing")

_OPG_DECIMALS = ASSET_DECIMALS_BY_ADDRESS[BASE_MAINNET_OPG_ADDRESS.lower()]


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


def compute_session_cost(model: str, usage: dict) -> SessionCost | None:
    """Compute the settled cost for a completed inference request.

    Returns the SessionCost pydantic model, or ``None`` on failure.  Predictable
    failures (unknown model, unavailable price) are blocked by the pre-inference
    gate, so anything reaching here is a provider-side error (e.g. missing
    usage) or a transient price-feed outage.  Logging as CRITICAL matches
    ``__main__._session_cost_calculator``'s contract: when this returns None,
    x402's downstream reader will skip settlement and the client is not charged.

    Returns both the OPG integer (what x402 charges) and the equivalent USD —
    derived from the SAME rounded OPG value, not the raw USD math — so the two
    numbers always reconcile via ``cost_usd = cost_opg / 10^decimals * price``.
    """
    # Imported lazily to keep this module free of process-level state at import
    # time (price_feed singleton is registered during app startup, after this
    # module has already been imported transitively via other paths).
    from tee_gateway.price_feed import get_price_feed

    try:
        cfg = get_model_config(model.strip().lower())
        in_tok = max(0, int(usage["prompt_tokens"]))
        out_tok = max(0, int(usage["completion_tokens"]))

        raw_usd = (Decimal(in_tok) * cfg.input_price_usd) + (
            Decimal(out_tok) * cfg.output_price_usd
        )
        token_price_usd = get_price_feed().get_price()
        if token_price_usd <= 0:
            raise ValueError(f"Token price is non-positive: {token_price_usd}")

        scale = Decimal(10) ** _OPG_DECIMALS
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
            in_tok,
            out_tok,
            str(raw_usd),
            str(settled_usd),
            str(token_price_usd),
            _OPG_DECIMALS,
            cost_smallest_units,
        )
        return SessionCost(
            cost_opg=cost_smallest_units,
            cost_usd=settled_usd,
            opg_price_usd=token_price_usd,
        )
    except Exception as exc:
        logger.critical(
            "Post-inference cost calculation failed (provider error) — "
            "client will NOT be charged: %s",
            exc,
            exc_info=True,
        )
        return None
