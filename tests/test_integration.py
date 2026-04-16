"""
Integration tests — require live network access.

These tests hit the real CoinGecko API and are intentionally excluded from
the standard unit-test run. Opt in with:

    pytest -m integration tests/test_integration.py

In CI, these run in a separate job (see .github/workflows/test.yml).
"""

import pytest
from decimal import Decimal


@pytest.mark.integration
class TestCoinGeckoPriceFeed:
    """Verify the live OPG/ETH price fetch end-to-end."""

    def test_fetch_returns_positive_decimal(self):
        """_fetch_opg_price_usd must return a positive Decimal from CoinGecko."""
        from tee_gateway.util import _fetch_opg_price_usd

        price = _fetch_opg_price_usd()
        assert isinstance(price, Decimal)
        assert price > 0, f"Expected positive price, got {price}"

    def test_price_is_plausible_eth_range(self):
        """ETH price should be within a sanity range ($100–$100 000).

        This is a loose check that guards against obviously wrong responses
        (e.g. CoinGecko returning a different currency or a zero value).
        Update bounds if ETH moves outside this range for an extended period.
        """
        from tee_gateway.util import _fetch_opg_price_usd

        price = _fetch_opg_price_usd()
        assert Decimal("100") < price < Decimal("100000"), (
            f"ETH price ${price} is outside the expected sanity range $100–$100 000"
        )

    def test_get_token_a_price_usd_returns_cached_value(self):
        """get_token_a_price_usd must return the same value on two rapid calls
        (second call must hit the cache, not make a second network request)."""
        from tee_gateway.util import _token_price_cache, get_token_a_price_usd

        # Reset cache so first call is a fresh fetch
        _token_price_cache["last_good"] = None
        _token_price_cache["updated_at"] = 0.0

        first = get_token_a_price_usd()
        second = get_token_a_price_usd()

        assert first == second, "Cache should return the same price on the second call"
        assert first > 0

    def test_dynamic_cost_uses_live_price(self):
        """Full pipeline: token counts + live ETH price → positive on-chain units."""
        from tee_gateway.definitions import BASE_TESTNET_OPG_ADDRESS
        from tee_gateway.util import _token_price_cache, dynamic_session_cost_calculator, get_token_a_price_usd

        # Reset cache to force a live fetch
        _token_price_cache["last_good"] = None
        _token_price_cache["updated_at"] = 0.0

        ctx = {
            "request_json": {"model": "gpt-4.1"},
            "response_json": {
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500}
            },
            "payment_requirements": {"asset": BASE_TESTNET_OPG_ADDRESS},
        }

        cost = dynamic_session_cost_calculator(ctx)
        live_price = get_token_a_price_usd()

        assert isinstance(cost, int)
        assert cost > 0

        # Sanity: cost should be far less than 1 full OPG (10^18 units)
        # for a small request at any plausible ETH price
        assert cost < 10 ** 18, f"Cost {cost} seems too large for a small request"

        print(f"\nLive ETH price: ${live_price}")
        print(f"Cost for gpt-4.1 (1000 input + 500 output tokens): {cost} OPG units")
