"""
Integration tests — require live network access.

These tests hit the real CoinGecko API and are intentionally excluded from
the standard unit-test run. Opt in with:

    pytest -m integration tests/test_integration.py

In CI, these run in a separate job (see .github/workflows/test.yml).

NOTE: OPG (opengradient) is listed on CoinGecko but currently has no trading
price. The fetch tests below verify the correct error behaviour until a price
becomes available. Tests that require a live price are skipped automatically
when OPG has no price data.
"""

import pytest
from decimal import Decimal


def _opg_has_price() -> bool:
    """Return True if CoinGecko currently reports a price for OPG."""
    try:
        from tee_gateway.util import _fetch_opg_price_usd

        _fetch_opg_price_usd()
        return True
    except Exception:
        return False


requires_opg_price = pytest.mark.skipif(
    not _opg_has_price(),
    reason="OPG has no trading price on CoinGecko yet",
)


@pytest.mark.integration
class TestCoinGeckoPriceFeed:
    """Verify the live OPG price fetch end-to-end via the configured CoinGecko token."""

    def test_coingecko_slug_is_recognised(self):
        """CoinGecko must recognise the OPG slug (i.e. return a dict for the coin,
        even if price data is absent). A completely unknown slug returns an empty dict."""
        import json
        import urllib.request

        from tee_gateway.config import OPG_PRICE_COINGECKO_ID

        url = (
            f"https://api.coingecko.com/api/v3/simple/price"
            f"?ids={OPG_PRICE_COINGECKO_ID}&vs_currencies=usd"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "tee-gateway/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        assert OPG_PRICE_COINGECKO_ID in data, (
            f"CoinGecko did not recognise slug '{OPG_PRICE_COINGECKO_ID}'. "
            f"Response: {data!r}"
        )

    def test_fetch_raises_clear_error_when_no_price(self):
        """When OPG has no trading price, _fetch_opg_price_usd must raise ValueError
        with a message indicating the price is unavailable — not a bare KeyError."""
        import urllib.error

        if _opg_has_price():
            pytest.skip("OPG now has a price — this test is no longer applicable")

        from tee_gateway.util import _fetch_opg_price_usd

        try:
            _fetch_opg_price_usd()
            pytest.fail("Expected ValueError but no exception was raised")
        except ValueError as exc:
            assert "no price" in str(exc), f"Unexpected ValueError message: {exc}"
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                pytest.skip(f"CoinGecko rate-limited the integration test run: {exc}")
            raise

    @requires_opg_price
    def test_fetch_returns_positive_decimal(self):
        """_fetch_opg_price_usd must return a positive Decimal from CoinGecko."""
        from tee_gateway.util import _fetch_opg_price_usd

        price = _fetch_opg_price_usd()
        assert isinstance(price, Decimal)
        assert price > 0, f"Expected positive price, got {price}"

    @requires_opg_price
    def test_price_is_within_sanity_bounds(self):
        """Fetched price must not exceed the configured sanity ceiling."""
        from tee_gateway.config import OPG_PRICE_SANITY_MAX_USD
        from tee_gateway.util import _fetch_opg_price_usd

        price = _fetch_opg_price_usd()
        max_price = Decimal(OPG_PRICE_SANITY_MAX_USD)
        assert 0 < price < max_price, (
            f"Price ${price} is outside the expected range (0, ${OPG_PRICE_SANITY_MAX_USD})"
        )

    @requires_opg_price
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

    @requires_opg_price
    def test_dynamic_cost_uses_live_price(self):
        """Full pipeline: token counts + live token price → positive on-chain units."""
        from tee_gateway.definitions import BASE_TESTNET_OPG_ADDRESS
        from tee_gateway.util import (
            _token_price_cache,
            dynamic_session_cost_calculator,
            get_token_a_price_usd,
        )

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
        # for a small request at any plausible token price
        assert cost < 10**18, f"Cost {cost} seems too large for a small request"

        from tee_gateway.config import OPG_PRICE_COINGECKO_ID

        print(f"\nLive price ({OPG_PRICE_COINGECKO_ID}): ${live_price}")
        print(f"Cost for gpt-4.1 (1000 input + 500 output tokens): {cost} OPG units")
