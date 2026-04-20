"""
Integration tests for tee_gateway.price_feed.

These tests make REAL network calls to the CoinGecko public API.

Expected behaviour
------------------
* ``TestCoinGeckoConnectivity`` — passes when the CoinGecko API is reachable.
  Skips on network errors or rate-limiting (429).
* ``TestOPGPriceFetchLive`` — skips when OPG is not yet priced on CoinGecko's
  Base platform (CoinGecko currently returns an empty price entry for the
  token).  Will pass automatically once the token is fully listed.

Run with::

    uv run pytest tee_gateway/test/test_price_feed_integration.py -v
"""

import unittest
from decimal import Decimal

import requests

from tee_gateway.definitions import BASE_MAINNET_OPG_ADDRESS
from tee_gateway.price_feed.config import (
    COINGECKO_BASE_URL,
    COINGECKO_PLATFORM,
    FETCH_TIMEOUT,
)
from tee_gateway.price_feed.feed import fetch_opg_price


def _get(url: str, **kwargs) -> requests.Response:
    """Wrapper that skips the test on network errors or rate-limiting."""
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, **kwargs)
    except requests.exceptions.RequestException as exc:
        raise unittest.SkipTest(f"Network unavailable: {exc}") from exc
    if resp.status_code == 429:
        raise unittest.SkipTest(
            "CoinGecko rate limit hit (429) — re-run after a short wait"
        )
    return resp


class TestCoinGeckoConnectivity(unittest.TestCase):
    """Verify that the CoinGecko API endpoint is reachable and well-formed."""

    def test_ping_endpoint_reachable(self):
        """CoinGecko /ping should return {gecko_says: ...}."""
        resp = _get(f"{COINGECKO_BASE_URL}/ping")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("gecko_says", resp.json())

    def test_base_platform_endpoint_returns_200(self):
        """The token_price/base endpoint should respond with HTTP 200 for a known token."""
        url = f"{COINGECKO_BASE_URL}/simple/token_price/{COINGECKO_PLATFORM}"
        # USDC on Base mainnet — reliably indexed on CoinGecko.
        usdc_base = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
        resp = _get(
            url, params={"contract_addresses": usdc_base, "vs_currencies": "usd"}
        )
        self.assertEqual(
            resp.status_code,
            200,
            f"Expected 200 from CoinGecko, got {resp.status_code}: {resp.text[:200]}",
        )
        data = resp.json()
        self.assertIsInstance(data, dict)
        self.assertIn(usdc_base, data, "USDC should be indexed on Base platform")
        self.assertIn("usd", data[usdc_base], "USDC price entry should have 'usd' key")


class TestOPGPriceFetchLive(unittest.TestCase):
    """Live fetch of the OPG token price.

    Both tests skip gracefully when OPG is not yet fully priced on CoinGecko
    (currently returns ``{address: {}}`` with no 'usd' key).  They will pass
    automatically once the token is listed with a live price.
    """

    def test_opg_response_structure(self):
        """Inspect the raw CoinGecko response for the OPG contract address."""
        url = f"{COINGECKO_BASE_URL}/simple/token_price/{COINGECKO_PLATFORM}"
        resp = _get(
            url,
            params={
                "contract_addresses": BASE_MAINNET_OPG_ADDRESS,
                "vs_currencies": "usd",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        print(f"\nCoinGecko response for OPG ({BASE_MAINNET_OPG_ADDRESS}): {data}")  # noqa: T201

        opg_lower = BASE_MAINNET_OPG_ADDRESS.lower()
        price_entry = data.get(opg_lower)
        # CoinGecko returns the address key with {} when the token is known but
        # not yet priced — skip in that case rather than fail.
        if not price_entry or "usd" not in price_entry:
            self.skipTest(
                f"OPG not yet priced on CoinGecko Base platform "
                f"(response: {data!r}). Will pass once the token is fully listed."
            )
        self.assertIsInstance(price_entry["usd"], (int, float))

    def test_opg_price_fetch_live(self):
        """End-to-end: fetch_opg_price() returns a positive Decimal price."""
        try:
            price = fetch_opg_price()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                self.skipTest("CoinGecko rate limit — re-run after a short wait")
            raise
        except ValueError as exc:
            if "Unexpected CoinGecko response" in str(exc):
                self.skipTest(
                    f"OPG ({BASE_MAINNET_OPG_ADDRESS}) not yet priced on "
                    f"CoinGecko Base platform. Details: {exc}"
                )
            raise

        self.assertIsInstance(price, Decimal)
        self.assertGreater(price, Decimal("0"), "Price must be positive")
        print(f"\nLive OPG price: ${price} USD")  # noqa: T201


if __name__ == "__main__":
    unittest.main()
