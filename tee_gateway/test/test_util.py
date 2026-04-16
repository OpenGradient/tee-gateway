"""
Tests for tee_gateway.util — OPG price fetching, caching, and dynamic cost calculation.

All tests are fully offline: urllib.request.urlopen is patched so no real
network call is ever made.
"""

import json
import unittest
from decimal import Decimal
from io import BytesIO
from unittest.mock import MagicMock, patch

from tee_gateway import util
from tee_gateway.config import OPG_PRICE_COINGECKO_ID
from tee_gateway.util import (
    _fetch_opg_price_usd,
    _token_price_cache,
    dynamic_session_cost_calculator,
    get_token_a_price_usd,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_urlopen_response(price: float) -> MagicMock:
    """Return a mock context-manager that urlopen returns with a CoinGecko payload."""
    body = json.dumps({OPG_PRICE_COINGECKO_ID: {"usd": price}}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _reset_price_cache() -> None:
    """Reset module-level price cache to pristine state between tests."""
    _token_price_cache["last_good"] = None
    _token_price_cache["updated_at"] = 0.0


# ---------------------------------------------------------------------------
# _fetch_opg_price_usd
# ---------------------------------------------------------------------------


class TestFetchOPGPrice(unittest.TestCase):
    """_fetch_opg_price_usd makes one HTTP call and returns a Decimal price."""

    def setUp(self):
        _reset_price_cache()

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_returns_decimal_price(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(3000.50)
        price = _fetch_opg_price_usd()
        self.assertIsInstance(price, Decimal)
        self.assertEqual(price, Decimal("3000.5"))

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_uses_configured_coingecko_id_in_url(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(3000.0)
        _fetch_opg_price_usd()
        call_args = mock_urlopen.call_args
        # First positional arg is the Request object
        req = call_args[0][0]
        self.assertIn(OPG_PRICE_COINGECKO_ID, req.full_url)

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_raises_on_non_positive_price(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(0.0)
        with self.assertRaises(ValueError):
            _fetch_opg_price_usd()

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_raises_on_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("connection refused")
        with self.assertRaises(OSError):
            _fetch_opg_price_usd()

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_raises_on_malformed_json(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not-json"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        with self.assertRaises(Exception):
            _fetch_opg_price_usd()


# ---------------------------------------------------------------------------
# get_token_a_price_usd — caching behaviour
# ---------------------------------------------------------------------------


class TestGetTokenAPriceUSD(unittest.TestCase):
    """get_token_a_price_usd must respect the TTL and fallback gracefully."""

    def setUp(self):
        _reset_price_cache()

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_returns_fetched_price_on_cold_cache(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(3000.0)
        price = get_token_a_price_usd()
        self.assertEqual(price, Decimal("3000.0"))

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_cache_hit_skips_second_network_call(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(3000.0)
        get_token_a_price_usd()  # populates cache
        get_token_a_price_usd()  # should hit cache
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("tee_gateway.util.urllib.request.urlopen")
    @patch("tee_gateway.util.time")
    def test_cache_expires_after_ttl(self, mock_time, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(3000.0)
        mock_time.time.return_value = 1_000_000.0
        get_token_a_price_usd()

        # Advance time past TTL
        from tee_gateway.config import OPG_PRICE_CACHE_TTL_SECONDS
        mock_time.time.return_value = 1_000_000.0 + OPG_PRICE_CACHE_TTL_SECONDS + 1
        mock_urlopen.return_value = _make_urlopen_response(3500.0)
        price = get_token_a_price_usd()

        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(price, Decimal("3500.0"))

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_stale_cache_used_on_refresh_failure(self, mock_urlopen):
        """If the cache is populated but a refresh fails, the last good price is returned."""
        mock_urlopen.return_value = _make_urlopen_response(3000.0)
        first = get_token_a_price_usd()
        self.assertEqual(first, Decimal("3000.0"))

        # Force cache to appear expired then make the refresh fail
        _token_price_cache["updated_at"] = 0.0
        mock_urlopen.side_effect = OSError("network down")
        second = get_token_a_price_usd()

        self.assertEqual(second, Decimal("3000.0"))  # stale value returned

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_hard_fallback_used_when_never_fetched_and_network_fails(self, mock_urlopen):
        """With empty cache and a failing network, the hard fallback price is returned."""
        mock_urlopen.side_effect = OSError("network down")
        price = get_token_a_price_usd()
        self.assertEqual(price, util._PRICE_HARD_FALLBACK_USD)
        self.assertGreater(price, 0)


# ---------------------------------------------------------------------------
# dynamic_session_cost_calculator — end-to-end with mocked price
# ---------------------------------------------------------------------------


class TestDynamicSessionCostCalculator(unittest.TestCase):
    """Full pipeline: token counts + model pricing + OPG price → on-chain units."""

    def setUp(self):
        _reset_price_cache()

    def _make_context(
        self,
        model: str = "gpt-4.1",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        asset: str = "0x240b09731D96979f50B2C649C9CE10FcF9C7987F",  # OPG testnet
    ) -> dict:
        return {
            "request_json": {"model": model},
            "response_json": {
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            },
            "payment_requirements": {"asset": asset},
        }

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_returns_positive_integer(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(3000.0)
        cost = dynamic_session_cost_calculator(self._make_context())
        self.assertIsInstance(cost, int)
        self.assertGreater(cost, 0)

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_higher_token_price_reduces_cost(self, mock_urlopen):
        """When OPG is worth more ($/token), fewer tokens are charged."""
        mock_urlopen.return_value = _make_urlopen_response(1000.0)
        _reset_price_cache()
        cost_cheap = dynamic_session_cost_calculator(self._make_context())

        mock_urlopen.return_value = _make_urlopen_response(5000.0)
        _reset_price_cache()
        cost_expensive = dynamic_session_cost_calculator(self._make_context())

        self.assertGreater(cost_cheap, cost_expensive)

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_more_tokens_increases_cost(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(3000.0)
        cost_small = dynamic_session_cost_calculator(
            self._make_context(prompt_tokens=10, completion_tokens=5)
        )
        _reset_price_cache()
        mock_urlopen.return_value = _make_urlopen_response(3000.0)
        cost_large = dynamic_session_cost_calculator(
            self._make_context(prompt_tokens=1000, completion_tokens=500)
        )
        self.assertGreater(cost_large, cost_small)

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_cost_scales_correctly(self, mock_urlopen):
        """Spot-check the math for gpt-4.1 at a known token price of $3 000.

        gpt-4.1 input: $0.000002/token, output: $0.000008/token
        100 input + 50 output = $0.0002 + $0.0004 = $0.0006 USD
        At token price = $3 000: 0.0006 / 3000 = 0.0000002 tokens
        In smallest units (10^18 decimals): 200_000_000_000
        """
        mock_urlopen.return_value = _make_urlopen_response(3000.0)
        cost = dynamic_session_cost_calculator(
            self._make_context(model="gpt-4.1", prompt_tokens=100, completion_tokens=50)
        )
        self.assertEqual(cost, 200_000_000_000)

    @patch("tee_gateway.util.urllib.request.urlopen")
    def test_zero_tokens_returns_zero(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(3000.0)
        cost = dynamic_session_cost_calculator(
            self._make_context(prompt_tokens=0, completion_tokens=0)
        )
        self.assertEqual(cost, 0)

    def test_raises_on_unknown_model(self):
        ctx = self._make_context(model="not-a-real-model")
        with self.assertRaises(ValueError):
            dynamic_session_cost_calculator(ctx)

    def test_raises_when_usage_missing(self):
        ctx = {
            "request_json": {"model": "gpt-4.1"},
            "response_json": {},
            "payment_requirements": {"asset": "0x240b09731D96979f50B2C649C9CE10FcF9C7987F"},
        }
        with self.assertRaises(ValueError):
            dynamic_session_cost_calculator(ctx)

    def test_raises_when_request_json_missing(self):
        with self.assertRaises(ValueError):
            dynamic_session_cost_calculator(
                {"response_json": {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}}
            )


if __name__ == "__main__":
    unittest.main()
