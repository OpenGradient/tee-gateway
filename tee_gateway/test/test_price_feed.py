"""
Unit tests for tee_gateway.price_feed.

All external HTTP calls are mocked — no network access required.

Test classes
------------
TestFetchOPGPrice        — the raw fetch_opg_price() helper in feed.py
TestOPGPriceFeedRefresh  — OPGPriceFeed._refresh_price() (retry, rate-limit, stats)
TestOPGPriceFeedGetPrice — OPGPriceFeed.get_price() (stale warning, ValueError before fetch)
TestOPGPriceFeedStatus   — OPGPriceFeed.get_status() snapshots
TestModuleLevelFunctions — start_price_feed() / get_opg_price_usd() / get_price_feed_status()
"""

import time
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

import requests

from tee_gateway.definitions import BASE_MAINNET_OPG_ADDRESS
from tee_gateway.price_feed import (
    OPGPriceFeed,
    get_opg_price_usd,
    get_price_feed_status,
)
from tee_gateway.price_feed.feed import fetch_opg_price

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPG_ADDRESS_LOWER = BASE_MAINNET_OPG_ADDRESS.lower()
SAMPLE_PRICE = Decimal("0.042")
SAMPLE_PRICE_FLOAT = 0.042

# Patch target prefix — all mocks go through the feed module.
_FEED = "tee_gateway.price_feed.feed"


def _mock_response(status_code: int = 200, json_body: dict | None = None) -> MagicMock:
    """Build a minimal mock requests.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_body or {}
    if status_code >= 400:
        http_err = requests.exceptions.HTTPError(response=mock)
        mock.raise_for_status.side_effect = http_err
    else:
        mock.raise_for_status.return_value = None
    return mock


def _coingecko_success_body() -> dict:
    return {OPG_ADDRESS_LOWER: {"usd": SAMPLE_PRICE_FLOAT}}


# ---------------------------------------------------------------------------
# TestFetchOPGPrice
# ---------------------------------------------------------------------------


class TestFetchOPGPrice(unittest.TestCase):
    """Tests for the fetch_opg_price() free function in feed.py."""

    @patch(f"{_FEED}.requests.get")
    def test_happy_path_returns_decimal(self, mock_get):
        mock_get.return_value = _mock_response(200, _coingecko_success_body())
        price = fetch_opg_price()
        self.assertIsInstance(price, Decimal)
        self.assertEqual(price, Decimal(str(SAMPLE_PRICE_FLOAT)))

    @patch(f"{_FEED}.requests.get")
    def test_passes_correct_params(self, mock_get):
        mock_get.return_value = _mock_response(200, _coingecko_success_body())
        fetch_opg_price()
        _, kwargs = mock_get.call_args
        self.assertIn("contract_addresses", kwargs["params"])
        self.assertEqual(kwargs["params"]["vs_currencies"], "usd")
        self.assertIn(
            "base", kwargs["url"] if "url" in kwargs else mock_get.call_args[0][0]
        )

    @patch(f"{_FEED}.requests.get")
    def test_raises_on_http_500(self, mock_get):
        mock_get.return_value = _mock_response(500)
        with self.assertRaises(requests.exceptions.HTTPError):
            fetch_opg_price()

    @patch(f"{_FEED}.requests.get")
    def test_raises_on_http_429(self, mock_get):
        mock_get.return_value = _mock_response(429)
        with self.assertRaises(requests.exceptions.HTTPError) as ctx:
            fetch_opg_price()
        self.assertEqual(ctx.exception.response.status_code, 429)

    @patch(f"{_FEED}.requests.get")
    def test_raises_on_empty_response_body(self, mock_get):
        mock_get.return_value = _mock_response(200, {})
        with self.assertRaises(ValueError, msg="should raise when address key absent"):
            fetch_opg_price()

    @patch(f"{_FEED}.requests.get")
    def test_raises_when_usd_key_missing(self, mock_get):
        mock_get.return_value = _mock_response(200, {OPG_ADDRESS_LOWER: {"eur": 0.04}})
        with self.assertRaises(ValueError):
            fetch_opg_price()

    @patch(f"{_FEED}.requests.get")
    def test_raises_on_network_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("timeout")
        with self.assertRaises(requests.exceptions.ConnectionError):
            fetch_opg_price()


# ---------------------------------------------------------------------------
# TestOPGPriceFeedRefresh
# ---------------------------------------------------------------------------


class TestOPGPriceFeedRefresh(unittest.TestCase):
    """Tests for OPGPriceFeed._refresh_price() — retry logic, rate-limit, stats."""

    def _feed(self, **kwargs) -> OPGPriceFeed:
        defaults = {"refresh_interval": 300, "max_retries": 3, "retry_delay": 0}
        defaults.update(kwargs)
        return OPGPriceFeed(**defaults)

    @patch(f"{_FEED}.fetch_opg_price")
    def test_successful_refresh_sets_price(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_PRICE
        feed = self._feed()
        feed._refresh_price()
        self.assertEqual(feed._price, SAMPLE_PRICE)

    @patch(f"{_FEED}.fetch_opg_price")
    def test_successful_refresh_updates_stats(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_PRICE
        feed = self._feed()
        feed._refresh_price()
        self.assertEqual(feed.total_fetches, 1)
        self.assertEqual(feed.total_errors, 0)
        self.assertEqual(feed.consecutive_failures, 0)
        self.assertIsNotNone(feed.last_success)

    @patch(f"{_FEED}.fetch_opg_price")
    def test_retry_on_transient_failure_then_success(self, mock_fetch):
        mock_fetch.side_effect = [
            ValueError("transient"),
            ValueError("transient"),
            SAMPLE_PRICE,
        ]
        feed = self._feed(max_retries=3, retry_delay=0)
        feed._refresh_price()
        self.assertEqual(feed._price, SAMPLE_PRICE)
        self.assertEqual(mock_fetch.call_count, 3)
        self.assertEqual(feed.total_fetches, 1)
        self.assertEqual(feed.total_errors, 0)

    @patch(f"{_FEED}.fetch_opg_price")
    def test_exhausted_retries_records_error_stats(self, mock_fetch):
        mock_fetch.side_effect = ValueError("always fails")
        feed = self._feed(max_retries=3, retry_delay=0)
        feed._refresh_price()
        self.assertEqual(feed.total_errors, 1)
        self.assertEqual(feed.consecutive_failures, 1)
        self.assertIsNotNone(feed.last_error)
        self.assertEqual(feed.total_fetches, 0)

    @patch(f"{_FEED}.fetch_opg_price")
    def test_exhausted_retries_keeps_last_known_price(self, mock_fetch):
        feed = self._feed(max_retries=2, retry_delay=0)
        feed._price = SAMPLE_PRICE
        feed.last_success = time.time()
        mock_fetch.side_effect = ValueError("fail")
        feed._refresh_price()
        self.assertEqual(feed._price, SAMPLE_PRICE)

    @patch(f"{_FEED}.fetch_opg_price")
    def test_success_after_failures_resets_consecutive_failures(self, mock_fetch):
        feed = self._feed(max_retries=1, retry_delay=0)
        mock_fetch.side_effect = ValueError("fail")
        feed._refresh_price()
        self.assertEqual(feed.consecutive_failures, 1)
        mock_fetch.side_effect = None
        mock_fetch.return_value = SAMPLE_PRICE
        feed._refresh_price()
        self.assertEqual(feed.consecutive_failures, 0)

    @patch(f"{_FEED}.fetch_opg_price")
    def test_rate_limit_breaks_retry_loop_immediately(self, mock_fetch):
        resp = MagicMock()
        resp.status_code = 429
        mock_fetch.side_effect = requests.exceptions.HTTPError(response=resp)
        feed = self._feed(max_retries=3, retry_delay=0)
        feed._refresh_price()
        self.assertEqual(mock_fetch.call_count, 1)
        self.assertEqual(feed.total_errors, 1)

    @patch(f"{_FEED}.time.sleep")
    @patch(f"{_FEED}.fetch_opg_price")
    def test_retry_delay_called_between_attempts(self, mock_fetch, mock_sleep):
        mock_fetch.side_effect = [ValueError("fail"), ValueError("fail"), SAMPLE_PRICE]
        feed = self._feed(max_retries=3, retry_delay=5)
        feed._refresh_price()
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_called_with(5)

    @patch(f"{_FEED}.time.sleep")
    @patch(f"{_FEED}.fetch_opg_price")
    def test_no_sleep_after_last_failed_attempt(self, mock_fetch, mock_sleep):
        mock_fetch.side_effect = ValueError("always fails")
        feed = self._feed(max_retries=3, retry_delay=5)
        feed._refresh_price()
        self.assertEqual(mock_sleep.call_count, 2)


# ---------------------------------------------------------------------------
# TestOPGPriceFeedGetPrice
# ---------------------------------------------------------------------------


class TestOPGPriceFeedGetPrice(unittest.TestCase):
    """Tests for OPGPriceFeed.get_price() behaviour."""

    def test_raises_before_any_successful_fetch(self):
        feed = OPGPriceFeed()
        with self.assertRaises(ValueError) as ctx:
            feed.get_price()
        self.assertIn("not yet available", str(ctx.exception))

    @patch(f"{_FEED}.fetch_opg_price")
    def test_returns_price_after_successful_refresh(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_PRICE
        feed = OPGPriceFeed(retry_delay=0)
        feed._refresh_price()
        self.assertEqual(feed.get_price(), SAMPLE_PRICE)

    @patch(f"{_FEED}.time.time")
    @patch(f"{_FEED}.fetch_opg_price")
    def test_warns_when_price_is_stale(self, mock_fetch, mock_time):
        mock_fetch.return_value = SAMPLE_PRICE
        feed = OPGPriceFeed(refresh_interval=300, retry_delay=0)

        mock_time.return_value = 0.0
        feed._refresh_price()

        # Advance past stale threshold (300 * 2 = 600s)
        mock_time.return_value = 601.0

        with self.assertLogs("llm_server.price_feed", level="WARNING") as log_ctx:
            price = feed.get_price()

        self.assertEqual(price, SAMPLE_PRICE)
        self.assertTrue(any("stale" in line.lower() for line in log_ctx.output))

    @patch(f"{_FEED}.time.time")
    @patch(f"{_FEED}.fetch_opg_price")
    def test_no_stale_warning_when_price_is_fresh(self, mock_fetch, mock_time):
        import logging

        mock_fetch.return_value = SAMPLE_PRICE
        feed = OPGPriceFeed(refresh_interval=300, retry_delay=0)

        mock_time.return_value = 0.0
        feed._refresh_price()
        mock_time.return_value = 100.0  # well within threshold

        with self.assertLogs("llm_server.price_feed", level="DEBUG") as log_ctx:
            logging.getLogger("llm_server.price_feed").debug("sentinel")
            feed.get_price()

        warning_lines = [
            line
            for line in log_ctx.output
            if "WARNING" in line and "stale" in line.lower()
        ]
        self.assertEqual(warning_lines, [])


# ---------------------------------------------------------------------------
# TestOPGPriceFeedStatus
# ---------------------------------------------------------------------------


class TestOPGPriceFeedStatus(unittest.TestCase):
    """Tests for OPGPriceFeed.get_status() snapshot."""

    def test_initial_status_has_no_price(self):
        feed = OPGPriceFeed()
        status = feed.get_status()
        self.assertIsNone(status["price_usd"])
        self.assertIsNone(status["last_success"])
        self.assertEqual(status["consecutive_failures"], 0)
        self.assertEqual(status["total_fetches"], 0)
        self.assertEqual(status["total_errors"], 0)

    @patch(f"{_FEED}.fetch_opg_price")
    def test_status_reflects_successful_fetch(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_PRICE
        feed = OPGPriceFeed(retry_delay=0)
        feed._refresh_price()
        status = feed.get_status()
        self.assertAlmostEqual(status["price_usd"], float(SAMPLE_PRICE), places=6)
        self.assertIsNotNone(status["last_success"])
        self.assertEqual(status["total_fetches"], 1)
        self.assertEqual(status["consecutive_failures"], 0)

    @patch(f"{_FEED}.fetch_opg_price")
    def test_status_reflects_failed_cycle(self, mock_fetch):
        mock_fetch.side_effect = ValueError("fail")
        feed = OPGPriceFeed(max_retries=1, retry_delay=0)
        feed._refresh_price()
        status = feed.get_status()
        self.assertIsNone(status["price_usd"])
        self.assertEqual(status["total_errors"], 1)
        self.assertEqual(status["consecutive_failures"], 1)
        self.assertIsNotNone(status["last_error"])

    def test_status_includes_refresh_interval(self):
        feed = OPGPriceFeed(refresh_interval=600)
        self.assertEqual(feed.get_status()["refresh_interval"], 600)

    @patch(f"{_FEED}.fetch_opg_price")
    def test_status_accumulates_multiple_error_cycles(self, mock_fetch):
        mock_fetch.side_effect = ValueError("fail")
        feed = OPGPriceFeed(max_retries=1, retry_delay=0)
        feed._refresh_price()
        feed._refresh_price()
        feed._refresh_price()
        status = feed.get_status()
        self.assertEqual(status["total_errors"], 3)
        self.assertEqual(status["consecutive_failures"], 3)


# ---------------------------------------------------------------------------
# TestModuleLevelFunctions
# ---------------------------------------------------------------------------


class TestModuleLevelFunctions(unittest.TestCase):
    """Tests for the module-level singleton helpers."""

    def test_get_opg_price_usd_raises_when_feed_is_none(self):
        with patch(f"{_FEED}._feed", None):
            with self.assertRaises(ValueError) as ctx:
                get_opg_price_usd()
        self.assertIn("not been started", str(ctx.exception))

    def test_get_opg_price_usd_delegates_to_feed(self):
        mock_feed = MagicMock()
        mock_feed.get_price.return_value = SAMPLE_PRICE
        with patch(f"{_FEED}._feed", mock_feed):
            price = get_opg_price_usd()
        self.assertEqual(price, SAMPLE_PRICE)
        mock_feed.get_price.assert_called_once()

    def test_get_price_feed_status_when_feed_is_none(self):
        with patch(f"{_FEED}._feed", None):
            status = get_price_feed_status()
        self.assertEqual(status, {"status": "not_started"})

    def test_get_price_feed_status_delegates_to_feed(self):
        expected = {"price_usd": 0.042, "total_fetches": 5}
        mock_feed = MagicMock()
        mock_feed.get_status.return_value = expected
        with patch(f"{_FEED}._feed", mock_feed):
            status = get_price_feed_status()
        self.assertEqual(status, expected)
        mock_feed.get_status.assert_called_once()


if __name__ == "__main__":
    unittest.main()
