"""
Unit tests for tee_gateway.price_feed and tee_gateway.util.make_cost_calculator.

All external HTTP calls are mocked — no network access required.

Test classes
------------
TestFetchOPGPrice        — the raw fetch_opg_price() helper in feed.py
TestOPGPriceFeedRefresh  — OPGPriceFeed._refresh_price() (retry, rate-limit, stats)
TestOPGPriceFeedGetPrice — OPGPriceFeed.get_price() (stale warning, ValueError before fetch)
TestOPGPriceFeedStatus   — OPGPriceFeed.get_status() snapshots
TestCalculateSessionCost — calculate_session_cost(context, get_price) in util.py
"""

import time
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

import requests

from tee_gateway.definitions import BASE_MAINNET_OPG_ADDRESS
from tee_gateway.price_feed import OPGPriceFeed
from tee_gateway.price_feed.feed import fetch_opg_price
from tee_gateway.util import calculate_session_cost

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
# TestMakeCostCalculator
# ---------------------------------------------------------------------------

_ASSET_ADDR = "0xdeadbeef"
_ASSET_ADDR_LOWER = _ASSET_ADDR.lower()
_ASSET_DECIMALS = 18


def _make_payment_requirements(asset: str = _ASSET_ADDR) -> dict:
    return {"asset": asset, "price": {"amount": "1000000000000000000", "asset": asset}}


def _make_context(
    model: str = "gpt-4.1-mini",
    input_tokens: int = 100,
    output_tokens: int = 50,
    price_usd: Decimal = Decimal("0.10"),
    asset: str = _ASSET_ADDR,
) -> dict:
    return {
        "request_json": {"model": model},
        "response_json": {
            "model": model,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            },
        },
        "payment_requirements": _make_payment_requirements(asset),
        "method": "POST",
        "path": "/v1/chat/completions",
        "status_code": 200,
        "is_streaming": False,
        "request_body_bytes": b"",
        "response_body_bytes": b"",
        "default_cost": 10**18,
    }


def _make_get_price(price_usd: Decimal = Decimal("0.10")) -> MagicMock:
    mock = MagicMock(return_value=price_usd)
    return mock


class TestCalculateSessionCost(unittest.TestCase):
    """Tests for calculate_session_cost(context, get_price)."""

    def _patch_definitions(self):
        return patch(
            "tee_gateway.util.ASSET_DECIMALS_BY_ADDRESS",
            {_ASSET_ADDR_LOWER: _ASSET_DECIMALS},
        )

    def _patch_model(
        self, input_price: str = "0.000001", output_price: str = "0.000002"
    ):
        cfg = MagicMock()
        cfg.input_price_usd = Decimal(input_price)
        cfg.output_price_usd = Decimal(output_price)
        return patch("tee_gateway.util.get_model_config", return_value=cfg)

    def test_calls_get_price(self):
        get_price = _make_get_price()
        with self._patch_definitions(), self._patch_model():
            calculate_session_cost(_make_context(), get_price)
        get_price.assert_called_once()

    def test_returns_positive_int(self):
        with self._patch_definitions(), self._patch_model():
            result = calculate_session_cost(_make_context(), _make_get_price())
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    def test_zero_tokens_returns_zero(self):
        with self._patch_definitions(), self._patch_model():
            result = calculate_session_cost(
                _make_context(input_tokens=0, output_tokens=0), _make_get_price()
            )
        self.assertEqual(result, 0)

    def test_raises_when_get_price_raises(self):
        get_price = MagicMock(side_effect=ValueError("price not available"))
        with self._patch_definitions(), self._patch_model():
            with self.assertRaises(ValueError):
                calculate_session_cost(_make_context(), get_price)

    def test_raises_when_non_positive_price(self):
        with self._patch_definitions(), self._patch_model():
            with self.assertRaises(ValueError):
                calculate_session_cost(_make_context(), _make_get_price(Decimal("0")))

    def test_raises_when_request_json_missing(self):
        ctx = _make_context()
        ctx["request_json"] = None
        with self._patch_definitions(), self._patch_model():
            with self.assertRaises(ValueError):
                calculate_session_cost(ctx, _make_get_price())

    def test_raises_when_usage_missing(self):
        ctx = _make_context()
        ctx["response_json"] = {"model": "gpt-4.1-mini"}
        with self._patch_definitions(), self._patch_model():
            with self.assertRaises(ValueError):
                calculate_session_cost(ctx, _make_get_price())

    def test_raises_when_asset_unknown(self):
        ctx = _make_context(asset="0xunknown")
        with (
            patch("tee_gateway.util.ASSET_DECIMALS_BY_ADDRESS", {}),
            self._patch_model(),
        ):
            with self.assertRaises(ValueError):
                calculate_session_cost(ctx, _make_get_price())

    def test_cost_scales_with_token_count(self):
        with self._patch_definitions(), self._patch_model():
            cost_small = calculate_session_cost(
                _make_context(input_tokens=10, output_tokens=5), _make_get_price()
            )
            cost_large = calculate_session_cost(
                _make_context(input_tokens=1000, output_tokens=500), _make_get_price()
            )
        self.assertGreater(cost_large, cost_small)

    def test_higher_token_price_yields_lower_cost(self):
        with self._patch_definitions(), self._patch_model():
            cost_cheap = calculate_session_cost(
                _make_context(), _make_get_price(Decimal("0.10"))
            )
            cost_expensive = calculate_session_cost(
                _make_context(), _make_get_price(Decimal("0.20"))
            )
        self.assertGreater(cost_cheap, cost_expensive)


if __name__ == "__main__":
    unittest.main()
