#!/usr/bin/env python3
"""
Unit tests for OPG token price feed functionality.
Tests CoinGecko integration, custom price feeds, and fallback behavior.
"""

import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock
import requests

# Import functions under test
from tee_gateway.util import (
    _fetch_price_from_coingecko,
    _fetch_price_from_custom_feed,
    _fetch_opg_price_usd,
    get_token_a_price_usd,
    _token_price_cache,
    TOKEN_A_PRICE_CACHE_TTL_SECONDS,
)


class TestCoinGeckoPriceFeed:
    """Tests for CoinGecko API integration."""

    def test_successful_price_fetch(self):
        """Test successful price retrieval from CoinGecko."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "0x240b09731d96979f50b2c649c9ce10fcf9c7987f": {"usd": 0.15}
        }
        mock_response.raise_for_status = MagicMock()

        with patch("tee_gateway.util.requests.get", return_value=mock_response):
            price = _fetch_price_from_coingecko()
            assert price == Decimal("0.15")

    def test_no_price_data_returns_none(self):
        """Test returns None when CoinGecko has no data for the token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        with patch("tee_gateway.util.requests.get", return_value=mock_response):
            price = _fetch_price_from_coingecko()
            assert price is None

    def test_api_error_returns_none(self):
        """Test returns None on API request failure."""
        with patch(
            "tee_gateway.util.requests.get",
            side_effect=requests.exceptions.Timeout("Connection timeout"),
        ):
            price = _fetch_price_from_coingecko()
            assert price is None

    def test_invalid_json_returns_none(self):
        """Test returns None on invalid JSON response."""
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = MagicMock()

        with patch("tee_gateway.util.requests.get", return_value=mock_response):
            price = _fetch_price_from_coingecko()
            assert price is None

    def test_zero_price_returns_none(self):
        """Test returns None when price is zero (invalid)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "0x240b09731d96979f50b2c649c9ce10fcf9c7987f": {"usd": 0}
        }
        mock_response.raise_for_status = MagicMock()

        with patch("tee_gateway.util.requests.get", return_value=mock_response):
            price = _fetch_price_from_coingecko()
            assert price is None


class TestCustomPriceFeed:
    """Tests for custom price feed endpoint integration."""

    def test_no_url_configured_returns_none(self):
        """Test returns None when no custom URL is configured."""
        with patch("tee_gateway.util.OPG_PRICE_FEED_URL", None):
            price = _fetch_price_from_custom_feed()
            assert price is None

    def test_successful_price_fetch(self):
        """Test successful price retrieval from custom endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"price": 0.25}
        mock_response.raise_for_status = MagicMock()

        with (
            patch("tee_gateway.util.OPG_PRICE_FEED_URL", "https://example.com/price"),
            patch("tee_gateway.util.requests.get", return_value=mock_response),
        ):
            price = _fetch_price_from_custom_feed()
            assert price == Decimal("0.25")

    def test_supports_usd_key(self):
        """Test supports 'usd' key in response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"usd": 0.30}
        mock_response.raise_for_status = MagicMock()

        with (
            patch("tee_gateway.util.OPG_PRICE_FEED_URL", "https://example.com/price"),
            patch("tee_gateway.util.requests.get", return_value=mock_response),
        ):
            price = _fetch_price_from_custom_feed()
            assert price == Decimal("0.30")

    def test_supports_value_key(self):
        """Test supports 'value' key in response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"value": "0.35"}
        mock_response.raise_for_status = MagicMock()

        with (
            patch("tee_gateway.util.OPG_PRICE_FEED_URL", "https://example.com/price"),
            patch("tee_gateway.util.requests.get", return_value=mock_response),
        ):
            price = _fetch_price_from_custom_feed()
            assert price == Decimal("0.35")

    def test_api_error_returns_none(self):
        """Test returns None on API request failure."""
        with (
            patch("tee_gateway.util.OPG_PRICE_FEED_URL", "https://example.com/price"),
            patch(
                "tee_gateway.util.requests.get",
                side_effect=requests.exceptions.ConnectionError("Network error"),
            ),
        ):
            price = _fetch_price_from_custom_feed()
            assert price is None


class TestOPGPriceFeed:
    """Tests for the main _fetch_opg_price_usd function."""

    def test_custom_feed_takes_priority(self):
        """Test custom price feed is tried first."""
        with (
            patch(
                "tee_gateway.util._fetch_price_from_custom_feed",
                return_value=Decimal("0.20"),
            ),
            patch(
                "tee_gateway.util._fetch_price_from_coingecko",
                return_value=Decimal("0.15"),
            ),
        ):
            price = _fetch_opg_price_usd()
            assert price == Decimal("0.20")

    def test_coingecko_used_when_custom_fails(self):
        """Test CoinGecko is used when custom feed returns None."""
        with (
            patch("tee_gateway.util._fetch_price_from_custom_feed", return_value=None),
            patch(
                "tee_gateway.util._fetch_price_from_coingecko",
                return_value=Decimal("0.15"),
            ),
        ):
            price = _fetch_opg_price_usd()
            assert price == Decimal("0.15")

    def test_static_price_used_when_apis_fail(self):
        """Test static price from env is used when APIs fail."""
        with (
            patch("tee_gateway.util._fetch_price_from_custom_feed", return_value=None),
            patch("tee_gateway.util._fetch_price_from_coingecko", return_value=None),
            patch("tee_gateway.util.OPG_STATIC_PRICE_USD", "0.10"),
        ):
            price = _fetch_opg_price_usd()
            assert price == Decimal("0.10")

    def test_fallback_to_one_when_all_fail(self):
        """Test falls back to 1:1 when all sources fail."""
        with (
            patch("tee_gateway.util._fetch_price_from_custom_feed", return_value=None),
            patch("tee_gateway.util._fetch_price_from_coingecko", return_value=None),
            patch("tee_gateway.util.OPG_STATIC_PRICE_USD", None),
        ):
            price = _fetch_opg_price_usd()
            assert price == Decimal("1")

    def test_invalid_static_price_falls_back(self):
        """Test invalid static price falls back to 1:1."""
        with (
            patch("tee_gateway.util._fetch_price_from_custom_feed", return_value=None),
            patch("tee_gateway.util._fetch_price_from_coingecko", return_value=None),
            patch("tee_gateway.util.OPG_STATIC_PRICE_USD", "not-a-number"),
        ):
            price = _fetch_opg_price_usd()
            assert price == Decimal("1")


class TestPriceCaching:
    """Tests for price caching behavior."""

    def test_cache_returns_cached_value(self):
        """Test cached value is returned within TTL."""
        # Reset cache
        _token_price_cache["value"] = Decimal("0.50")
        _token_price_cache["updated_at"] = 9999999999.0  # Far future

        with patch("tee_gateway.util.time.time", return_value=9999999999.0):
            price = get_token_a_price_usd()
            assert price == Decimal("0.50")

    def test_cache_refresh_when_expired(self):
        """Test cache is refreshed when TTL expired."""
        # Set expired cache
        _token_price_cache["value"] = Decimal("0.50")
        _token_price_cache["updated_at"] = 0.0

        with (
            patch("tee_gateway.util.time.time", return_value=1000.0),
            patch(
                "tee_gateway.util._fetch_opg_price_usd", return_value=Decimal("0.75")
            ),
        ):
            price = get_token_a_price_usd()
            assert price == Decimal("0.75")
            assert _token_price_cache["value"] == Decimal("0.75")
