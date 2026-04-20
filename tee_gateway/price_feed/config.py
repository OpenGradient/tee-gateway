"""
Configuration constants and dataclass for the OPG price feed.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# CoinGecko API
# ---------------------------------------------------------------------------
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_PLATFORM = "base"  # Base mainnet platform identifier on CoinGecko
FETCH_TIMEOUT = 10  # seconds per HTTP request

# ---------------------------------------------------------------------------
# Refresh / retry defaults
# ---------------------------------------------------------------------------
DEFAULT_REFRESH_INTERVAL = 300  # 5 minutes between background refresh cycles
DEFAULT_MAX_RETRIES = 3  # attempts per refresh cycle before giving up
DEFAULT_RETRY_DELAY = 10  # seconds between retry attempts within a cycle

# ---------------------------------------------------------------------------
# Stale-price warning threshold
# ---------------------------------------------------------------------------
# get_price() logs WARNING when last successful fetch is older than
# STALE_WARNING_MULTIPLIER × refresh_interval seconds.
STALE_WARNING_MULTIPLIER = 2


@dataclass(frozen=True)
class PriceFeedConfig:
    """Runtime configuration for the OPG price feed background service."""

    refresh_interval: int = DEFAULT_REFRESH_INTERVAL
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
