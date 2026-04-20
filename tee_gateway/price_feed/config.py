"""
Configuration constants and dataclass for the OPG price feed.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal


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
# TGE (Token Generation Event) fallback
# ---------------------------------------------------------------------------
# Before the TGE cutover, OPG is not yet listed on CoinGecko.  Return a fixed
# fallback price so inference requests can be priced immediately at launch.
# After the cutover, the live CoinGecko price is used.
TGE_CUTOVER_UTC = datetime(2026, 4, 21, 12, 30, 0, tzinfo=timezone.utc)
TGE_FALLBACK_PRICE_USD = Decimal("0.10")

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
