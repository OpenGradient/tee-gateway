"""
Runtime configuration objects for the TEE gateway.

These are populated once at startup via POST /v1/keys and passed
explicitly to the subsystems that need them, rather than being stored
as environment variables.
"""

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# OPG / token price feed
# ---------------------------------------------------------------------------

# How long (seconds) to reuse a cached price before fetching a fresh one.
# At 120 s the gateway makes at most 30 CoinGecko calls/hour — well within
# the free-tier limit (30/min).
OPG_PRICE_CACHE_TTL_SECONDS: int = 120

# Number of times to retry a failed CoinGecko fetch before giving up.
# Each attempt uses the same 5-second timeout; retries are immediate (no backoff).
OPG_PRICE_FETCH_RETRIES: int = 3

# CoinGecko coin ID for the OPG token.
# https://www.coingecko.com/en/coins/opengradient
OPG_PRICE_COINGECKO_ID: str = "opengradient"

# Sanity bounds for the fetched token price.
# Used in integration tests to catch obviously wrong API responses
# (wrong currency, implausibly large value).
# Update when OPG establishes a trading range.
OPG_PRICE_SANITY_MAX_USD: str = (
    "1000000"  # $1 000 000 — rules out obviously corrupt data
)

# ---------------------------------------------------------------------------
# Heartbeat defaults
# ---------------------------------------------------------------------------
DEFAULT_HEARTBEAT_INTERVAL = 900  # 15 minutes
DEFAULT_HEARTBEAT_BUFFER = (
    300  # 5 minutes — subtracted from time.time() to compensate for enclave clock drift
)
DEFAULT_FACILITATOR_TIMEOUT = 20  # seconds
HEARTBEAT_MAX_RETRIES = 3
HEARTBEAT_RETRY_DELAY = 10  # seconds


@dataclass(frozen=True)
class ProviderConfig:
    """API keys for each supported LLM provider."""

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None


@dataclass(frozen=True)
class HeartbeatConfig:
    """Configuration for the on-chain TEE heartbeat service."""

    contract_address: str
    facilitator_url: str
    interval: int = DEFAULT_HEARTBEAT_INTERVAL
    timestamp_buffer: int = DEFAULT_HEARTBEAT_BUFFER
    facilitator_timeout: int = DEFAULT_FACILITATOR_TIMEOUT
