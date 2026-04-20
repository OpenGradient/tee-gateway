"""
Background OPG/USD price feed using the CoinGecko public API.

Runs as a daemon thread that proactively refreshes the OPG token price at a
configurable interval, with retry on per-cycle fetch failure and early exit on
rate limiting.

Usage
-----
Create an ``OPGPriceFeed`` instance in the application entry point, call
``start()``, then pass it explicitly to wherever the price is needed (e.g.
``calculate_session_cost(...)`` in ``util.py``).
"""

import logging
import threading
import time
from decimal import Decimal
from typing import Any, Optional

import requests

from tee_gateway.definitions import BASE_MAINNET_OPG_ADDRESS
from tee_gateway.price_feed.config import (
    COINGECKO_BASE_URL,
    COINGECKO_PLATFORM,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REFRESH_INTERVAL,
    DEFAULT_RETRY_DELAY,
    FETCH_TIMEOUT,
    STALE_WARNING_MULTIPLIER,
)

logger = logging.getLogger("llm_server.price_feed")


class OPGPriceFeed:
    """Fetches and caches the OPG/USD price from CoinGecko in a background thread."""

    def __init__(
        self,
        refresh_interval: int = DEFAULT_REFRESH_INTERVAL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        self._refresh_interval = refresh_interval
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._price: Optional[Decimal] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # Status tracking — updated under _lock on every refresh cycle outcome.
        self.last_success: Optional[float] = None  # epoch seconds of last good fetch
        self.last_error: Optional[str] = None  # description of last failure (if any)
        self.consecutive_failures: int = 0  # reset to 0 on any successful fetch
        self.total_fetches: int = 0  # cumulative successful fetches
        self.total_errors: int = 0  # cumulative failed refresh cycles

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background refresh loop, including the initial price fetch.

        The initial fetch runs inside the background thread so startup is
        non-blocking.  ``get_price()`` will raise ``ValueError`` until the
        first fetch completes; any error propagates as HTTP 500 via the
        strict cost-resolution patch in ``__main__.py``.

        Idempotent — calling ``start()`` on an already-running feed is a no-op.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.info("OPG price feed already running, ignoring duplicate start()")
            return
        self._thread = threading.Thread(
            target=self._run_with_initial_fetch, name="opg-price-feed", daemon=True
        )
        self._thread.start()
        logger.info(
            "OPG price feed started (refresh_interval=%ds, max_retries=%d)",
            self._refresh_interval,
            self._max_retries,
        )

    def get_price(self) -> Decimal:
        """Return the latest cached OPG/USD price.

        Raises ``ValueError`` if no price has been successfully fetched yet.
        Logs a warning (but still returns the price) if the cached value is
        older than ``STALE_WARNING_MULTIPLIER * refresh_interval`` seconds —
        this indicates the background loop has missed at least one refresh
        cycle and may be experiencing persistent errors.
        """
        now = time.time()
        with self._lock:
            if self._price is None:
                raise ValueError(
                    "OPG price not yet available — "
                    "price feed has not completed a successful fetch"
                )
            if self.last_success is not None:
                age = now - self.last_success
                stale_threshold = self._refresh_interval * STALE_WARNING_MULTIPLIER
                if age > stale_threshold:
                    logger.warning(
                        "OPG price data is stale: last successful fetch was %.0fs ago "
                        "(threshold: %.0fs); consecutive failures: %d",
                        age,
                        stale_threshold,
                        self.consecutive_failures,
                    )
            return self._price

    def get_status(self) -> dict[str, Any]:
        """Return a health snapshot suitable for logging or a /health endpoint."""
        with self._lock:
            return {
                "price_usd": float(self._price) if self._price is not None else None,
                "last_success": self.last_success,
                "last_error": self.last_error,
                "consecutive_failures": self.consecutive_failures,
                "total_fetches": self.total_fetches,
                "total_errors": self.total_errors,
                "refresh_interval": self._refresh_interval,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_with_initial_fetch(self) -> None:
        self._refresh_price()
        while True:
            time.sleep(self._refresh_interval)
            self._refresh_price()

    def _refresh_price(self) -> None:
        """Attempt to fetch a fresh price, retrying on transient failure.

        - On success: updates the cached price and resets ``consecutive_failures``.
        - On HTTP 429: logs a rate-limit warning and exits the retry loop early
          (no point hammering a rate-limited API).
        - On exhausted retries: increments ``consecutive_failures`` and retains
          the last known good price so live traffic is not disrupted by a
          transient CoinGecko outage.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                price = fetch_opg_price()
                with self._lock:
                    self._price = price
                    self.last_success = time.time()
                    self.last_error = None
                    self.consecutive_failures = 0
                    self.total_fetches += 1
                logger.info(
                    "OPG price updated: $%.6f USD (attempt %d/%d)",
                    float(price),
                    attempt,
                    self._max_retries,
                )
                return
            except requests.exceptions.HTTPError as exc:
                last_exc = exc
                status_code = (
                    exc.response.status_code if exc.response is not None else None
                )
                if status_code == 429:
                    logger.warning(
                        "CoinGecko rate limit hit (429) on attempt %d/%d; "
                        "skipping remaining retries for this cycle",
                        attempt,
                        self._max_retries,
                    )
                    break
                logger.warning(
                    "OPG price fetch attempt %d/%d failed (HTTP %s): %s",
                    attempt,
                    self._max_retries,
                    status_code,
                    exc,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "OPG price fetch attempt %d/%d failed: %s",
                    attempt,
                    self._max_retries,
                    exc,
                )

            if attempt < self._max_retries:
                time.sleep(self._retry_delay)

        # All attempts exhausted (or rate-limited out) — record the failure.
        with self._lock:
            self.total_errors += 1
            self.consecutive_failures += 1
            self.last_error = str(last_exc) if last_exc is not None else "unknown error"

        logger.error(
            "OPG price refresh failed (consecutive failures: %d); "
            "retaining last known price (%s)",
            self.consecutive_failures,
            self._price,
        )


def fetch_opg_price() -> Decimal:
    """Fetch the current OPG/USD price from CoinGecko.  Raises on any error."""
    url = f"{COINGECKO_BASE_URL}/simple/token_price/{COINGECKO_PLATFORM}"
    params = {
        "contract_addresses": BASE_MAINNET_OPG_ADDRESS,
        "vs_currencies": "usd",
    }
    response = requests.get(url, params=params, timeout=FETCH_TIMEOUT)
    response.raise_for_status()

    data: dict = response.json()
    # CoinGecko keys the result by the lowercased contract address.
    price_entry = data.get(BASE_MAINNET_OPG_ADDRESS.lower())
    if not isinstance(price_entry, dict) or "usd" not in price_entry:
        raise ValueError(
            f"Unexpected CoinGecko response for {BASE_MAINNET_OPG_ADDRESS}: {data!r}"
        )

    return Decimal(str(price_entry["usd"]))
