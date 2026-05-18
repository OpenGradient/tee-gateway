from .config import PriceFeedConfig
from .feed import OPGPriceFeed

__all__ = [
    "OPGPriceFeed",
    "PriceFeedConfig",
    "get_price_feed",
    "set_price_feed",
]


_price_feed: OPGPriceFeed | None = None


def set_price_feed(feed: OPGPriceFeed) -> None:
    """Register the process-wide OPG price feed. Called once from app startup."""
    global _price_feed
    _price_feed = feed


def get_price_feed() -> OPGPriceFeed:
    """Return the registered process-wide price feed. Raises if not initialized."""
    if _price_feed is None:
        raise RuntimeError(
            "OPG price feed not initialized — set_price_feed() must be called "
            "during app startup before pricing-dependent code runs."
        )
    return _price_feed
