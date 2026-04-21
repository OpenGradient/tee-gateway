import datetime
import os

from tee_gateway import typing_utils
import logging
import requests
import threading
import time
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import Any

logger = logging.getLogger("llm_server.dynamic_pricing")

# ---------------------------------------------------------------------------
# OPG Price Feed Configuration
# ---------------------------------------------------------------------------
# CoinGecko API endpoint for Base network tokens
COINGECKO_BASE_API = "https://api.coingecko.com/api/v3/simple/token_price/base"
# OPG token contract address on Base (from definitions.py)
OPG_CONTRACT_ADDRESS = "0x240b09731D96979f50B2C649C9CE10FcF9C7987F"

# Environment variables for price feed configuration
# Set OPG_PRICE_FEED_URL to use a custom price feed endpoint (e.g., DEX oracle)
# Set OPG_STATIC_PRICE_USD to use a fixed price (e.g., "0.10" for $0.10)
OPG_PRICE_FEED_URL = os.getenv("OPG_PRICE_FEED_URL")
OPG_STATIC_PRICE_USD = os.getenv("OPG_STATIC_PRICE_USD")

# Timeout for external API calls (seconds)
PRICE_FEED_TIMEOUT = 5


def _deserialize(data, klass):
    """Deserializes dict, list, str into an object.

    :param data: dict, list or str.
    :param klass: class literal, or string of class name.

    :return: object.
    """
    if data is None:
        return None

    if klass in (int, float, str, bool, bytearray):
        return _deserialize_primitive(data, klass)
    elif klass is object:
        return _deserialize_object(data)
    elif klass == datetime.date:
        return deserialize_date(data)
    elif klass == datetime.datetime:
        return deserialize_datetime(data)
    elif typing_utils.is_generic(klass):
        if typing_utils.is_list(klass):
            return _deserialize_list(data, klass.__args__[0])
        if typing_utils.is_dict(klass):
            return _deserialize_dict(data, klass.__args__[1])
    else:
        return deserialize_model(data, klass)


def _deserialize_primitive(data, klass):
    """Deserializes to primitive type.

    :param data: data to deserialize.
    :param klass: class literal.

    :return: int, long, float, str, bool.
    :rtype: int | long | float | str | bool
    """
    try:
        value = klass(data)
    except UnicodeEncodeError:
        value = data
    except TypeError:
        value = data
    return value


def _deserialize_object(value):
    """Return an original value.

    :return: object.
    """
    return value


def deserialize_date(string):
    """Deserializes string to date.

    :param string: str.
    :type string: str
    :return: date.
    :rtype: date
    """
    if string is None:
        return None

    try:
        from dateutil.parser import parse  # type: ignore[import-untyped]

        return parse(string).date()
    except ImportError:
        return string


def deserialize_datetime(string):
    """Deserializes string to datetime.

    The string should be in iso8601 datetime format.

    :param string: str.
    :type string: str
    :return: datetime.
    :rtype: datetime
    """
    if string is None:
        return None

    try:
        from dateutil.parser import parse  # type: ignore[import-untyped]

        return parse(string)
    except ImportError:
        return string


def deserialize_model(data, klass):
    """Deserializes list or dict to model.

    :param data: dict, list.
    :type data: dict | list
    :param klass: class literal.
    :return: model object.
    """
    instance = klass()

    if not instance.openapi_types:
        return data

    for attr, attr_type in instance.openapi_types.items():
        if (
            data is not None
            and instance.attribute_map[attr] in data
            and isinstance(data, (list, dict))
        ):
            value = data[instance.attribute_map[attr]]
            setattr(instance, attr, _deserialize(value, attr_type))

    return instance


def _deserialize_list(data, boxed_type):
    """Deserializes a list and its elements.

    :param data: list to deserialize.
    :type data: list
    :param boxed_type: class literal.

    :return: deserialized list.
    :rtype: list
    """
    return [_deserialize(sub_data, boxed_type) for sub_data in data]


def _deserialize_dict(data, boxed_type):
    """Deserializes a dict and its elements.

    :param data: dict to deserialize.
    :type data: dict
    :param boxed_type: class literal.

    :return: deserialized dict.
    :rtype: dict
    """
    return {k: _deserialize(v, boxed_type) for k, v in data.items()}


from tee_gateway.definitions import (  # noqa: E402
    ASSET_DECIMALS_BY_ADDRESS,
)
from tee_gateway.model_registry import get_model_config  # noqa: E402

TOKEN_A_PRICE_CACHE_TTL_SECONDS = 60

_token_price_cache: dict[str, Any] = {
    "value": Decimal("1"),
    "updated_at": 0.0,
}
_token_price_lock = threading.Lock()


def _fetch_price_from_coingecko() -> Decimal | None:
    """Fetch OPG token price from CoinGecko API.

    Returns the USD price of OPG on Base network, or None if unavailable.
    """
    try:
        url = f"{COINGECKO_BASE_API}?contract_addresses={OPG_CONTRACT_ADDRESS}&vs_currencies=usd"
        response = requests.get(url, timeout=PRICE_FEED_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        # CoinGecko returns: {"0x...": {"usd": 0.123}}
        price_data = data.get(OPG_CONTRACT_ADDRESS.lower(), {})
        usd_price = price_data.get("usd")

        if usd_price is not None:
            price = Decimal(str(usd_price))
            if price > 0:
                logger.info("OPG price from CoinGecko: $%s", price)
                return price

        logger.warning("CoinGecko returned no price data for OPG")
        return None

    except requests.exceptions.RequestException as e:
        logger.warning("CoinGecko API request failed: %s", e)
        return None
    except (KeyError, ValueError, InvalidOperation) as e:
        logger.warning("Failed to parse CoinGecko response: %s", e)
        return None


def _fetch_price_from_custom_feed() -> Decimal | None:
    """Fetch OPG price from a custom price feed URL.

    The custom endpoint should return JSON with a 'price' field (USD value).
    Example response: {"price": 0.123} or {"price": "0.123"}
    """
    if not OPG_PRICE_FEED_URL:
        return None

    try:
        response = requests.get(OPG_PRICE_FEED_URL, timeout=PRICE_FEED_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        # Support both {"price": X} and {"usd": X} formats
        raw_price = data.get("price") or data.get("usd") or data.get("value")
        if raw_price is not None:
            price = Decimal(str(raw_price))
            if price > 0:
                logger.info("OPG price from custom feed: $%s", price)
                return price

        logger.warning("Custom price feed returned no valid price")
        return None

    except requests.exceptions.RequestException as e:
        logger.warning("Custom price feed request failed: %s", e)
        return None
    except (KeyError, ValueError, InvalidOperation) as e:
        logger.warning("Failed to parse custom price feed response: %s", e)
        return None


def _fetch_opg_price_usd() -> Decimal:
    """Fetch the USD price of OPG token for payment calculations.

    Tries multiple price sources in order:
    1. Custom price feed URL (if OPG_PRICE_FEED_URL is set)
    2. CoinGecko API (for mainnet tokens)
    3. Static price from environment (if OPG_STATIC_PRICE_USD is set)
    4. Fallback to 1:1 ratio with warning (suitable for USDC or testing)

    Returns:
        Decimal: The OPG/USD exchange rate (always positive).
    """
    # 1. Try custom price feed first (highest priority for operators)
    price = _fetch_price_from_custom_feed()
    if price is not None:
        return price

    # 2. Try CoinGecko API
    price = _fetch_price_from_coingecko()
    if price is not None:
        return price

    # 3. Use static price from environment if configured
    if OPG_STATIC_PRICE_USD:
        try:
            static_price = Decimal(OPG_STATIC_PRICE_USD)
            if static_price > 0:
                logger.info("Using static OPG price from env: $%s", static_price)
                return static_price
        except (ValueError, InvalidOperation):
            logger.error("Invalid OPG_STATIC_PRICE_USD value: %s", OPG_STATIC_PRICE_USD)

    # 4. Fallback to 1:1 — appropriate for USDC payments or testing
    logger.warning(
        "No OPG price available from any source. Using 1:1 fallback ratio. "
        "Set OPG_STATIC_PRICE_USD or OPG_PRICE_FEED_URL for accurate pricing."
    )
    return Decimal("1")


def get_token_a_price_usd() -> Decimal:
    """Get the cached USD price of the payment token.

    Returns the cached price if still valid (within TTL), otherwise fetches
    a fresh price from available sources. Thread-safe.
    """
    now = time.time()
    with _token_price_lock:
        cached_value = _token_price_cache.get("value")
        cached_at = float(_token_price_cache.get("updated_at") or 0.0)
        if (
            isinstance(cached_value, Decimal)
            and (now - cached_at) < TOKEN_A_PRICE_CACHE_TTL_SECONDS
        ):
            return cached_value

        value = _fetch_opg_price_usd()
        _token_price_cache["value"] = value
        _token_price_cache["updated_at"] = now
        return value


def _as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(by_alias=True, exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return None


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _normalize_model_name(model: str | None) -> str | None:
    if not model:
        return None
    return str(model).strip().lower()


def _extract_usage_tokens(
    response_json: dict[str, Any] | None,
) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from response JSON.

    Raises ValueError if usage data is missing or malformed — no silent fallback.
    """
    if not isinstance(response_json, dict):
        raise ValueError("response_json is not a dict; cannot extract usage tokens")
    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        raise ValueError(
            "response_json has no 'usage' dict; cannot extract usage tokens"
        )

    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if prompt_tokens is None or completion_tokens is None:
        raise ValueError(f"usage dict is missing token counts: {usage!r}")

    try:
        return max(0, int(prompt_tokens)), max(0, int(completion_tokens))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse token counts from usage: {usage!r}") from exc


def _extract_model_from_context(
    request_json: dict[str, Any] | None,
    response_json: dict[str, Any] | None,
) -> str:
    """Extract and normalize model name from request JSON.

    Uses only the request model name — the response model field is ignored
    because providers may return a versioned alias that differs from the
    user-facing name.  Raises ValueError if the model name is absent.
    """
    if not isinstance(request_json, dict):
        raise ValueError("request_json is not a dict; cannot extract model name")
    req_model = request_json.get("model")
    if not req_model:
        raise ValueError("request_json has no 'model' field")
    normalized = _normalize_model_name(req_model)
    if not normalized:
        raise ValueError(f"model name normalizes to empty string: {req_model!r}")
    return normalized


def _extract_asset_decimals_from_requirements(payment_requirements: Any) -> int:
    req = _as_dict(payment_requirements) or {}

    asset = req.get("asset")
    if not asset and isinstance(req.get("price"), dict):
        asset = req["price"].get("asset")

    if not isinstance(asset, str) or not asset:
        raise ValueError(
            f"payment_requirements has no recognizable asset address; "
            f"cannot determine token decimals: {req!r}"
        )

    asset_lower = asset.lower()
    if asset_lower not in ASSET_DECIMALS_BY_ADDRESS:
        raise ValueError(
            f"Unknown asset address {asset!r}; not in ASSET_DECIMALS_BY_ADDRESS. "
            f"Add it to definitions.py before accepting payments with this token."
        )
    return ASSET_DECIMALS_BY_ADDRESS[asset_lower]


def dynamic_session_cost_calculator(context: dict[str, Any]) -> int:
    """Compute UPTO per-request cost in token smallest units from actual usage.

    Raises ValueError on any missing or unrecognised input — no silent fallback.
    """
    request_json = context.get("request_json")
    response_json = context.get("response_json")

    if not isinstance(request_json, dict) or not isinstance(response_json, dict):
        raise ValueError(
            "dynamic_session_cost_calculator requires both request_json and response_json"
        )

    model = _extract_model_from_context(request_json, response_json)

    # get_model_config raises ValueError for unknown models — no fallback
    cfg = get_model_config(model)

    input_tokens, output_tokens = _extract_usage_tokens(response_json)

    input_rate = cfg.input_price_usd
    output_rate = cfg.output_price_usd

    total_usd = (Decimal(input_tokens) * input_rate) + (
        Decimal(output_tokens) * output_rate
    )
    token_price_usd = get_token_a_price_usd()
    if token_price_usd <= 0:
        raise ValueError(f"Token A price is non-positive: {token_price_usd}")

    token_amount = total_usd / token_price_usd
    decimals = _extract_asset_decimals_from_requirements(
        context.get("payment_requirements")
    )
    scale = Decimal(10) ** decimals
    cost_smallest_units = int(
        (token_amount * scale).to_integral_value(rounding=ROUND_CEILING)
    )

    logger.info(
        "DYNAMIC_SESSION_COST model=%s input_tokens=%d output_tokens=%d total_usd=%s token_price_usd=%s decimals=%d cost=%d",
        model,
        input_tokens,
        output_tokens,
        str(total_usd),
        str(token_price_usd),
        decimals,
        cost_smallest_units,
    )
    return max(0, cost_smallest_units)
