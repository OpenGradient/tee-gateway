import datetime

import typing
from tee_gateway import typing_utils
import logging
import os
import sys
import threading
import time
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import Any

logger = logging.getLogger("llm_server.dynamic_pricing")


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
    elif klass == object:
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
        from dateutil.parser import parse
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
        from dateutil.parser import parse
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
        if data is not None \
                and instance.attribute_map[attr] in data \
                and isinstance(data, (list, dict)):
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
    return [_deserialize(sub_data, boxed_type)
            for sub_data in data]


def _deserialize_dict(data, boxed_type):
    """Deserializes a dict and its elements.

    :param data: dict to deserialize.
    :type data: dict
    :param boxed_type: class literal.

    :return: deserialized dict.
    :rtype: dict
    """
    return {k: _deserialize(v, boxed_type)
            for k, v in data.items() }

from tee_gateway.definitions import (  # noqa: E402
    USDC_ADDRESS,
    BASE_OPG_ADDRESS,
    ASSET_DECIMALS_BY_ADDRESS,
    DEFAULT_ASSET_DECIMALS,
)
from tee_gateway.model_registry import get_model_config  # noqa: E402
TOKEN_A_PRICE_CACHE_TTL_SECONDS = 60

_token_price_cache: dict[str, Any] = {
    "value": Decimal("1"),
    "updated_at": 0.0,
}
_token_price_lock = threading.Lock()


def _fetch_token_a_price_usd_mock() -> Decimal:
    """Return the USD price of the payment token used for cost calculation.

    Currently returns a fixed 1:1 ratio, which is correct for USDC-denominated
    payments (1 USDC ≈ $1 USD). For OPG-denominated payments, replace this
    with a live price feed (e.g. a DEX oracle or CoinGecko API call) that
    returns the current OPG/USD exchange rate so that token amounts are
    calculated correctly against the model's USD pricing.
    """
    return Decimal("1")


def get_token_a_price_usd() -> Decimal:
    now = time.time()
    with _token_price_lock:
        cached_value = _token_price_cache.get("value")
        cached_at = float(_token_price_cache.get("updated_at") or 0.0)
        if isinstance(cached_value, Decimal) and (now - cached_at) < TOKEN_A_PRICE_CACHE_TTL_SECONDS:
            return cached_value

        value = _fetch_token_a_price_usd_mock()
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


def _extract_usage_tokens(response_json: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(response_json, dict):
        return None
    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        return None

    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if prompt_tokens is None or completion_tokens is None:
        return None

    try:
        return max(0, int(prompt_tokens)), max(0, int(completion_tokens))
    except (TypeError, ValueError):
        return None


def _extract_model_from_context(
    request_json: dict[str, Any] | None,
    response_json: dict[str, Any] | None,
) -> str | None:
    req_model = request_json.get("model") if isinstance(request_json, dict) else None
    resp_model = response_json.get("model") if isinstance(response_json, dict) else None
    return _normalize_model_name(req_model or resp_model)


def _extract_asset_decimals_from_requirements(payment_requirements: Any) -> int:
    req = _as_dict(payment_requirements) or {}

    asset = req.get("asset")
    if not asset and isinstance(req.get("price"), dict):
        asset = req["price"].get("asset")

    if isinstance(asset, str):
        return ASSET_DECIMALS_BY_ADDRESS.get(asset.lower(), DEFAULT_ASSET_DECIMALS)
    return DEFAULT_ASSET_DECIMALS


def dynamic_session_cost_calculator(context: dict[str, Any]) -> int:
    """Compute UPTO per-request cost in token smallest units from actual usage.

    Raises ValueError if the model is not in the registry (no silent fallback).
    """
    request_json = context.get("request_json")
    response_json = context.get("response_json")

    if not isinstance(request_json, dict) or not isinstance(response_json, dict):
        raise ValueError("dynamic_session_cost_calculator requires both request_json and response_json")

    model = _extract_model_from_context(request_json, response_json)
    if not model:
        raise ValueError("Could not extract model name from request/response")

    # get_model_config raises ValueError for unknown models — no fallback
    cfg = get_model_config(model)

    usage_tokens = _extract_usage_tokens(response_json)
    if not usage_tokens:
        logger.warning("No usage tokens in response for model=%s; charging zero", model)
        return 0

    input_tokens, output_tokens = usage_tokens

    input_rate = cfg.input_price_usd
    output_rate = cfg.output_price_usd

    total_usd = (Decimal(input_tokens) * input_rate) + (Decimal(output_tokens) * output_rate)
    token_price_usd = get_token_a_price_usd()
    if token_price_usd <= 0:
        raise ValueError(f"Token A price is non-positive: {token_price_usd}")

    token_amount = total_usd / token_price_usd
    decimals = _extract_asset_decimals_from_requirements(context.get("payment_requirements"))
    scale = Decimal(10) ** decimals
    cost_smallest_units = int((token_amount * scale).to_integral_value(rounding=ROUND_CEILING))

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
