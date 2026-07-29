"""Dedicated web-search endpoint (POST /v1/web_search).

The gateway runs no tool loop of its own: a client whose model asks to search
calls this endpoint, feeds ``content`` back to its model as the tool result,
and shows ``citations`` to its user. The search itself runs inside the enclave
against Exa (see web_search.py), so a query rides the same encrypted channel as
a chat request and is never visible to the relay or the gateway operator.

Request body::

    {"query": "...", "num_results": 6, "recency_days": 30}

``query`` is required; the other two are optional and clamped to the module's
bounds. Any other fields are ignored (but still part of the signed request
hash, which covers the body exactly as received).

The response is signed like every other paid endpoint — RSA-PSS over
``keccak256(requestHash || outputHash || timestamp)`` with the request hash
computed over the canonical (sorted-keys) JSON body and the output hash over
``content`` — and carries an ``opengradient`` cost block settled by x402 at the
flat ``WEB_SEARCH_PRICE_USD`` rate. Failures (Exa unreachable, provider error)
return 502 without a cost block, so they are never settled; a missing Exa key
returns 503.
"""

import json
import logging
import time
import uuid

from flask import request

from tee_gateway.pricing import compute_web_search_cost
from tee_gateway.tee_manager import get_tee_keys, compute_tee_msg_hash
from tee_gateway.web_search import (
    execute_web_search_call,
    web_search_available,
)

logger = logging.getLogger(__name__)


def create_web_search():
    """POST /v1/web_search — run one Exa search and return signed results."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return {"error": "Request must be a JSON object"}, 415

    if not web_search_available():
        return {"error": "Web search is not configured on this gateway"}, 503

    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"error": "A non-empty `query` string is required"}, 400

    # Hash the body exactly as the client sent it (canonicalized), so the
    # client can recompute the request hash from what it built. The `endpoint`
    # discriminator the OHTTP dispatcher pops never reaches this handler.
    request_bytes = json.dumps(body, sort_keys=True).encode("utf-8")

    outcome = execute_web_search_call(body)

    if outcome.is_error:
        # Not billable, so no cost block: x402 skips settlement on this
        # request. `content` is still model-readable ("the search failed"), so
        # a client that wants its model to recover can relay it.
        return {"error": outcome.content}, 502

    timestamp = int(time.time())
    msg_hash, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
        request_bytes, outcome.content, timestamp
    )
    tee_keys = get_tee_keys()
    signature = tee_keys.sign_data(msg_hash)

    response = {
        "id": f"websearch-{uuid.uuid4()}",
        "object": "web_search.result",
        "created": timestamp,
        "query": query.strip(),
        "content": outcome.content,
        "citations": outcome.citations,
        "tee_signature": signature,
        "tee_request_hash": input_hash_hex,
        "tee_output_hash": output_hash_hex,
        "tee_timestamp": timestamp,
        "tee_id": f"0x{tee_keys.get_tee_id()}",
    }

    # A search that ran but matched nothing still consumed an Exa request and
    # is billable (outcome.billable is True for both). Cost-calculation
    # failure (price feed down) logs CRITICAL and simply omits the block —
    # the request goes unsettled, matching the chat path's fail-open contract.
    cost = compute_web_search_cost()
    if cost is not None:
        response["opengradient"] = cost.model_dump(mode="json")

    return response
