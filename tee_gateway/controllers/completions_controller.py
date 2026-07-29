import json
import time
import uuid
import logging

import connexion
from typing import Any

from langchain_core.messages import HumanMessage

from tee_gateway.models.create_completion_request import CreateCompletionRequest

from tee_gateway.tee_manager import get_tee_keys, compute_tee_msg_hash
from tee_gateway.llm_backend import (
    get_chat_model_cached,
    extract_usage,
)
from tee_gateway.model_registry import model_supports_web_search
from tee_gateway.pricing import compute_session_cost
from tee_gateway.search_loop import SearchLoopState, run_search_loop
from tee_gateway.web_search import get_web_search_tool, web_search_available

logger = logging.getLogger(__name__)


def create_completion(body):
    """Creates a text completion for the provided prompt with TEE signing."""
    if connexion.request.is_json:
        body = CreateCompletionRequest.from_dict(connexion.request.get_json())
    else:
        return {"error": "Request must be application/json"}, 415

    try:
        web_search = bool(getattr(body, "web_search", False))

        request_dict = {
            "model": body.model,
            "prompt": body.prompt,
            "temperature": float(body.temperature)
            if body.temperature is not None
            else 0.0,
        }
        if body.max_tokens is not None:
            request_dict["max_tokens"] = body.max_tokens
        if body.stop:
            request_dict["stop"] = body.stop
        if web_search:
            request_dict["web_search"] = True

        request_bytes = json.dumps(request_dict, sort_keys=True).encode("utf-8")

        base_model = get_chat_model_cached(
            model=body.model,
            temperature=float(body.temperature)
            if body.temperature is not None
            else 0.0,
            max_tokens=body.max_tokens or 4096,
        )

        # Web search is the gateway's own function tool, executed in-enclave for
        # any model that can call a function — see web_search.py. Skipped when no
        # Exa key was injected rather than advertising a tool that always fails.
        search_enabled = (
            web_search
            and model_supports_web_search(body.model)
            and web_search_available()
        )
        if web_search and not search_enabled:
            logger.warning(
                "web_search requested for %s but search is unavailable; "
                "completing without it",
                body.model,
            )

        messages: list[Any] = [HumanMessage(content=body.prompt)]
        search_state = SearchLoopState()
        if search_enabled:
            response = run_search_loop(
                base_model.bind_tools([get_web_search_tool()]),
                base_model,
                messages,
                search_state,
            )
        else:
            response = base_model.invoke(messages)

        # Some providers return content as a list of blocks; flatten to the text
        # the caller expects.
        if isinstance(response.content, list):
            response_content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in response.content
            )
        else:
            response_content = response.content or ""
        # With search on, usage is the sum over every round of the loop.
        usage = search_state.usage if search_enabled else extract_usage(response)

        timestamp = int(time.time())
        msg_hash, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
            request_bytes, response_content, timestamp
        )
        tee_keys = get_tee_keys()
        signature = tee_keys.sign_data(msg_hash)

        completion_response: dict[str, Any] = {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "text_completion",
            "created": timestamp,
            "model": body.model,
            "choices": [
                {
                    "text": response_content,
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
            "tee_signature": signature,
            "tee_request_hash": input_hash_hex,
            "tee_output_hash": output_hash_hex,
            "tee_timestamp": timestamp,
            "tee_id": f"0x{tee_keys.get_tee_id()}",
        }
        if usage:
            cost = compute_session_cost(
                body.model, usage, web_search_count=search_state.search_count
            )
            if cost is not None:
                completion_response["opengradient"] = cost.model_dump(mode="json")
            if search_state.citations:
                completion_response["citations"] = search_state.citations
        return completion_response

    except Exception as e:
        logger.error(f"Completion error: {str(e)}", exc_info=True)
        return {
            "error": str(e) or "Request processing failed",
            "exception_type": type(e).__name__,
        }, 500
