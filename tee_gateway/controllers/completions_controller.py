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
    get_provider_from_model,
    extract_usage,
    non_streaming_invoke_kwargs,
)
from tee_gateway.pricing import compute_session_cost
from tee_gateway import moderation

logger = logging.getLogger(__name__)


def _prompt_texts(prompt: Any) -> list[str]:
    """Flatten a completion ``prompt`` (str or list of str) into texts to screen."""
    if isinstance(prompt, str):
        return [prompt] if prompt else []
    if isinstance(prompt, list):
        return [p for p in prompt if isinstance(p, str) and p]
    return []


def create_completion(body):
    """Creates a text completion for the provided prompt with TEE signing."""
    if connexion.request.is_json:
        body = CreateCompletionRequest.from_dict(connexion.request.get_json())
    else:
        return {"error": "Request must be application/json"}, 415

    # Screen the prompt before any provider call.
    blocked = moderation.enforce(
        _prompt_texts(body.prompt),
        safety_identifier=moderation.payment_safety_identifier(),
    )
    if blocked:
        return blocked

    try:
        # The web_search flag is a deprecated no-op: search moved to the
        # dedicated /v1/web_search endpoint. It stays in the hashed request
        # dict so signatures from clients that still send it verify unchanged.
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

        model = get_chat_model_cached(
            model=body.model,
            temperature=float(body.temperature)
            if body.temperature is not None
            else 0.0,
            max_tokens=body.max_tokens or 4096,
        )

        messages = [HumanMessage(content=body.prompt)]
        provider = get_provider_from_model(body.model)
        response = model.invoke(messages, **non_streaming_invoke_kwargs(provider))

        # Some providers can return content as a list of blocks; flatten to the
        # text the caller expects.
        if isinstance(response.content, list):
            response_content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in response.content
            )
        else:
            response_content = response.content or ""
        usage = extract_usage(response)

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
            cost = compute_session_cost(body.model, usage)
            if cost is not None:
                completion_response["opengradient"] = cost.model_dump(mode="json")
        return completion_response

    except Exception as e:
        logger.error(f"Completion error: {str(e)}", exc_info=True)
        return {
            "error": str(e) or "Request processing failed",
            "exception_type": type(e).__name__,
        }, 500
