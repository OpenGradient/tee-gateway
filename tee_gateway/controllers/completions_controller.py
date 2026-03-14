import json
import time
import uuid
import logging

import connexion
from langchain_core.messages import HumanMessage

from tee_gateway.models.create_completion_request import CreateCompletionRequest

from tee_gateway.tee_manager import get_tee_keys, compute_tee_msg_hash
from tee_gateway.llm_backend import get_chat_model_cached, extract_usage

logger = logging.getLogger(__name__)


def create_completion(body):
    """Creates a text completion for the provided prompt with TEE signing."""
    if connexion.request.is_json:
        body = CreateCompletionRequest.from_dict(connexion.request.get_json())
    else:
        return {"error": "Request must be application/json"}, 415

    try:
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

        request_bytes = json.dumps(request_dict, sort_keys=True).encode("utf-8")

        model = get_chat_model_cached(
            model=body.model,
            temperature=float(body.temperature)
            if body.temperature is not None
            else 0.0,
            max_tokens=body.max_tokens or 4096,
        )

        messages = [HumanMessage(content=body.prompt)]
        response = model.invoke(messages)

        response_content = response.content or ""
        usage = extract_usage(response)

        timestamp = int(time.time())
        msg_hash, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
            request_bytes, response_content, timestamp
        )
        tee_keys = get_tee_keys()
        signature = tee_keys.sign_data(msg_hash)

        return {
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

    except Exception as e:
        logger.error(f"Completion error: {str(e)}", exc_info=True)
        return {"error": "Request processing failed"}, 500
