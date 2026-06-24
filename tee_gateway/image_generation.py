"""Endpoint-based image generation.

xAI (Aurora), ByteDance (Seedream/Seedance) and Z.ai (GLM-Image) expose image
generation through a dedicated OpenAI-compatible ``POST /images/generations``
endpoint rather than the chat path. This module owns everything specific to that
flow:

  * ``generate_images`` — shape and send the provider request, always returning
    inline ``data:`` URIs (a provider-hosted URL is fetched into the enclave so
    the client never sees a raw URL).
  * ``create_image_generation_response`` /
    ``create_image_generation_streaming_response`` — surface the result on
    ``/v1/chat/completions`` exactly like Gemini's inline-image models (images
    ride out-of-band under the message ``images`` key), signed and billed flat
    per generated image.

Per-provider request quirks (response format, ``n`` support, reference-image
editing, extra params) live in the model registry, keeping this code flat.
"""

import base64
import ipaddress
import json
import logging
import time
import uuid
from typing import Any, List, Optional
from urllib.parse import urlparse

import httpx
from flask import Response

from tee_gateway import llm_backend
from tee_gateway.models.create_chat_completion_request import (
    CreateChatCompletionRequest,
)
from tee_gateway.models.create_chat_completion_response import (
    CreateChatCompletionResponse,
)
from tee_gateway.model_registry import get_model_config
from tee_gateway.pricing import compute_session_cost
from tee_gateway.tee_manager import get_tee_keys, compute_tee_msg_hash

logger = logging.getLogger(__name__)

# Endpoint path appended to each provider's OpenAI-compatible base URL.
_IMAGE_GENERATION_PATH = "/images/generations"

# Provider -> the shared HTTP client attribute on ``llm_backend`` (built after
# key injection). Looked up by attribute so a patched/rebuilt client is picked up.
_IMAGE_CLIENT_ATTRS = {
    "x-ai": "xai_http_client",
    "bytedance": "bytedance_http_client",
    "zai": "zai_http_client",
}

# Bounds on the URL fetch (egress hardening). Provider images are well under the
# size cap; the redirect cap stops a redirect chain from being chased off-host.
_MAX_IMAGE_BYTES = 25 * 1024 * 1024  # 25 MiB
_MAX_REDIRECTS = 3
_ALLOWED_FETCH_SCHEMES = {"http", "https"}

# Shared keyless client for fetching provider-hosted image URLs into the enclave.
_image_fetch_client: Optional[httpx.Client] = None


def _validate_fetch_url(url: str) -> None:
    """Guard the image fetch against SSRF / egress abuse.

    Only http(s) URLs are allowed, and IP-literal hosts in private, loopback,
    link-local, or otherwise non-global ranges are rejected. Hostnames are not
    resolved here — this helper is only ever called with a URL from a provider's
    own ``/images/generations`` response (never client input, which is forwarded
    to the provider rather than dereferenced) — so the scheme/IP checks plus the
    size and redirect caps in ``_fetch_url_as_data_uri`` bound the blast radius if
    that trust is ever misplaced.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_FETCH_SCHEMES:
        raise ValueError(f"Refusing to fetch image from non-http(s) URL: {url!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"Refusing to fetch image from URL without a host: {url!r}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return  # hostname, not an IP literal — left to the provider's CDN
    if not ip.is_global:
        raise ValueError(f"Refusing to fetch image from non-public address: {host}")


def _fetch_url_as_data_uri(url: str) -> str:
    """Fetch a provider-hosted image URL and return it as a ``data:`` URI.

    Only ever called with URLs from a provider's ``/images/generations`` response
    — never with client-supplied input, which is forwarded to the provider rather
    than dereferenced inside the enclave. Lets URL-returning providers be surfaced
    identically to inline-byte ones: the bytes are pulled into the enclave and
    ride out inside the OHTTP/TEE envelope, so the client never sees a raw URL.

    Hardened against egress abuse: http(s) only, non-public IP hosts rejected,
    redirects capped, and the body capped at ``_MAX_IMAGE_BYTES`` (streamed so an
    oversized response is aborted instead of buffered whole).
    """
    _validate_fetch_url(url)

    global _image_fetch_client
    if _image_fetch_client is None:
        _image_fetch_client = httpx.Client(
            timeout=httpx.Timeout(timeout=120.0, connect=15.0),
            follow_redirects=True,
            max_redirects=_MAX_REDIRECTS,
        )

    with _image_fetch_client.stream("GET", url) as resp:
        resp.raise_for_status()
        declared = resp.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"Refusing to fetch image larger than {_MAX_IMAGE_BYTES} bytes"
            )
        mime = (
            (resp.headers.get("content-type") or "image/jpeg").split(";", 1)[0].strip()
        )
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_bytes():
            total += len(chunk)
            if total > _MAX_IMAGE_BYTES:
                raise ValueError(
                    f"Refusing to fetch image larger than {_MAX_IMAGE_BYTES} bytes"
                )
            chunks.append(chunk)

    b64 = base64.b64encode(b"".join(chunks)).decode("ascii")
    return f"data:{mime or 'image/jpeg'};base64,{b64}"


def generate_images(
    model: str,
    prompt: str,
    n: int = 1,
    reference_images: Optional[List[str]] = None,
) -> tuple[list[str], int]:
    """Generate images via a provider's OpenAI-compatible images endpoint.

    ``reference_images`` carries input images for image-to-image editing (e.g. a
    follow-up "add a hat" that builds on the previously generated image), sent on
    the same endpoint via the ``image`` field (a URL or base64 ``data:`` URI, or
    an array of up to 10). Models whose endpoint doesn't support it ignore them.

    Returns ``(data_uris, image_count)``. Every entry is a ``data:`` URI — when a
    provider returns a hosted URL instead of inline bytes, the gateway fetches it
    so the client always receives bytes. The count drives per-image billing.
    """
    cfg = get_model_config(model)
    provider = cfg.provider

    client_attr = _IMAGE_CLIENT_ATTRS.get(provider)
    if client_attr is None:
        raise ValueError(
            f"Provider {provider!r} does not support the image-generation endpoint"
        )
    client: Optional[httpx.Client] = getattr(llm_backend, client_attr, None)
    if client is None:
        raise RuntimeError(f"{provider} HTTP client has not been initialized")

    # n is clamped to the OpenAI-compatible providers' documented 1..10 range.
    count = max(1, min(int(n), 10))
    payload: dict[str, Any] = {"model": cfg.api_name, "prompt": prompt}
    if cfg.image_response_format is not None:
        payload["response_format"] = cfg.image_response_format
    if cfg.image_send_n:
        payload["n"] = count
    if cfg.image_extra_params:
        payload.update(cfg.image_extra_params)
    # Image-to-image editing: forward reference images via the ``image`` field (a
    # single string, or an array for multi-reference edits, up to 10). Without
    # this a follow-up edit like "add a hat" would ignore the prior image and
    # generate a fresh one from the prompt text alone. Filter to non-empty strings
    # so a malformed entry can't break JSON serialization of the request payload.
    if reference_images and cfg.image_supports_reference:
        refs = [r for r in reference_images if isinstance(r, str) and r][:10]
        if refs:
            payload["image"] = refs[0] if len(refs) == 1 else refs

    logger.info(
        "Generating %d image(s) - Provider: %s, Model: %s",
        count,
        provider,
        cfg.api_name,
    )
    resp = client.post(_IMAGE_GENERATION_PATH, json=payload)
    resp.raise_for_status()
    data = resp.json().get("data", []) or []

    images: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        b64 = item.get("b64_json")
        if b64:
            images.append(f"data:image/jpeg;base64,{b64}")
        elif item.get("url"):
            images.append(_fetch_url_as_data_uri(item["url"]))

    return images, len(images)


def _extract_image_inputs(langchain_messages: list) -> tuple[str, list[str]]:
    """Pull the text prompt and any reference images out of the user turns.

    Image-generation models take a single text prompt rather than a chat
    transcript, so we join the text of all human messages (system/assistant/tool
    turns are ignored). A user turn doing an image-to-image edit carries the prior
    image as an ``image_url``/``image`` content part alongside its text; we collect
    those reference images so they can be forwarded to the provider — without them
    a follow-up like "add a hat" would ignore the previous image and generate from
    the prompt text alone.

    Only the *latest* user turn's references are returned: each user turn replaces
    the running set (and a text-only turn clears it). Carrying references forward
    from earlier edits would forward stale images and burn through the provider's
    10-image cap. Values are coerced/filtered to strings so malformed input can't
    break downstream JSON serialization.

    Returns ``(prompt, reference_images)``.
    """
    from langchain_core.messages import HumanMessage

    text_parts: list[str] = []
    reference_images: list[str] = []
    for m in langchain_messages:
        if not isinstance(m, HumanMessage):
            continue
        content = m.content
        # References reset per user turn so the latest turn is authoritative.
        turn_images: list[str] = []
        if isinstance(content, str):
            if content:
                text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    if part:
                        text_parts.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text = part.get("text")
                    if text:
                        text_parts.append(str(text))
                elif ptype == "image_url":
                    image_url = part.get("image_url")
                    url = (
                        image_url.get("url")
                        if isinstance(image_url, dict)
                        else image_url
                    )
                    if isinstance(url, str) and url:
                        turn_images.append(url)
                elif ptype == "image":
                    # Standard LangChain image block: inline base64 + mime, or url.
                    b64 = part.get("base64")
                    url = part.get("url")
                    if isinstance(b64, str) and b64:
                        mime = part.get("mime_type")
                        mime = mime if isinstance(mime, str) and mime else "image/png"
                        turn_images.append(f"data:{mime};base64,{b64}")
                    elif isinstance(url, str) and url:
                        turn_images.append(url)
        else:
            continue  # unrecognized content shape: no text, leave references as-is
        reference_images = turn_images
    return "\n".join(text_parts), reference_images


def _run_image_generation(
    chat_request: CreateChatCompletionRequest, request_bytes: bytes
) -> dict[str, Any]:
    """Shared core for endpoint-based image generation.

    Generates the images (always inline bytes — the gateway fetches any
    provider-hosted URL), signs the response, and computes billing. Image
    generation has no text to sign, so the signature covers the request hash and
    an empty output; the image bytes ride out-of-band inside the OHTTP envelope.
    Billing is flat per generated image. Returns the pieces the streaming and
    non-streaming responders assemble into their respective envelopes.
    """
    langchain_messages = llm_backend.convert_messages(chat_request.messages)
    prompt, reference_images = _extract_image_inputs(langchain_messages)
    images, image_count = generate_images(
        chat_request.model,
        prompt,
        n=chat_request.n or 1,
        reference_images=reference_images,
    )

    timestamp = int(time.time())
    msg_hash, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
        request_bytes, "", timestamp
    )
    tee_keys = get_tee_keys()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost = compute_session_cost(chat_request.model, usage, image_count=image_count)

    return {
        "images": images,
        "usage": usage,
        "opengradient": cost.model_dump(mode="json") if cost is not None else None,
        "tee_signature": tee_keys.sign_data(msg_hash),
        "tee_request_hash": input_hash_hex,
        "tee_output_hash": output_hash_hex,
        "tee_timestamp": timestamp,
        "tee_id": f"0x{tee_keys.get_tee_id()}",
    }


def create_image_generation_response(
    chat_request: CreateChatCompletionRequest, request_bytes: bytes
):
    """Non-streaming image generation via a provider's images endpoint.

    Surfaces generated images on the message under the ``images`` key exactly
    like Gemini's inline-image models.
    """
    result = _run_image_generation(chat_request, request_bytes)

    message_dict: dict[str, Any] = {"role": "assistant", "content": ""}
    if result["images"]:
        message_dict["images"] = result["images"]

    openai_response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": result["tee_timestamp"],
        "model": chat_request.model,
        "choices": [{"index": 0, "message": message_dict, "finish_reason": "stop"}],
        "tee_signature": result["tee_signature"],
        "tee_request_hash": result["tee_request_hash"],
        "tee_output_hash": result["tee_output_hash"],
        "tee_timestamp": result["tee_timestamp"],
        "tee_id": result["tee_id"],
        "usage": result["usage"],
    }
    if result["opengradient"] is not None:
        openai_response["opengradient"] = result["opengradient"]

    CreateChatCompletionResponse.from_dict(openai_response)
    return openai_response


def create_image_generation_streaming_response(
    chat_request: CreateChatCompletionRequest, request_bytes: bytes
):
    """Streaming image generation: image gen is not a token stream, so we invoke
    once and emit the result on the final SSE frame (mirrors the Gemini path)."""

    def generate():
        try:
            result = _run_image_generation(chat_request, request_bytes)

            final_data: dict[str, Any] = {
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "model": chat_request.model,
                "tee_signature": result["tee_signature"],
                "tee_timestamp": result["tee_timestamp"],
                "tee_request_hash": result["tee_request_hash"],
                "tee_output_hash": result["tee_output_hash"],
                "tee_id": result["tee_id"],
                "usage": result["usage"],
            }
            # Images travel out-of-band on the final frame; not part of the hash.
            if result["images"]:
                final_data["images"] = result["images"]
            if result["opengradient"] is not None:
                final_data["opengradient"] = result["opengradient"]

            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Image generation streaming error: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'error': 'Stream processing failed', 'exception_type': type(e).__name__})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
