"""In-enclave moderation of image requests, backed by OpenAI's moderation endpoint.

Image requests on /v1/chat/completions — image generation, image editing
(reference images ride the same request), and the inline-image chat models —
are scored against OpenAI's ``omni-moderation-latest`` model
(https://platform.openai.com/docs/guides/moderation) before they are forwarded
to any provider. Plain text chat is NOT moderated; extending scope there is a
deliberate future change (``should_moderate_model`` is the one gate to widen).
The check covers the newest user turn — its prompt text and any attached
images (editing/compositing inputs arrive as ``data:`` URIs, so they are
scored too). Earlier turns were scored by their own requests; re-scoring the
whole transcript would double-count one offense on every subsequent message
and on every round of a client tool loop.

What the verdict does:

  * The full result (flagged, categories, per-category scores) is attached to
    the response under the ``moderation`` key — inside the encrypted OHTTP
    envelope, so only the end client sees it. Like ``images`` and ``usage`` it
    rides in the signed envelope without being part of the output hash.
  * A compact, content-free abuse signal is surfaced as outer response headers
    (``X-Moderation-Flagged`` / ``X-Moderation-Categories`` /
    ``X-Moderation-Blocked``) that the OHTTP relay is allowed to see. This is a
    deliberate, narrow exception to the "relay learns nothing" stance: the
    relay operates the user-facing service and needs a per-request abuse bit to
    enforce strike/blacklist policy on its own users. The headers never carry
    prompt content, scores, model names, or token counts, and they are only
    emitted when a request is actually flagged — clean traffic is
    byte-identical to before.
  * Requests flagged for a category in ``BLOCKED_CATEGORIES`` (child sexual
    abuse material, by default) are refused outright — the gateway returns an
    HTTP 451 error and never forwards the prompt to the model provider.

Failure policy is fail-open: if the moderation endpoint is unreachable, errors,
or no OpenAI key was injected, the request proceeds unscored (``checked`` is
False and no flag headers are emitted). Moderation is an abuse-rate-limiting
layer, not the only line of defense — providers run their own safety systems —
and a moderation outage must not take down inference for every clean user.

The moderation call itself is free (OpenAI does not bill the endpoint), so it
has no effect on x402 billing: the ``opengradient`` cost block is computed
exactly as before.

The OpenAI API key is injected at runtime via ``POST /v1/keys`` like every
provider key; without it the gateway reports ``moderation_enabled: false`` on
``/health``.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from tee_gateway.model_registry import get_model_config

logger = logging.getLogger(__name__)

MODERATION_BASE_URL = "https://api.openai.com/v1"
MODERATION_MODEL = "omni-moderation-latest"

# Categories that cause the request to be refused in the enclave rather than
# forwarded to a provider. Everything else is reported (and counted by the
# relay's strike policy) but still served. Deliberately minimal: this is the
# category with essentially no legitimate-use false-positive cost, and the one
# provider trust-and-safety teams act on.
BLOCKED_CATEGORIES = frozenset({"sexual/minors"})

# Bounds on what we send to the moderation endpoint. The newest user turn is
# almost always far below these; they exist so a pathological request can't
# make the pre-flight check slower than the inference it guards.
MAX_TEXT_CHARS = 20_000
MAX_IMAGE_PARTS = 5

# Score floor below which a category's score is omitted from the response
# payload (flagged categories are always included). Keeps the moderation block
# readable instead of thirteen near-zero floats.
SCORE_FLOOR = 0.01

# Tight budget: this runs synchronously in front of every image request, so it
# must degrade fast. Images push payloads into the megabytes, hence the write
# allowance; the endpoint itself typically answers in well under a second.
_MODERATION_TIMEOUT = httpx.Timeout(timeout=15.0, connect=5.0, read=10.0, write=10.0)
_MODERATION_LIMITS = httpx.Limits(max_keepalive_connections=5, max_connections=20)

_moderation_http_client: Optional[httpx.Client] = None


def configure_moderation_client(api_key: Optional[str]) -> None:
    """Build (or tear down) the shared moderation HTTP client after key injection.

    Called from ``llm_backend.set_provider_config`` alongside the provider
    clients. Uses its own client rather than the shared OpenAI chat client so
    the pre-flight check gets a much tighter timeout than a 180s inference call.
    """
    global _moderation_http_client

    old = _moderation_http_client
    if api_key:
        _moderation_http_client = httpx.Client(
            base_url=MODERATION_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_MODERATION_TIMEOUT,
            limits=_MODERATION_LIMITS,
            http2=True,
            follow_redirects=False,
        )
    else:
        _moderation_http_client = None

    if old is not None:
        old.close()


def moderation_available() -> bool:
    """Whether an OpenAI key was injected, i.e. whether prompts can be scored."""
    return _moderation_http_client is not None


def should_moderate_model(model: str) -> bool:
    """Whether requests for this model fall inside the moderation scope.

    Only image requests are scored: models served via a provider images
    endpoint (generation and editing) and inline-image chat models. Widening
    moderation to text chat means widening this predicate — everything
    downstream (relay strike policy, app warnings) keys off the flag headers
    and needs no change. Unknown models are left to the normal routing error
    path rather than moderated speculatively.
    """
    try:
        cfg = get_model_config(model)
    except Exception:
        return False
    return bool(cfg.image_generation or cfg.image_output)


@dataclass(frozen=True)
class ModerationOutcome:
    """The verdict for one request.

    ``checked`` is False when the request could not be scored (no key, endpoint
    failure, or nothing to score) — the fail-open path. ``blocked`` implies
    ``flagged``; it means the request must not reach a model provider.
    ``categories`` holds only the category names the moderation model flagged.
    """

    checked: bool
    flagged: bool = False
    blocked: bool = False
    categories: tuple[str, ...] = ()
    category_scores: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_response_dict(self) -> dict[str, Any]:
        """The ``moderation`` block attached inside the (sealed) response body."""
        block: dict[str, Any] = {
            "model": MODERATION_MODEL,
            "checked": self.checked,
            "flagged": self.flagged,
            "blocked": self.blocked,
            "categories": list(self.categories),
            "category_scores": self.category_scores,
        }
        if self.error:
            block["error"] = self.error
        return block

    def headers(self) -> dict[str, str]:
        """Content-free flag headers for the relay's strike/blacklist policy.

        Empty for clean (or unchecked) requests so unflagged traffic exposes
        nothing new to the relay.
        """
        if not self.flagged:
            return {}
        headers = {
            "X-Moderation-Flagged": "true",
            "X-Moderation-Categories": ",".join(self.categories),
        }
        if self.blocked:
            headers["X-Moderation-Blocked"] = "true"
        return headers


_UNCHECKED = ModerationOutcome(checked=False)


def extract_moderation_input(messages: list) -> tuple[str, list[str]]:
    """Pull the text and image URLs of the newest user turn.

    Accepts the request's message objects (or plain dicts) and returns
    ``(text, image_urls)`` for the last message with role ``user``. Images are
    kept only if they are ``data:`` URIs or http(s) URLs — the two forms the
    moderation endpoint accepts. Other attachment kinds (PDFs, audio) are not
    scoreable and are skipped.
    """
    for msg in reversed(messages or []):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        if (role or "").lower() != "user":
            continue
        content = (
            msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        )
        return _split_content(content)
    return "", []


def _split_content(content: Any) -> tuple[str, list[str]]:
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []

    text_parts: list[str] = []
    images: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            text_parts.append(str(part))
            continue
        ptype = part.get("type")
        if ptype == "text":
            text = part.get("text", "") or ""
            if text:
                text_parts.append(text)
        elif ptype == "image_url":
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if isinstance(url, str) and url.startswith(
                ("data:", "http://", "https://")
            ):
                images.append(url)
        elif ptype == "image":
            data = part.get("base64") or part.get("data")
            if data:
                mime = part.get("mime_type") or "image/png"
                images.append(f"data:{mime};base64,{data}")
    return "\n".join(text_parts), images


def moderate_messages(messages: list) -> ModerationOutcome:
    """Score the newest user turn of a chat request. Never raises.

    Every failure mode returns an unchecked outcome (fail-open); a positive
    verdict requires an actual moderation-model response.
    """
    client = _moderation_http_client
    if client is None:
        return _UNCHECKED

    text, images = extract_moderation_input(messages)
    text = text[:MAX_TEXT_CHARS]
    images = images[:MAX_IMAGE_PARTS]
    if not text.strip() and not images:
        return _UNCHECKED

    if images:
        input_payload: Any = []
        if text.strip():
            input_payload.append({"type": "text", "text": text})
        input_payload.extend(
            {"type": "image_url", "image_url": {"url": url}} for url in images
        )
    else:
        input_payload = text

    try:
        response = client.post(
            "/moderations",
            json={"model": MODERATION_MODEL, "input": input_payload},
        )
    except httpx.HTTPError as exc:
        logger.warning(
            "Moderation request failed — proceeding unscored: %s: %s",
            type(exc).__name__,
            exc,
        )
        return ModerationOutcome(checked=False, error="moderation request failed")

    if response.status_code != 200:
        logger.warning(
            "Moderation endpoint returned %s — proceeding unscored: %s",
            response.status_code,
            _error_detail(response),
        )
        return ModerationOutcome(checked=False, error="moderation endpoint error")

    try:
        result = response.json()["results"][0]
        flagged = bool(result.get("flagged"))
        raw_categories = result.get("categories") or {}
        raw_scores = result.get("category_scores") or {}
        flagged_categories = tuple(
            sorted(name for name, hit in raw_categories.items() if hit)
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning(
            "Moderation response unparseable — proceeding unscored: %s: %s",
            type(exc).__name__,
            exc,
        )
        return ModerationOutcome(checked=False, error="moderation response invalid")

    scores = {
        name: round(float(score), 4)
        for name, score in raw_scores.items()
        if isinstance(score, (int, float))
        and (score >= SCORE_FLOOR or name in flagged_categories)
    }
    blocked = any(name in BLOCKED_CATEGORIES for name in flagged_categories)

    if flagged:
        # Log category names only — never prompt content.
        logger.warning(
            "Moderation flagged request categories=%s blocked=%s",
            ",".join(flagged_categories),
            blocked,
        )

    return ModerationOutcome(
        checked=True,
        flagged=flagged,
        blocked=blocked,
        categories=flagged_categories,
        category_scores=scores,
    )


def _error_detail(response: httpx.Response) -> str:
    """A compact, safe-to-log description of a non-200 moderation response."""
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])[:200]
    except Exception:
        pass
    return response.text[:200] if response.text else "<empty body>"
