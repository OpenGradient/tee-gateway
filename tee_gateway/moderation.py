"""In-enclave content moderation gate.

The gateway routes decrypted client content to seven upstream providers, some of
which have weaker guardrails than others (notably on the image generation/edit
paths). Enforcement by any single provider protects that provider's policy, not
this operator's account — abusive *attempts* are still logged against our keys,
and because the OHTTP layer strips end-user identity (see
``ohttp_controller._IDENTIFYING_FIELDS``) the upstream provider cannot block the
individual abuser and holds us responsible instead.

This module adds one automated check that runs *inside the enclave*, before any
provider call, uniformly across every endpoint and model. Because it is fully
automated and never surfaces content to a human or the operator, it does not
weaken the TEE confidentiality guarantee: the content is already decrypted here
in order to be routed at all.

Design notes:

- The backend is pluggable (``ModerationBackend``). A concrete
  ``OpenAIModerationBackend`` is provided (OpenAI ``/moderations`` is free, has
  dedicated ``sexual/minors`` coverage, and is a single HTTP call, so it adds no
  new dependency and has negligible effect on the enclave image / PCR). A
  ``StubBackend`` placeholder is also provided for wiring the gate in before a
  classifier is chosen.
- ``fail_closed`` controls behavior when the backend is *unreachable* (a network
  or provider error, distinct from a policy hit). Default is fail-closed: if the
  gate cannot run, the request is rejected. Flip to fail-open only deliberately.
- Nothing here is billed. A blocked or gate-unavailable request never reaches a
  provider and never produces a cost block, so x402 settles nothing for it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cap the number of characters we send to the moderation backend per request.
# Abusive prompts are short; this bounds latency and backend cost on pathological
# inputs without meaningfully reducing detection.
_MAX_MODERATION_CHARS = 40_000

# Outer-response headers the enclave emits on a policy hit. They ride the same
# relay-visible channel as the cost headers (see
# ``ohttp_controller._FORWARDED_HEADER_PREFIXES``), so the relay — which maps the
# sealed request to the payer it bills — can ban or throttle that user without
# any client content or identity ever entering the enclave. The gateway still
# blocks the individual request; this flag is what enables persistent, per-user
# enforcement at the layer that actually holds identity.
MODERATION_FLAG_HEADER = "X-OG-Moderation-Flagged"
MODERATION_CATEGORIES_HEADER = "X-OG-Moderation-Categories"


class ModerationUnavailable(Exception):
    """The moderation backend could not produce a decision (network/provider error).

    Distinct from a normal "flagged" decision — this means the gate itself failed
    to run, and ``fail_closed`` decides whether that blocks the request.
    """


@dataclass(frozen=True)
class ModerationDecision:
    """Outcome of a moderation check.

    ``allowed`` is the only field the call sites must consult. ``categories`` and
    ``backend`` are for logging and metrics.
    """

    allowed: bool
    categories: list[str] = field(default_factory=list)
    backend: str = "none"


class ModerationBackend(ABC):
    """A pluggable content classifier."""

    name: str = "abstract"

    @abstractmethod
    def check(
        self, texts: list[str], image_data_uris: Optional[list[str]] = None
    ) -> ModerationDecision:
        """Classify the given text (and optional inline images).

        Must raise ``ModerationUnavailable`` if it cannot reach a verdict, rather
        than returning ``allowed=True`` on error — the fail-closed policy depends
        on the two being distinguishable.
        """
        raise NotImplementedError


class StubBackend(ModerationBackend):
    """Placeholder backend that allows everything.

    Used only to wire the gate into the call sites before a real classifier is
    selected. It never raises, so with this backend the gate is effectively a
    no-op regardless of ``fail_closed``. Replace before relying on the gate.
    """

    name = "stub"

    def check(
        self, texts: list[str], image_data_uris: Optional[list[str]] = None
    ) -> ModerationDecision:
        return ModerationDecision(allowed=True, backend=self.name)


class OpenAIModerationBackend(ModerationBackend):
    """Backend using OpenAI's free ``/moderations`` endpoint.

    Reuses the shared OpenAI ``httpx`` client built at key injection
    (``llm_backend.openai_http_client``), whose base URL is
    ``https://api.openai.com/v1``, so this posts to ``/moderations``. A transport
    error, non-2xx status, or unparseable body raises ``ModerationUnavailable``.
    """

    name = "openai"

    def __init__(self, model: str = "omni-moderation-latest") -> None:
        self.model = model

    def check(
        self, texts: list[str], image_data_uris: Optional[list[str]] = None
    ) -> ModerationDecision:
        # Imported lazily to avoid a circular import at module load
        # (llm_backend imports nothing from here, but keep the edge one-way).
        from tee_gateway import llm_backend

        client = llm_backend.openai_http_client
        if client is None:
            raise ModerationUnavailable("OpenAI HTTP client not initialized")

        joined = "\n\n".join(t for t in texts if t)[:_MAX_MODERATION_CHARS]

        # The omni moderation model accepts multimodal input arrays; include any
        # inline image references so the image gen/edit paths are covered too.
        input_items: list[dict[str, Any]] = []
        if joined:
            input_items.append({"type": "text", "text": joined})
        for uri in image_data_uris or []:
            if isinstance(uri, str) and uri.startswith("data:"):
                input_items.append({"type": "image_url", "image_url": {"url": uri}})

        if not input_items:
            # Nothing to classify (e.g. an empty request); treat as allowed.
            return ModerationDecision(allowed=True, backend=self.name)

        try:
            resp = client.post(
                "/moderations",
                json={"model": self.model, "input": input_items},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # httpx errors, non-2xx, JSON decode
            raise ModerationUnavailable(str(exc)) from exc

        results = data.get("results") or []
        flagged_categories: list[str] = []
        any_flagged = False
        for result in results:
            if not isinstance(result, dict):
                continue
            if result.get("flagged"):
                any_flagged = True
            categories = result.get("categories") or {}
            if isinstance(categories, dict):
                flagged_categories.extend(
                    name for name, hit in categories.items() if hit
                )

        return ModerationDecision(
            allowed=not any_flagged,
            categories=sorted(set(flagged_categories)),
            backend=self.name,
        )


# --- Module state ----------------------------------------------------------

_backend: ModerationBackend = StubBackend()
_enabled: bool = False
_fail_closed: bool = True


def configure_moderation(
    backend: Optional[ModerationBackend] = None,
    *,
    enabled: bool = False,
    fail_closed: bool = True,
) -> None:
    """Configure the moderation gate. Call once at startup / key injection.

    ``enabled=False`` (the default) leaves the gate off so behavior — and the PCR
    measurement path — is unchanged until an operator opts in. When enabled with
    the default ``StubBackend`` the gate still allows everything; supply a real
    backend (e.g. ``OpenAIModerationBackend()``) to actually filter.
    """
    global _backend, _enabled, _fail_closed
    if backend is not None:
        _backend = backend
    _enabled = enabled
    _fail_closed = fail_closed
    logger.info(
        "Moderation configured: enabled=%s fail_closed=%s backend=%s",
        _enabled,
        _fail_closed,
        _backend.name,
    )


def is_enabled() -> bool:
    return _enabled


def enforce(
    texts: list[str],
    image_data_uris: Optional[list[str]] = None,
    *,
    safety_identifier: Optional[str] = None,
) -> Optional[tuple[dict, int, dict[str, str]]]:
    """Run the gate. Return ``None`` to allow, or a ``(body, status, headers)`` tuple to reject.

    Call sites use it as::

        blocked = moderation.enforce(texts, images)
        if blocked:
            return blocked

    The tuple is a Flask/connexion response triple, so a controller returns it
    directly. On a policy hit the ``headers`` carry ``X-OG-Moderation-Flagged``
    (and the matched categories); in the OHTTP path these propagate to the relay
    as outer headers so it can ban the payer it bills. On a screening *outage*
    (fail-closed 503) **no** flag header is emitted — an unavailable classifier
    is not evidence of abuse and must not trigger a ban.

    - Policy hit -> ``403`` with flag headers.
    - Backend unavailable and ``fail_closed`` -> ``503`` (no flag); otherwise allow.
    ``safety_identifier`` is logged for enclave-side correlation only; the relay
    uses its own billing identity to enforce.
    """
    if not _enabled:
        return None

    try:
        decision = _backend.check(texts, image_data_uris)
    except ModerationUnavailable as exc:
        logger.error(
            "Moderation backend unavailable (fail_closed=%s, id=%s): %s",
            _fail_closed,
            safety_identifier,
            exc,
        )
        if _fail_closed:
            return (
                {
                    "error": "moderation_unavailable",
                    "message": "Request could not be screened and was not processed.",
                },
                503,
                {},
            )
        return None

    if decision.allowed:
        return None

    logger.warning(
        "Request blocked by moderation: categories=%s backend=%s id=%s",
        decision.categories,
        decision.backend,
        safety_identifier,
    )
    headers = {MODERATION_FLAG_HEADER: "1"}
    if decision.categories:
        # A comma-joined list of policy-class labels (e.g. "sexual/minors") — a
        # category name, never any client content — so the relay can prioritize.
        headers[MODERATION_CATEGORIES_HEADER] = ",".join(decision.categories)
    return (
        {
            "error": "content_policy_violation",
            "message": "This request was rejected by content moderation.",
        },
        403,
        headers,
    )


def payment_safety_identifier() -> Optional[str]:
    """Derive a stable, non-reversible identifier for the paying wallet, if any.

    The OHTTP layer strips client identity, but x402 ties every paid request to a
    payer address exposed on Flask's request-global ``g`` by the payment
    middleware. We hash that address (never the content) so a flagged source can
    be attributed, rate-limited, or blocked — and passed upstream as a provider
    "safety identifier" — without de-anonymizing anyone. Returns ``None`` when no
    payment context is present (e.g. unpaid/local calls).
    """
    try:
        import hashlib

        import x402.http.middleware.flask as x402_flask

        payload = getattr(x402_flask.g, "payment_payload", None)
        address = _extract_payer_address(payload)
        if not address:
            return None
        return hashlib.sha256(address.lower().encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


def _extract_payer_address(payload: Any) -> Optional[str]:
    """Best-effort pull of the payer address from an x402 payment payload."""
    if payload is None:
        return None
    # Payload shapes vary across x402 schemes; probe the common locations.
    for getter in (
        lambda p: getattr(p, "from_address", None),
        lambda p: getattr(getattr(p, "payload", None), "from_address", None),
        lambda p: p.get("from") if isinstance(p, dict) else None,
        lambda p: (p.get("payload") or {}).get("from") if isinstance(p, dict) else None,
    ):
        try:
            value = getter(payload)
        except Exception:
            value = None
        if isinstance(value, str) and value:
            return value
    return None


# --- Extraction helpers ----------------------------------------------------


def texts_from_messages(messages: list) -> list[str]:
    """Collect moderatable text from OpenAI-format chat messages (dicts or models).

    Pulls the text of user, system, and tool turns (the client-authored content);
    multimodal ``text`` parts are included, non-text parts are skipped here and
    handled by ``images_from_messages``.
    """
    texts: list[str] = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role not in ("user", "system", "tool", "function", None):
            continue
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if isinstance(content, str):
            if content:
                texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text") or ""
                    if text:
                        texts.append(text)
                elif isinstance(part, str) and part:
                    texts.append(part)
    return texts


def images_from_messages(messages: list) -> list[str]:
    """Collect inline ``data:`` image URIs from user turns for image moderation.

    Only inline data URIs are returned; plain remote URLs are deliberately not
    dereferenced inside the enclave (matching the image-generation path's policy
    of never fetching client-supplied URLs).
    """
    images: list[str] = []
    for msg in messages:
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("image_url", "image"):
                image_url = part.get("image_url", part)
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if isinstance(url, str) and url.startswith("data:"):
                    images.append(url)
    return images
