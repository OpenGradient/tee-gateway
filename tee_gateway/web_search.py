"""In-enclave web search, backed by Exa.

This module replaces the four provider-native web search tools that used to back
the ``web_search`` request flag (OpenAI's Responses-API ``web_search``,
Anthropic's ``web_search_20250305``, Gemini's ``google_search`` grounding, and
xAI's Responses-API ``web_search``). Those differed in every dimension that
matters: which models could use them, what the results looked like, what the
response reported back, and what a "search" cost ($0.01–$0.035, with xAI billing
per *citation*). Models on providers without one (ByteDance, Nous, Z.ai) simply
could not search at all.

Instead the gateway advertises ONE ordinary function tool — see
``get_web_search_tool`` — and executes it itself, inside the enclave, against
Exa. Consequences worth stating plainly:

  * It works on every model that can call a function, which is every non-image
    model in the registry. There is no per-provider capability matrix.
  * Results, excerpt sizes, and citations are identical across models, so answer
    quality stops depending on whose search backend the model happened to ship.
  * A search is one flat price on every model (``WEB_SEARCH_PRICE_USD``), so
    clients can verify the surcharge as ``searches * price`` rather than
    reverse-engineering a provider's billable unit.
  * The query never leaves the TEE except to Exa. Nothing about the search is
    visible to the LLM provider beyond the result text the model is shown, and
    nothing is visible to the gateway operator at all.

The Exa API key is injected at runtime via ``POST /v1/keys`` like every provider
key; it is never baked into the image.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Exa's search API. Single fixed host — this client is never pointed at a
# caller-supplied URL, so it needs none of the SSRF guarding that the image
# fetcher in image_generation.py does.
EXA_BASE_URL = "https://api.exa.ai"

# Exa search mode. "auto" lets Exa pick between its neural and keyword indices
# per query, which behaves best on the open-ended questions chat users ask.
# The "deep*" modes cost ~2x and add seconds of latency; not worth it inside an
# interactive chat turn.
EXA_SEARCH_TYPE = "auto"

# The tool name the model sees. Also the name the controllers match on to decide
# a tool call is ours to execute rather than the client's.
WEB_SEARCH_TOOL_NAME = "web_search"

DEFAULT_NUM_RESULTS = 6
MAX_NUM_RESULTS = 10

# Per-result excerpt cap, and a cap on the whole formatted block. Search results
# are the single largest thing we inject into a prompt, and the caller pays for
# every one of those input tokens on every subsequent round of the tool loop —
# so this bounds both context pressure and cost. ~12k chars ≈ 3k tokens.
MAX_RESULT_CHARS = 1_500
MAX_TOTAL_CHARS = 12_000

# Exa's ceiling on the published-date filter we map `recency_days` onto.
MAX_RECENCY_DAYS = 3_650

_EXA_TIMEOUT = httpx.Timeout(timeout=30.0, connect=10.0, read=25.0, write=10.0)
_EXA_LIMITS = httpx.Limits(max_keepalive_connections=5, max_connections=20)

_exa_http_client: Optional[httpx.Client] = None


def configure_exa_client(api_key: Optional[str]) -> None:
    """Build (or tear down) the shared Exa HTTP client after key injection.

    Called from ``llm_backend.set_provider_config`` alongside the provider
    clients. Passing an empty key leaves web search unavailable, which the
    controllers surface as a clear error rather than silently answering without
    searching.
    """
    global _exa_http_client

    old = _exa_http_client
    if api_key:
        _exa_http_client = httpx.Client(
            base_url=EXA_BASE_URL,
            headers={
                "x-api-key": api_key,
                "content-type": "application/json",
            },
            timeout=_EXA_TIMEOUT,
            limits=_EXA_LIMITS,
            follow_redirects=False,
        )
    else:
        _exa_http_client = None

    if old is not None:
        old.close()


def web_search_available() -> bool:
    """Whether an Exa key was injected, i.e. whether searches can run."""
    return _exa_http_client is not None


# ---------------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------------


def get_web_search_tool() -> dict[str, Any]:
    """The ``web_search`` function tool, in OpenAI function-calling format.

    One spec for every provider: langchain converts this to each provider's own
    tool format on ``bind_tools``. The schema is deliberately flat — three
    scalar parameters, one required — because nested objects, unions, and
    ``additionalProperties`` are exactly where the providers' function-calling
    schema subsets diverge (Gemini's is the narrowest). Flat and boring is what
    makes this work identically on all of them.
    """
    return {
        "type": "function",
        "function": {
            "name": WEB_SEARCH_TOOL_NAME,
            "description": (
                "Search the live web and get back ranked results with an "
                "excerpt of each page. Use this whenever the answer depends on "
                "information you may not have: current events, anything after "
                "your training cutoff, prices, releases, documentation, or any "
                "specific fact you are not confident about. Prefer searching "
                "over guessing. Write a focused natural-language query rather "
                "than keywords, and call this again with a refined query if the "
                "first set of results does not answer the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to search for, as a focused natural-language query."
                        ),
                    },
                    "num_results": {
                        "type": "integer",
                        "description": (
                            f"How many results to return, 1-{MAX_NUM_RESULTS}. "
                            f"Defaults to {DEFAULT_NUM_RESULTS}. Ask for more "
                            "only when the question needs broad coverage."
                        ),
                    },
                    "recency_days": {
                        "type": "integer",
                        "description": (
                            "Only return pages published within this many days. "
                            "Omit unless the question is genuinely "
                            "time-sensitive — it discards older pages that are "
                            "often the best sources."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebSearchOutcome:
    """The result of one ``web_search`` tool call.

    ``content`` is the text handed back to the model as the tool result.
    ``citations`` are surfaced to the client out-of-band (like generated
    images) so a UI can show its sources. ``billable`` is False for failures
    and for empty/invalid calls, so a caller is never charged for a search that
    produced nothing.
    """

    content: str
    citations: list[dict[str, str]] = field(default_factory=list)
    billable: bool = False
    is_error: bool = False
    # Exa's own reported price for this call. Logged for margin reconciliation
    # only — the client is billed the flat published rate, never this number,
    # so settlement never depends on a third party's self-reported figure.
    reported_cost_usd: Optional[float] = None


def execute_web_search_call(args: dict[str, Any]) -> WebSearchOutcome:
    """Run one ``web_search`` tool call from its (already-parsed) arguments.

    Never raises: every failure mode comes back as an error outcome whose
    ``content`` reads as an instruction to the model, so a flaky search degrades
    into "the model was told the search failed" rather than a dead request.
    """
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return WebSearchOutcome(
            content=(
                "Web search error: a non-empty `query` string is required. "
                "Call the tool again with a query."
            ),
            is_error=True,
        )
    query = query.strip()

    num_results = _clamp_int(
        args.get("num_results"), DEFAULT_NUM_RESULTS, 1, MAX_NUM_RESULTS
    )
    recency_days = _clamp_int(args.get("recency_days"), 0, 0, MAX_RECENCY_DAYS)

    return run_web_search(query, num_results, recency_days or None)


def run_web_search(
    query: str,
    num_results: int = DEFAULT_NUM_RESULTS,
    recency_days: Optional[int] = None,
) -> WebSearchOutcome:
    """Query Exa and format the results for model consumption."""
    client = _exa_http_client
    if client is None:
        logger.error("web_search requested but no Exa API key was injected")
        return WebSearchOutcome(
            content=(
                "Web search error: search is not configured on this gateway. "
                "Answer from your own knowledge and say that you could not "
                "verify it against the web."
            ),
            is_error=True,
        )

    payload: dict[str, Any] = {
        "query": query,
        "type": EXA_SEARCH_TYPE,
        "numResults": num_results,
        # Ask for page text only. `highlights` and `summary` are each billed as
        # another content type per page, and the excerpt is what the model
        # actually reasons over.
        "contents": {"text": {"maxCharacters": MAX_RESULT_CHARS}},
    }
    if recency_days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=recency_days)
        payload["startPublishedDate"] = cutoff.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    try:
        response = client.post("/search", json=payload)
    except httpx.HTTPError as exc:
        logger.warning("Exa search transport error for %r: %s", query, exc)
        return WebSearchOutcome(
            content=f"Web search error: could not reach the search service ({exc}).",
            is_error=True,
        )

    if response.status_code != 200:
        # Surface the provider's own message: Exa's 4xx bodies name the actual
        # problem (bad key, quota, malformed filter), which is what makes this
        # debuggable from enclave logs where we cannot attach a debugger.
        detail = _error_detail(response)
        logger.warning(
            "Exa search failed for %r: HTTP %s %s", query, response.status_code, detail
        )
        return WebSearchOutcome(
            content=(
                f"Web search error: the search service returned "
                f"HTTP {response.status_code} ({detail})."
            ),
            is_error=True,
        )

    try:
        body = response.json()
    except ValueError as exc:
        logger.warning("Exa returned non-JSON for %r: %s", query, exc)
        return WebSearchOutcome(
            content="Web search error: the search service returned a malformed response.",
            is_error=True,
        )

    results = body.get("results")
    if not isinstance(results, list):
        results = []

    cost = body.get("costDollars")
    reported_cost = (
        float(cost["total"])
        if isinstance(cost, dict) and isinstance(cost.get("total"), (int, float))
        else None
    )

    if not results:
        # A search that ran but matched nothing still consumed an Exa request, so
        # it is billable — and the model needs to be told, or it will silently
        # answer as if it had searched successfully.
        logger.info("Exa search for %r returned no results", query)
        return WebSearchOutcome(
            content=(
                f'No web results were found for "{query}". Try a broader or '
                "differently-worded query, or tell the user you could not find "
                "anything."
            ),
            billable=True,
            reported_cost_usd=reported_cost,
        )

    content, citations = _format_results(query, results)
    logger.info(
        "Exa search ok — query=%r results=%d chars=%d reported_cost_usd=%s",
        query,
        len(citations),
        len(content),
        reported_cost,
    )
    return WebSearchOutcome(
        content=content,
        citations=citations,
        billable=True,
        reported_cost_usd=reported_cost,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_results(query: str, results: list[Any]) -> tuple[str, list[dict[str, str]]]:
    """Render Exa results as numbered, citable blocks for the model.

    Stops adding results once MAX_TOTAL_CHARS is reached so one verbose page
    cannot crowd out the rest (and cannot balloon the input-token bill on every
    later round of the tool loop). Citations are collected only for results that
    actually made it into the text, so what the UI shows as a source is exactly
    what the model was shown.
    """
    header = f'Web search results for "{query}":\n'
    blocks: list[str] = []
    citations: list[dict[str, str]] = []
    used = len(header)

    for item in results:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue

        title = _clean_str(item.get("title")) or url
        published = _clean_str(item.get("publishedDate"))
        author = _clean_str(item.get("author"))
        text = _clean_str(item.get("text"))
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS].rstrip() + "…"

        index = len(citations) + 1
        lines = [f"[{index}] {title}", f"URL: {url}"]
        if published:
            lines.append(f"Published: {published[:10]}")
        if author:
            lines.append(f"Author: {author}")
        if text:
            lines.append(text)
        block = "\n".join(lines)

        if citations and used + len(block) + 2 > MAX_TOTAL_CHARS:
            break

        blocks.append(block)
        used += len(block) + 2
        citation: dict[str, str] = {"title": title, "url": url}
        if published:
            citation["published_date"] = published
        citations.append(citation)

    if not blocks:
        return (
            f'No usable web results were found for "{query}".',
            [],
        )

    footer = (
        "\nCite the sources you relied on by their URL, and say so plainly if "
        "the results do not answer the question."
    )
    return (header + "\n" + "\n\n".join(blocks) + "\n" + footer, citations)


def _clean_str(value: Any) -> str:
    """Coerce an Exa field to a stripped string (fields are nullable)."""
    return value.strip() if isinstance(value, str) else ""


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    """Coerce a model-supplied number to an int in [low, high].

    Models pass these as strings and floats often enough that being strict here
    would turn a usable call into a retry loop.
    """
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def _error_detail(response: httpx.Response) -> str:
    """Best-effort human-readable detail from an Exa error response."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200].strip() or response.reason_phrase
    if isinstance(body, dict):
        for key in ("error", "message", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value[:200]
    return str(body)[:200]
