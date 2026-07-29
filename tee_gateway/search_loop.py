"""The in-enclave web-search tool loop.

The gateway advertises ``web_search`` (see ``web_search.get_web_search_tool``) as
an ordinary function tool, which means the model asks for a search the same way
it asks for any other tool — and something has to answer it. That something is
this module: it runs the search inside the enclave, feeds the results back, and
lets the model continue, all within one client request.

Two properties this preserves, both of which the previous provider-native search
gave up:

  * The client's tool protocol is untouched. A caller that passes its own
    ``tools`` still gets tool calls handed back to execute; only ``web_search``
    calls are intercepted. A caller that passes no tools never learns a loop
    happened — it sends one request and gets one answer.
  * Billing stays honest about a loop's real cost. Each round re-sends the whole
    conversation *plus* every prior search result, so the input tokens are
    genuinely spent several times over. ``SearchLoopState`` accumulates usage
    across every round so the caller is charged for all of it rather than for
    the last round alone.

The loop is bounded (``MAX_SEARCH_ROUNDS``) and the final round is run with the
search tool unbound, so a model that would otherwise keep searching is forced to
answer with what it has instead of spending the caller's money indefinitely.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, ToolMessage

from tee_gateway.web_search import (
    WEB_SEARCH_TOOL_NAME,
    execute_web_search_call,
)

logger = logging.getLogger(__name__)

# How many times the model may search before it must answer. Each round costs a
# full prompt re-send, so this is a cost ceiling as much as a latency one: four
# rounds is enough for "search, refine, cross-check" without letting a model
# that has decided to keep googling run up an unbounded bill.
MAX_SEARCH_ROUNDS = 4


@dataclass
class SearchLoopState:
    """Accumulator threaded through every round of one request's loop.

    Kept separate from the loop functions because the streaming controller runs
    its rounds itself (it has to forward SSE frames as they arrive) while the
    non-streaming controller delegates the whole loop — both share this state.
    """

    search_count: int = 0
    citations: list[dict[str, str]] = field(default_factory=list)
    # Running token totals across every round, in the shape extract_usage
    # returns. None until some round actually reports usage, matching the
    # "provider reported nothing, so do not charge" convention elsewhere.
    usage: Optional[dict[str, int]] = None
    rounds: int = 0

    def add_usage(self, round_usage: Optional[dict[str, int]]) -> None:
        """Fold one round's token usage into the running totals."""
        if not round_usage:
            return
        if self.usage is None:
            self.usage = {}
        for key, value in round_usage.items():
            if isinstance(value, (int, float)):
                self.usage[key] = self.usage.get(key, 0) + int(value)

    def add_citations(self, citations: list[dict[str, str]]) -> None:
        """Append citations from one search, de-duplicated by URL.

        A refining second search very often re-surfaces the best hit from the
        first, and showing the same source twice reads as a bug.
        """
        seen = {c.get("url") for c in self.citations}
        for citation in citations:
            url = citation.get("url")
            if url and url not in seen:
                seen.add(url)
                self.citations.append(citation)


def split_tool_calls(
    tool_calls: Optional[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition a turn's tool calls into (ours, the client's).

    "Ours" are ``web_search`` calls this gateway executes; the rest belong to
    tools the caller supplied and must be handed back for the caller to run.
    """
    ours: list[dict[str, Any]] = []
    theirs: list[dict[str, Any]] = []
    for call in tool_calls or []:
        name = call.get("name") if isinstance(call, dict) else None
        if name == WEB_SEARCH_TOOL_NAME:
            ours.append(call)
        else:
            theirs.append(call)
    return ours, theirs


def execute_search_calls(
    calls: list[dict[str, Any]],
    state: SearchLoopState,
    on_search: Optional[Callable[[str], None]] = None,
) -> list[ToolMessage]:
    """Run each ``web_search`` call and build the ToolMessages to feed back.

    ``on_search`` is invoked with each query before it runs, so the streaming
    controller can tell the client what is being searched for while it waits.
    Only searches that actually reached Exa are counted as billable.
    """
    messages: list[ToolMessage] = []
    for call in calls:
        args = call.get("args")
        if not isinstance(args, dict):
            args = {}
        query = args.get("query")
        if on_search is not None and isinstance(query, str) and query.strip():
            try:
                on_search(query.strip())
            except Exception:
                # A status callback is cosmetic; never let it kill the search.
                logger.debug("web_search status callback failed", exc_info=True)

        outcome = execute_web_search_call(args)
        if outcome.billable:
            state.search_count += 1
        state.add_citations(outcome.citations)

        messages.append(
            ToolMessage(
                content=outcome.content,
                tool_call_id=call.get("id") or "",
                name=WEB_SEARCH_TOOL_NAME,
                status="error" if outcome.is_error else "success",
            )
        )
    return messages


def strip_search_tool_calls(message: AIMessage) -> list[dict[str, Any]]:
    """Client-facing tool calls for a turn that mixes our tool with theirs.

    A turn asking for both ``web_search`` and one of the caller's tools cannot be
    completed by either side alone: we cannot run their tool, and they cannot run
    ours. The caller's tools win — their loop is the outer one and will come back
    to us — so our calls are dropped here and the model re-issues them on the
    next turn if it still wants to search. Rare in practice; logged when it
    happens so it does not stay invisible.
    """
    ours, theirs = split_tool_calls(getattr(message, "tool_calls", None))
    if ours:
        logger.info(
            "Dropping %d web_search call(s) from a turn that also called %d "
            "client tool(s); the model can re-issue them next turn",
            len(ours),
            len(theirs),
        )
    return theirs


def run_search_loop(
    model: Any,
    model_without_search: Any,
    messages: list[Any],
    state: SearchLoopState,
    invoke: Optional[Callable[[Any, list[Any]], AIMessage]] = None,
    on_search: Optional[Callable[[str], None]] = None,
    max_rounds: int = MAX_SEARCH_ROUNDS,
) -> AIMessage:
    """Drive the loop to a terminal turn and return it (non-streaming callers).

    Terminal means: a plain answer, or a turn calling one of the *caller's*
    tools. ``messages`` is extended in place with each round's assistant turn and
    search results, so the caller can inspect the full trajectory afterwards.

    ``model`` has the search tool bound; ``model_without_search`` does not and is
    used for the final round, which is what converts the round cap into "answer
    now" rather than "return an unanswerable search request". ``invoke`` lets a
    caller substitute its own invocation (the Anthropic structured-output path
    does not use plain ``.invoke``).
    """
    call_model = invoke if invoke is not None else (lambda m, msgs: m.invoke(msgs))

    for round_index in range(max_rounds + 1):
        last_round = round_index == max_rounds
        active_model = model_without_search if last_round else model

        response = call_model(active_model, messages)
        state.rounds = round_index + 1
        state.add_usage(_message_usage(response))

        ours, theirs = split_tool_calls(getattr(response, "tool_calls", None))
        if theirs or not ours:
            # Terminal: either a plain answer or the caller's tools to run.
            return response

        messages.append(response)
        messages.extend(execute_search_calls(ours, state, on_search))

    # Unreachable: the last iteration binds no search tool, so `ours` is empty
    # and the loop returns above.
    raise RuntimeError("search loop exited without a terminal response")


def _message_usage(message: Any) -> Optional[dict[str, int]]:
    """Token usage for one round, in the shape the cost calculator expects."""
    metadata = getattr(message, "usage_metadata", None)
    if not metadata:
        return None
    details = metadata.get("output_token_details") or {}
    return {
        "prompt_tokens": metadata.get("input_tokens", 0),
        "completion_tokens": metadata.get("output_tokens", 0),
        "total_tokens": metadata.get("total_tokens", 0),
        "reasoning_tokens": details.get("reasoning", 0),
    }
