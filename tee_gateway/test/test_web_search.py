"""
Unit tests for in-enclave web search (Exa) across providers.

Covers:
  - web_search: tool spec, argument coercion, Exa request shaping, result
    formatting, and every failure mode of the Exa call
  - model_registry: one flat per-search price and which models can search
  - search_loop: tool-call partitioning, round accumulation, the round cap, and
    terminal conditions
  - pricing.compute_session_cost: per-search surcharge added to token cost
  - chat_controller: the web_search flag binds the tool, searches are executed
    in-enclave rather than handed to the client, and every round is billed
"""

import json
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from tee_gateway import web_search as ws
from tee_gateway.model_registry import (
    WEB_SEARCH_PRICE_USD,
    get_web_search_price_usd,
    model_supports_web_search,
)
from tee_gateway.pricing import SessionCost, compute_session_cost
from tee_gateway.search_loop import (
    MAX_SEARCH_ROUNDS,
    SearchLoopState,
    execute_search_calls,
    run_search_loop,
    split_tool_calls,
    strip_search_tool_calls,
)
from tee_gateway.controllers.chat_controller import create_chat_completion


# ---------------------------------------------------------------------------
# Exa test doubles
# ---------------------------------------------------------------------------


def _exa_result(url: str, title: str = "A title", text: str = "Some body text"):
    return {
        "title": title,
        "url": url,
        "publishedDate": "2026-03-04T10:00:00.000Z",
        "author": "An author",
        "id": url,
        "text": text,
    }


def _exa_response(status: int = 200, body: dict | None = None):
    """A stand-in for httpx.Response carrying just what the code reads."""
    response = Mock()
    response.status_code = status
    response.json.return_value = body if body is not None else {}
    response.text = json.dumps(body) if body is not None else ""
    response.reason_phrase = "OK" if status == 200 else "Error"
    return response


class _ExaClient:
    """Records the payloads posted to /search and replays queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads: list[dict] = []

    def post(self, path, json=None):  # noqa: A002 - matches httpx.Client.post
        self.payloads.append(json)
        return self.responses.pop(0) if self.responses else _exa_response(200, {})


def _with_exa(*responses):
    """Patch in a fake Exa client, returning it so payloads can be asserted."""
    client = _ExaClient(responses)
    return patch.object(ws, "_exa_http_client", client), client


# ---------------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------------


class TestWebSearchToolSpec(unittest.TestCase):
    def test_single_provider_agnostic_function_tool(self):
        """One spec for every provider — no per-provider variants any more."""
        tool = ws.get_web_search_tool()
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "web_search")

    def test_schema_is_flat_and_only_requires_a_query(self):
        """Nested/exotic schemas are where provider support diverges."""
        params = ws.get_web_search_tool()["function"]["parameters"]
        self.assertEqual(params["required"], ["query"])
        self.assertEqual(
            set(params["properties"]), {"query", "num_results", "recency_days"}
        )
        for prop in params["properties"].values():
            self.assertIn(prop["type"], {"string", "integer"})

    def test_availability_tracks_the_injected_key(self):
        ws.configure_exa_client("test-key")
        self.assertTrue(ws.web_search_available())
        ws.configure_exa_client(None)
        self.assertFalse(ws.web_search_available())


# ---------------------------------------------------------------------------
# Argument coercion
# ---------------------------------------------------------------------------


class TestArgumentCoercion(unittest.TestCase):
    def test_missing_or_blank_query_is_an_unbilled_error(self):
        for args in ({}, {"query": "   "}, {"query": 42}):
            outcome = ws.execute_web_search_call(args)
            self.assertTrue(outcome.is_error, args)
            self.assertFalse(outcome.billable, args)
            self.assertIn("query", outcome.content)

    def test_num_results_is_clamped_not_rejected(self):
        """Models pass these as strings and floats; be forgiving."""
        self.assertEqual(ws._clamp_int("3", 6, 1, 10), 3)
        self.assertEqual(ws._clamp_int(4.7, 6, 1, 10), 4)
        self.assertEqual(ws._clamp_int(99, 6, 1, 10), 10)
        self.assertEqual(ws._clamp_int(0, 6, 1, 10), 1)
        self.assertEqual(ws._clamp_int("nonsense", 6, 1, 10), 6)
        self.assertEqual(ws._clamp_int(None, 6, 1, 10), 6)
        self.assertEqual(ws._clamp_int(True, 6, 1, 10), 6)


# ---------------------------------------------------------------------------
# Exa request shaping
# ---------------------------------------------------------------------------


class TestExaRequestShaping(unittest.TestCase):
    def test_default_request_asks_for_text_only(self):
        """highlights/summary are each billed per page; text is what's used."""
        patcher, client = _with_exa(
            _exa_response(200, {"results": [_exa_result("https://a.com")]})
        )
        with patcher:
            ws.execute_web_search_call({"query": "who won"})

        payload = client.payloads[0]
        self.assertEqual(payload["query"], "who won")
        self.assertEqual(payload["type"], ws.EXA_SEARCH_TYPE)
        self.assertEqual(payload["numResults"], ws.DEFAULT_NUM_RESULTS)
        self.assertEqual(
            payload["contents"], {"text": {"maxCharacters": ws.MAX_RESULT_CHARS}}
        )
        self.assertNotIn("startPublishedDate", payload)

    def test_num_results_is_forwarded_and_capped(self):
        patcher, client = _with_exa(
            _exa_response(200, {"results": []}),
            _exa_response(200, {"results": []}),
        )
        with patcher:
            ws.execute_web_search_call({"query": "q", "num_results": 3})
            ws.execute_web_search_call({"query": "q", "num_results": 500})

        self.assertEqual(client.payloads[0]["numResults"], 3)
        self.assertEqual(client.payloads[1]["numResults"], ws.MAX_NUM_RESULTS)

    def test_recency_days_becomes_a_published_date_floor(self):
        patcher, client = _with_exa(_exa_response(200, {"results": []}))
        with patcher:
            ws.execute_web_search_call({"query": "q", "recency_days": 7})

        cutoff = client.payloads[0]["startPublishedDate"]
        self.assertRegex(cutoff, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.000Z$")


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


class TestResultFormatting(unittest.TestCase):
    def test_results_are_numbered_with_urls_and_citations(self):
        patcher, _ = _with_exa(
            _exa_response(
                200,
                {
                    "results": [
                        _exa_result("https://a.com", "First", "Body A"),
                        _exa_result("https://b.com", "Second", "Body B"),
                    ]
                },
            )
        )
        with patcher:
            outcome = ws.execute_web_search_call({"query": "q"})

        self.assertTrue(outcome.billable)
        self.assertFalse(outcome.is_error)
        self.assertIn("[1] First", outcome.content)
        self.assertIn("https://a.com", outcome.content)
        self.assertIn("[2] Second", outcome.content)
        self.assertIn("Body B", outcome.content)
        self.assertEqual(
            outcome.citations,
            [
                {
                    "title": "First",
                    "url": "https://a.com",
                    "published_date": "2026-03-04T10:00:00.000Z",
                },
                {
                    "title": "Second",
                    "url": "https://b.com",
                    "published_date": "2026-03-04T10:00:00.000Z",
                },
            ],
        )

    def test_results_without_a_url_are_skipped(self):
        patcher, _ = _with_exa(
            _exa_response(
                200,
                {"results": [{"title": "No URL"}, _exa_result("https://ok.com")]},
            )
        )
        with patcher:
            outcome = ws.execute_web_search_call({"query": "q"})

        self.assertEqual([c["url"] for c in outcome.citations], ["https://ok.com"])

    def test_total_size_is_capped_and_citations_match_what_was_shown(self):
        """One verbose page must not crowd out the rest, or balloon the bill."""
        big = "x" * ws.MAX_RESULT_CHARS
        results = [_exa_result(f"https://a{i}.com", f"T{i}", big) for i in range(20)]
        patcher, _ = _with_exa(_exa_response(200, {"results": results}))
        with patcher:
            outcome = ws.execute_web_search_call({"query": "q"})

        self.assertLessEqual(len(outcome.content), ws.MAX_TOTAL_CHARS + 500)
        self.assertLess(len(outcome.citations), 20)
        # Every citation corresponds to a block actually put in front of the model.
        for citation in outcome.citations:
            self.assertIn(citation["url"], outcome.content)

    def test_long_excerpts_are_truncated(self):
        patcher, _ = _with_exa(
            _exa_response(
                200,
                {"results": [_exa_result("https://a.com", "T", "y" * 9_000)]},
            )
        )
        with patcher:
            outcome = ws.execute_web_search_call({"query": "q"})

        self.assertNotIn("y" * (ws.MAX_RESULT_CHARS + 1), outcome.content)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestSearchFailureModes(unittest.TestCase):
    def test_no_key_injected_is_a_recoverable_error(self):
        with patch.object(ws, "_exa_http_client", None):
            outcome = ws.execute_web_search_call({"query": "q"})
        self.assertTrue(outcome.is_error)
        self.assertFalse(outcome.billable)
        self.assertIn("not configured", outcome.content)

    def test_http_error_surfaces_the_provider_detail_and_is_not_billed(self):
        patcher, _ = _with_exa(_exa_response(401, {"error": "invalid api key"}))
        with patcher:
            outcome = ws.execute_web_search_call({"query": "q"})

        self.assertTrue(outcome.is_error)
        self.assertFalse(outcome.billable)
        self.assertIn("401", outcome.content)
        self.assertIn("invalid api key", outcome.content)

    def test_transport_error_is_not_billed(self):
        import httpx

        client = Mock()
        client.post.side_effect = httpx.ConnectError("no route")
        with patch.object(ws, "_exa_http_client", client):
            outcome = ws.execute_web_search_call({"query": "q"})

        self.assertTrue(outcome.is_error)
        self.assertFalse(outcome.billable)

    def test_malformed_json_is_not_billed(self):
        response = _exa_response(200)
        response.json.side_effect = ValueError("nope")
        patcher, _ = _with_exa(response)
        with patcher:
            outcome = ws.execute_web_search_call({"query": "q"})

        self.assertTrue(outcome.is_error)
        self.assertFalse(outcome.billable)

    def test_zero_results_is_billable_but_tells_the_model(self):
        """The Exa request was consumed, and the model must not fake an answer."""
        patcher, _ = _with_exa(_exa_response(200, {"results": []}))
        with patcher:
            outcome = ws.execute_web_search_call({"query": "obscure thing"})

        self.assertTrue(outcome.billable)
        self.assertFalse(outcome.is_error)
        self.assertIn("No web results", outcome.content)
        self.assertEqual(outcome.citations, [])

    def test_reported_cost_is_captured_for_reconciliation_only(self):
        patcher, _ = _with_exa(
            _exa_response(
                200,
                {
                    "results": [_exa_result("https://a.com")],
                    "costDollars": {"total": 0.008},
                },
            )
        )
        with patcher:
            outcome = ws.execute_web_search_call({"query": "q"})

        self.assertEqual(outcome.reported_cost_usd, 0.008)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class TestWebSearchPricing(unittest.TestCase):
    def test_every_text_model_supports_search(self):
        for model in (
            "gpt-4.1",
            "claude-sonnet-4-5",
            "gemini-2.5-flash",
            "grok-4",
            "seed-1.6",
            "hermes-4-405b",
            "glm-5.2",
        ):
            self.assertTrue(model_supports_web_search(model), model)

    def test_image_models_do_not(self):
        for model in ("grok-2-image", "gemini-2.5-flash-image"):
            self.assertFalse(model_supports_web_search(model), model)

    def test_one_flat_price_across_providers(self):
        """The whole point: a search costs the same wherever it runs."""
        prices = {
            model: get_web_search_price_usd(model)
            for model in (
                "gpt-4.1",
                "claude-sonnet-4-5",
                "gemini-2.5-flash",
                "grok-4",
                "seed-1.6",
                "hermes-4-405b",
                "glm-5.2",
            )
        }
        self.assertEqual(set(prices.values()), {WEB_SEARCH_PRICE_USD})

    def test_image_models_are_free(self):
        self.assertEqual(get_web_search_price_usd("grok-2-image"), Decimal("0"))

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            get_web_search_price_usd("not-a-real-model")


def _usage(input_tokens: int = 100, output_tokens: int = 50) -> dict:
    return {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}


def _cost(usage, model, web_search_count=0, price=Decimal("0.10")):
    feed = SimpleNamespace(get_price=lambda: price)
    with patch("tee_gateway.price_feed.get_price_feed", return_value=feed):
        return compute_session_cost(model, usage, web_search_count=web_search_count)


class TestSessionCostWithWebSearch(unittest.TestCase):
    def test_web_search_increases_cost(self):
        base = _cost(_usage(), "gpt-4.1")
        searched = _cost(_usage(), "gpt-4.1", web_search_count=2)
        self.assertIsInstance(base, SessionCost)
        self.assertIsInstance(searched, SessionCost)
        self.assertGreater(searched.cost_opg, base.cost_opg)

    def test_surcharge_is_exactly_searches_times_the_flat_rate(self):
        """The client-verifiable property: surcharge == searches * rate."""
        base = _cost(_usage(), "gpt-4.1")
        searched = _cost(_usage(), "gpt-4.1", web_search_count=3)
        scale = Decimal(10) ** 18
        delta_usd = (Decimal(searched.cost_opg - base.cost_opg) / scale) * Decimal(
            "0.10"
        )
        self.assertAlmostEqual(delta_usd, 3 * WEB_SEARCH_PRICE_USD, places=6)

    def test_same_surcharge_on_a_different_provider(self):
        deltas = []
        for model in ("gpt-4.1", "seed-1.6"):
            base = _cost(_usage(), model)
            searched = _cost(_usage(), model, web_search_count=2)
            deltas.append(searched.cost_opg - base.cost_opg)
        self.assertEqual(deltas[0], deltas[1])

    def test_zero_searches_matches_no_web_search(self):
        a = _cost(_usage(), "gpt-4.1", web_search_count=0)
        b = _cost(_usage(), "gpt-4.1")
        self.assertEqual(a.cost_opg, b.cost_opg)


# ---------------------------------------------------------------------------
# search_loop
# ---------------------------------------------------------------------------


def _search_call(query="q", call_id="call_1"):
    return {"name": "web_search", "args": {"query": query}, "id": call_id}


def _client_call(name="get_weather", call_id="call_2"):
    return {"name": name, "args": {}, "id": call_id}


def _ai(content="", tool_calls=None, tokens=(10, 5)):
    message = AIMessage(content=content, tool_calls=tool_calls or [])
    message.usage_metadata = {
        "input_tokens": tokens[0],
        "output_tokens": tokens[1],
        "total_tokens": sum(tokens),
    }
    return message


class _ScriptedModel:
    """Returns queued AIMessages, recording the messages it was invoked with."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0)


class TestSplitToolCalls(unittest.TestCase):
    def test_partitions_ours_from_the_callers(self):
        ours, theirs = split_tool_calls([_search_call(), _client_call()])
        self.assertEqual([c["name"] for c in ours], ["web_search"])
        self.assertEqual([c["name"] for c in theirs], ["get_weather"])

    def test_handles_none_and_empty(self):
        self.assertEqual(split_tool_calls(None), ([], []))
        self.assertEqual(split_tool_calls([]), ([], []))


class TestSearchLoopState(unittest.TestCase):
    def test_usage_sums_across_rounds(self):
        state = SearchLoopState()
        state.add_usage({"prompt_tokens": 100, "completion_tokens": 10})
        state.add_usage({"prompt_tokens": 400, "completion_tokens": 20})
        assert state.usage is not None
        self.assertEqual(state.usage["prompt_tokens"], 500)
        self.assertEqual(state.usage["completion_tokens"], 30)

    def test_usage_stays_none_when_nothing_reported(self):
        state = SearchLoopState()
        state.add_usage(None)
        self.assertIsNone(state.usage)

    def test_citations_are_deduped_by_url(self):
        state = SearchLoopState()
        state.add_citations([{"title": "A", "url": "https://a.com"}])
        state.add_citations(
            [
                {"title": "A again", "url": "https://a.com"},
                {"title": "B", "url": "https://b.com"},
            ]
        )
        self.assertEqual(
            [c["url"] for c in state.citations], ["https://a.com", "https://b.com"]
        )


class TestExecuteSearchCalls(unittest.TestCase):
    def test_builds_tool_messages_and_counts_billable_searches(self):
        patcher, _ = _with_exa(
            _exa_response(200, {"results": [_exa_result("https://a.com")]})
        )
        state = SearchLoopState()
        with patcher:
            messages = execute_search_calls([_search_call("news", "abc")], state)

        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], ToolMessage)
        self.assertEqual(messages[0].tool_call_id, "abc")
        self.assertEqual(messages[0].name, "web_search")
        self.assertEqual(messages[0].status, "success")
        self.assertEqual(state.search_count, 1)
        self.assertEqual(len(state.citations), 1)

    def test_failed_search_yields_an_error_tool_message_and_no_charge(self):
        patcher, _ = _with_exa(_exa_response(500, {"error": "boom"}))
        state = SearchLoopState()
        with patcher:
            messages = execute_search_calls([_search_call()], state)

        self.assertEqual(messages[0].status, "error")
        self.assertEqual(state.search_count, 0)

    def test_status_callback_receives_each_query(self):
        patcher, _ = _with_exa(
            _exa_response(200, {"results": []}), _exa_response(200, {"results": []})
        )
        seen: list[str] = []
        with patcher:
            execute_search_calls(
                [_search_call("first", "1"), _search_call("second", "2")],
                SearchLoopState(),
                on_search=seen.append,
            )
        self.assertEqual(seen, ["first", "second"])

    def test_a_throwing_status_callback_does_not_break_the_search(self):
        patcher, _ = _with_exa(_exa_response(200, {"results": []}))
        state = SearchLoopState()
        with patcher:
            messages = execute_search_calls(
                [_search_call()],
                state,
                on_search=Mock(side_effect=RuntimeError("ui gone")),
            )
        self.assertEqual(len(messages), 1)
        self.assertEqual(state.search_count, 1)


class TestRunSearchLoop(unittest.TestCase):
    def test_plain_answer_returns_immediately(self):
        model = _ScriptedModel(_ai("just an answer"))
        state = SearchLoopState()
        result = run_search_loop(model, model, [], state)
        self.assertEqual(result.content, "just an answer")
        self.assertEqual(state.search_count, 0)
        self.assertEqual(state.rounds, 1)

    def test_searches_then_answers_and_feeds_results_back(self):
        model = _ScriptedModel(
            _ai("", [_search_call("og price")]),
            _ai("The answer, with sources."),
        )
        patcher, _ = _with_exa(
            _exa_response(200, {"results": [_exa_result("https://a.com")]})
        )
        state = SearchLoopState()
        messages: list = []
        with patcher:
            result = run_search_loop(model, model, messages, state)

        self.assertEqual(result.content, "The answer, with sources.")
        self.assertEqual(state.search_count, 1)
        self.assertEqual(state.rounds, 2)
        # The second invocation saw the assistant turn plus the search results.
        second_round = model.calls[1]
        self.assertIsInstance(second_round[-1], ToolMessage)
        self.assertIn("https://a.com", second_round[-1].content)

    def test_every_round_is_billed_not_just_the_last(self):
        """Each round re-sends the conversation; the caller pays for all of it."""
        model = _ScriptedModel(
            _ai("", [_search_call("a")], tokens=(100, 10)),
            _ai("", [_search_call("b")], tokens=(600, 12)),
            _ai("done", tokens=(1200, 40)),
        )
        patcher, _ = _with_exa(
            _exa_response(200, {"results": [_exa_result("https://a.com")]}),
            _exa_response(200, {"results": [_exa_result("https://b.com")]}),
        )
        state = SearchLoopState()
        with patcher:
            run_search_loop(model, model, [], state)

        assert state.usage is not None
        self.assertEqual(state.usage["prompt_tokens"], 1900)
        self.assertEqual(state.usage["completion_tokens"], 62)
        self.assertEqual(state.search_count, 2)

    def test_client_tool_call_is_terminal(self):
        model = _ScriptedModel(_ai("", [_client_call()]))
        state = SearchLoopState()
        result = run_search_loop(model, model, [], state)
        self.assertEqual([c["name"] for c in result.tool_calls], ["get_weather"])
        self.assertEqual(state.rounds, 1)

    def test_round_cap_forces_an_answer_with_the_search_tool_unbound(self):
        """A model that keeps searching must still terminate."""
        searching = _ScriptedModel(
            *[_ai("", [_search_call(f"q{i}")]) for i in range(MAX_SEARCH_ROUNDS)]
        )
        answering = _ScriptedModel(_ai("forced answer"))
        patcher, _ = _with_exa(
            *[
                _exa_response(200, {"results": [_exa_result(f"https://a{i}.com")]})
                for i in range(MAX_SEARCH_ROUNDS)
            ]
        )
        state = SearchLoopState()
        with patcher:
            result = run_search_loop(searching, answering, [], state)

        self.assertEqual(result.content, "forced answer")
        self.assertEqual(state.search_count, MAX_SEARCH_ROUNDS)
        self.assertEqual(state.rounds, MAX_SEARCH_ROUNDS + 1)
        # The final round went to the model without the search tool bound.
        self.assertEqual(len(answering.calls), 1)

    def test_mixed_turn_drops_our_calls_and_keeps_the_callers(self):
        message = _ai("", [_search_call(), _client_call()])
        self.assertEqual(
            [c["name"] for c in strip_search_tool_calls(message)], ["get_weather"]
        )


# ---------------------------------------------------------------------------
# chat_controller integration
# ---------------------------------------------------------------------------


def _mock_tee_keys():
    tee = Mock()
    tee.sign_data.return_value = "bW9ja3NpZ25hdHVyZQ=="
    tee.get_tee_id.return_value = "abcdef01" * 8
    return tee


class _ControllerHarness(unittest.TestCase):
    """Shared patching for the chat_controller tests."""

    def setUp(self):
        self.patchers = [
            patch("tee_gateway.controllers.chat_controller.compute_session_cost"),
            patch("tee_gateway.controllers.chat_controller.get_tee_keys"),
            patch("tee_gateway.controllers.chat_controller.get_chat_model_cached"),
            patch("tee_gateway.controllers.chat_controller.connexion"),
            patch(
                "tee_gateway.controllers.chat_controller.web_search_available",
                return_value=True,
            ),
        ]
        (
            self.cost,
            self.tee,
            self.get_model,
            self.connexion,
            self.available,
        ) = [p.start() for p in self.patchers]
        self.addCleanup(lambda: [p.stop() for p in self.patchers])
        self.tee.return_value = _mock_tee_keys()
        self.cost.return_value = None

    def request(self, **overrides):
        body = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "latest news?"}],
            "stream": False,
        }
        body.update(overrides)
        self.connexion.request.is_json = True
        self.connexion.request.get_json.return_value = body


class TestChatControllerNonStreaming(_ControllerHarness):
    def test_flag_binds_our_function_tool_on_an_anthropic_model(self):
        self.request(web_search=True)
        model = Mock()
        model.invoke.return_value = _ai("Here is the news.")
        model.bind_tools.return_value = model
        self.get_model.return_value = model

        patcher, _ = _with_exa()
        with patcher:
            result = create_chat_completion(None)

        bound = model.bind_tools.call_args[0][0]
        self.assertEqual(
            [t["function"]["name"] for t in bound if isinstance(t, dict)],
            ["web_search"],
        )
        # No provider-native tool types any more.
        for tool in bound:
            self.assertEqual(tool.get("type"), "function")
        self.assertIn("choices", result)

    def test_search_runs_in_enclave_and_is_never_handed_to_the_client(self):
        self.request(web_search=True)
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = [
            _ai("", [_search_call("og token price")]),
            _ai("It trades at $X."),
        ]
        self.get_model.return_value = model

        patcher, _ = _with_exa(
            _exa_response(200, {"results": [_exa_result("https://coin.com")]})
        )
        with patcher:
            result = create_chat_completion(None)

        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("tool_calls", choice["message"])
        self.assertEqual(choice["message"]["content"], "It trades at $X.")
        self.assertEqual(
            [c["url"] for c in choice["message"]["citations"]], ["https://coin.com"]
        )
        self.assertEqual(self.cost.call_args.kwargs["web_search_count"], 1)

    def test_bytedance_can_search_too(self):
        """The case the old native-search implementation could not serve at all."""
        self.request(model="seed-1.6", web_search=True)
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = [
            _ai("", [_search_call("q")]),
            _ai("answer"),
        ]
        self.get_model.return_value = model

        patcher, _ = _with_exa(
            _exa_response(200, {"results": [_exa_result("https://a.com")]})
        )
        with patcher:
            result = create_chat_completion(None)

        self.assertEqual(result["choices"][0]["message"]["content"], "answer")
        self.assertEqual(self.cost.call_args.kwargs["web_search_count"], 1)

    def test_billed_usage_is_the_sum_over_rounds(self):
        self.request(web_search=True)
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = [
            _ai("", [_search_call("q")], tokens=(100, 10)),
            _ai("answer", tokens=(700, 30)),
        ]
        self.get_model.return_value = model

        patcher, _ = _with_exa(
            _exa_response(200, {"results": [_exa_result("https://a.com")]})
        )
        with patcher:
            result = create_chat_completion(None)

        self.assertEqual(result["usage"]["prompt_tokens"], 800)
        self.assertEqual(result["usage"]["completion_tokens"], 40)

    def test_client_tools_still_come_back_for_the_client_to_run(self):
        self.request(
            web_search=True,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
        )
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.return_value = _ai("", [_client_call()])
        self.get_model.return_value = model

        patcher, _ = _with_exa()
        with patcher:
            result = create_chat_completion(None)

        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(
            [tc["function"]["name"] for tc in choice["message"]["tool_calls"]],
            ["get_weather"],
        )

    def test_no_flag_binds_nothing_and_bills_no_search(self):
        self.request(model="gpt-4.1", messages=[{"role": "user", "content": "hi"}])
        model = Mock()
        model.invoke.return_value = _ai("hi")
        model.bind_tools.return_value = model
        self.get_model.return_value = model

        create_chat_completion(None)

        model.bind_tools.assert_not_called()
        self.assertEqual(self.cost.call_args.kwargs["web_search_count"], 0)

    def test_missing_exa_key_answers_without_searching(self):
        """Better a plain answer than a tool that always fails."""
        self.available.return_value = False
        self.request(web_search=True)
        model = Mock()
        model.invoke.return_value = _ai("answer from memory")
        model.bind_tools.return_value = model
        self.get_model.return_value = model

        result = create_chat_completion(None)

        model.bind_tools.assert_not_called()
        self.assertEqual(self.cost.call_args.kwargs["web_search_count"], 0)
        self.assertEqual(
            result["choices"][0]["message"]["content"], "answer from memory"
        )


def _chunk(content="", tool_call_chunks=None, usage=None):
    message = AIMessageChunk(content=content, tool_call_chunks=tool_call_chunks or [])
    if usage:
        message.usage_metadata = {
            "input_tokens": usage[0],
            "output_tokens": usage[1],
            "total_tokens": sum(usage),
        }
    return message


def _search_chunk(query="q", call_id="call_1"):
    return _chunk(
        tool_call_chunks=[
            {
                "name": "web_search",
                "args": json.dumps({"query": query}),
                "id": call_id,
                "index": 0,
            }
        ]
    )


def _sse_frames(response):
    """Parse a Flask SSE response into the list of JSON data frames."""
    raw = "".join(
        part.decode("utf-8") if isinstance(part, bytes) else part
        for part in response.response
    )
    frames = []
    for line in raw.split("\n\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            frames.append(json.loads(line[len("data: ") :]))
    return frames


class TestChatControllerStreaming(_ControllerHarness):
    def test_search_rounds_are_invisible_to_the_client(self):
        """The client sees status, then the answer — never our tool calls."""
        self.request(web_search=True, stream=True)
        model = Mock()
        model.bind_tools.return_value = model
        model.stream.side_effect = [
            iter([_search_chunk("og price"), _chunk(usage=(100, 10))]),
            iter([_chunk("It "), _chunk("trades."), _chunk(usage=(700, 20))]),
        ]
        self.get_model.return_value = model

        patcher, _ = _with_exa(
            _exa_response(200, {"results": [_exa_result("https://coin.com")]})
        )
        with patcher:
            frames = _sse_frames(create_chat_completion(None))

        # A status frame naming the query reached the client.
        statuses = [f["web_search"] for f in frames if "web_search" in f]
        self.assertEqual(statuses, [{"status": "searching", "query": "og price"}])

        # No tool_calls delta was ever forwarded.
        for frame in frames:
            delta = frame.get("choices", [{}])[0].get("delta", {})
            self.assertNotIn("tool_calls", delta)

        final = frames[-1]
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")
        self.assertEqual([c["url"] for c in final["citations"]], ["https://coin.com"])
        # Both rounds are billed.
        self.assertEqual(final["usage"]["prompt_tokens"], 800)
        self.assertEqual(self.cost.call_args.kwargs["web_search_count"], 1)

    def test_streamed_text_is_forwarded_and_signed(self):
        self.request(web_search=True, stream=True)
        model = Mock()
        model.bind_tools.return_value = model
        model.stream.side_effect = [
            iter([_search_chunk(), _chunk(usage=(10, 1))]),
            iter([_chunk("Hello "), _chunk("world"), _chunk(usage=(20, 2))]),
        ]
        self.get_model.return_value = model

        patcher, _ = _with_exa(
            _exa_response(200, {"results": [_exa_result("https://a.com")]})
        )
        with patcher:
            frames = _sse_frames(create_chat_completion(None))

        text = "".join(
            f["choices"][0]["delta"].get("content", "")
            for f in frames
            if f.get("choices")
        )
        self.assertEqual(text, "Hello world")
        self.assertIn("tee_signature", frames[-1])

    def test_client_tool_calls_still_stream_through(self):
        self.request(
            web_search=True,
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
        )
        model = Mock()
        model.bind_tools.return_value = model
        model.stream.side_effect = [
            iter(
                [
                    _chunk(
                        tool_call_chunks=[
                            {
                                "name": "get_weather",
                                "args": "{}",
                                "id": "c1",
                                "index": 0,
                            }
                        ]
                    ),
                    _chunk(usage=(10, 1)),
                ]
            )
        ]
        self.get_model.return_value = model

        patcher, _ = _with_exa()
        with patcher:
            frames = _sse_frames(create_chat_completion(None))

        names = [
            tc["function"]["name"]
            for f in frames
            for tc in f.get("choices", [{}])[0].get("delta", {}).get("tool_calls", [])
            if tc.get("function", {}).get("name")
        ]
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(frames[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_failed_search_still_produces_an_answer(self):
        self.request(web_search=True, stream=True)
        model = Mock()
        model.bind_tools.return_value = model
        model.stream.side_effect = [
            iter([_search_chunk(), _chunk(usage=(10, 1))]),
            iter([_chunk("Couldn't verify that."), _chunk(usage=(20, 2))]),
        ]
        self.get_model.return_value = model

        patcher, _ = _with_exa(_exa_response(503, {"error": "unavailable"}))
        with patcher:
            frames = _sse_frames(create_chat_completion(None))

        self.assertEqual(frames[-1]["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("citations", frames[-1])
        self.assertEqual(self.cost.call_args.kwargs["web_search_count"], 0)


if __name__ == "__main__":
    unittest.main()
