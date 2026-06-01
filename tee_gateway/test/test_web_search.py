"""
Unit tests for native web search support across providers.

Covers:
  - model_registry: per-search pricing lookup and provider support predicate
  - llm_backend.get_web_search_tool: provider-specific tool specs
  - llm_backend.extract_web_search_count: counting billable search units from
    each provider's response shape
  - pricing.compute_session_cost: per-search surcharge added to token cost
  - chat_controller: web_search flag binds the tool and bills the searches
"""

import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch, Mock

from langchain_core.messages import AIMessage

from tee_gateway.model_registry import (
    get_web_search_price_usd,
    provider_supports_web_search,
)
from tee_gateway.llm_backend import (
    get_web_search_tool,
    extract_web_search_count,
    extract_web_search_events,
)
from tee_gateway.pricing import SessionCost, compute_session_cost
from tee_gateway.controllers.chat_controller import create_chat_completion


# ---------------------------------------------------------------------------
# model_registry pricing
# ---------------------------------------------------------------------------


class TestWebSearchPricing(unittest.TestCase):
    def test_provider_support_predicate(self):
        for provider in ("openai", "anthropic", "google", "x-ai"):
            self.assertTrue(provider_supports_web_search(provider))
        self.assertFalse(provider_supports_web_search("bytedance"))

    def test_price_uses_provider_default(self):
        # gpt-4.1 -> openai default ($0.01/search)
        self.assertEqual(get_web_search_price_usd("gpt-4.1"), Decimal("0.01"))
        # grok-4 -> xAI default ($0.025/source)
        self.assertEqual(get_web_search_price_usd("grok-4"), Decimal("0.025"))
        # gemini -> google default ($0.035/grounded request)
        self.assertEqual(get_web_search_price_usd("gemini-2.5-flash"), Decimal("0.035"))

    def test_unsupported_provider_is_free(self):
        # ByteDance has no native web search -> no charge
        self.assertEqual(get_web_search_price_usd("seed-1.6"), Decimal("0"))

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            get_web_search_price_usd("not-a-real-model")


# ---------------------------------------------------------------------------
# llm_backend.get_web_search_tool
# ---------------------------------------------------------------------------


class TestGetWebSearchTool(unittest.TestCase):
    def test_openai_tool(self):
        self.assertEqual(get_web_search_tool("openai"), {"type": "web_search"})

    def test_anthropic_tool(self):
        tool = get_web_search_tool("anthropic")
        self.assertEqual(tool["type"], "web_search_20250305")
        self.assertEqual(tool["name"], "web_search")

    def test_google_tool(self):
        self.assertEqual(get_web_search_tool("google"), {"google_search": {}})

    def test_xai_and_bytedance_have_no_bound_tool(self):
        # xAI configures search at construction; bytedance is unsupported.
        self.assertIsNone(get_web_search_tool("x-ai"))
        self.assertIsNone(get_web_search_tool("bytedance"))


# ---------------------------------------------------------------------------
# llm_backend.extract_web_search_count
# ---------------------------------------------------------------------------


class TestExtractWebSearchCount(unittest.TestCase):
    def test_none_message(self):
        self.assertEqual(extract_web_search_count(None), 0)

    def test_plain_text_response_has_no_searches(self):
        self.assertEqual(extract_web_search_count(AIMessage(content="hi")), 0)

    def test_openai_web_search_call_blocks(self):
        msg = AIMessage(
            content=[
                {"type": "web_search_call", "id": "ws_1"},
                {"type": "text", "text": "answer"},
                {"type": "web_search_call", "id": "ws_2"},
            ]
        )
        self.assertEqual(extract_web_search_count(msg), 2)

    def test_anthropic_server_tool_use_blocks(self):
        msg = AIMessage(
            content=[
                {"type": "text", "text": "let me search"},
                {"type": "server_tool_use", "name": "web_search", "id": "srv_1"},
                {"type": "web_search_tool_result", "content": []},
            ]
        )
        # Only the server_tool_use (the request) is billed, not the result block.
        self.assertEqual(extract_web_search_count(msg), 1)

    def test_xai_citations_counted_as_sources(self):
        msg = AIMessage(content="answer")
        msg.additional_kwargs = {
            "citations": ["https://a.com", "https://b.com", "https://c.com"]
        }
        self.assertEqual(extract_web_search_count(msg), 3)

    def test_google_grounding_counts_as_one_request(self):
        msg = AIMessage(content="answer")
        msg.response_metadata = {
            "grounding_metadata": {"web_search_queries": ["q1", "q2"]}
        }
        # Google bills per grounded request, not per query.
        self.assertEqual(extract_web_search_count(msg), 1)


# ---------------------------------------------------------------------------
# llm_backend.extract_web_search_events (live UI status)
# ---------------------------------------------------------------------------


class TestExtractWebSearchEvents(unittest.TestCase):
    def test_none_message(self):
        self.assertEqual(extract_web_search_events(None, set()), [])

    def test_plain_text_chunk_has_no_events(self):
        self.assertEqual(extract_web_search_events(AIMessage(content="hi"), set()), [])

    def test_string_content_chunk_has_no_events(self):
        # Streamed text deltas arrive as plain strings, not block lists.
        self.assertEqual(extract_web_search_events(AIMessage(content=""), set()), [])

    def test_openai_web_search_call_emits_event(self):
        msg = AIMessage(content=[{"type": "web_search_call", "id": "ws_1"}])
        events = extract_web_search_events(msg, set())
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0]["query"])

    def test_anthropic_server_tool_use_emits_event_with_query(self):
        msg = AIMessage(
            content=[
                {
                    "type": "server_tool_use",
                    "name": "web_search",
                    "id": "srv_1",
                    "input": {"query": "latest news"},
                }
            ]
        )
        events = extract_web_search_events(msg, set())
        self.assertEqual(events, [{"query": "latest news"}])

    def test_openai_query_from_action(self):
        msg = AIMessage(
            content=[
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "action": {"query": "weather today"},
                }
            ]
        )
        events = extract_web_search_events(msg, set())
        self.assertEqual(events, [{"query": "weather today"}])

    def test_dedupes_block_across_chunks_by_id(self):
        seen: set = set()
        # Same search block reappears across chunks (Anthropic input deltas).
        first = AIMessage(
            content=[{"type": "server_tool_use", "name": "web_search", "id": "srv_1"}]
        )
        second = AIMessage(
            content=[
                {
                    "type": "server_tool_use",
                    "name": "web_search",
                    "id": "srv_1",
                    "input": {"query": "now complete"},
                }
            ]
        )
        self.assertEqual(len(extract_web_search_events(first, seen)), 1)
        # Already seen -> no duplicate event on the next chunk.
        self.assertEqual(extract_web_search_events(second, seen), [])

    def test_distinct_searches_each_emit(self):
        seen: set = set()
        first = AIMessage(content=[{"type": "web_search_call", "id": "ws_1"}])
        second = AIMessage(content=[{"type": "web_search_call", "id": "ws_2"}])
        self.assertEqual(len(extract_web_search_events(first, seen)), 1)
        self.assertEqual(len(extract_web_search_events(second, seen)), 1)

    def test_dedupes_by_index_when_no_id(self):
        seen: set = set()
        chunk = AIMessage(content=[{"type": "web_search_call", "index": 0}])
        self.assertEqual(len(extract_web_search_events(chunk, seen)), 1)
        self.assertEqual(extract_web_search_events(chunk, seen), [])

    def test_non_search_blocks_ignored(self):
        msg = AIMessage(
            content=[
                {"type": "text", "text": "answer"},
                {"type": "web_search_tool_result", "content": []},
            ]
        )
        self.assertEqual(extract_web_search_events(msg, set()), [])


# ---------------------------------------------------------------------------
# pricing.compute_session_cost with web search
# ---------------------------------------------------------------------------


def _usage(input_tokens: int = 100, output_tokens: int = 50) -> dict:
    return {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}


def _call(usage, model, web_search_count=0, price=Decimal("0.10")):
    feed = SimpleNamespace(get_price=lambda: price)
    with patch("tee_gateway.price_feed.get_price_feed", return_value=feed):
        return compute_session_cost(model, usage, web_search_count=web_search_count)


class TestSessionCostWithWebSearch(unittest.TestCase):
    def test_web_search_increases_cost(self):
        base = _call(_usage(), "gpt-4.1")
        searched = _call(_usage(), "gpt-4.1", web_search_count=2)
        self.assertIsInstance(base, SessionCost)
        self.assertIsInstance(searched, SessionCost)
        self.assertGreater(searched.cost_opg, base.cost_opg)

    def test_web_search_surcharge_amount(self):
        """Two openai searches at $0.01 each add $0.02 of USD cost."""
        base = _call(_usage(), "gpt-4.1")
        searched = _call(_usage(), "gpt-4.1", web_search_count=2)
        # cost_usd is reconciled from rounded OPG; compare via the underlying math.
        # At price $0.10/OPG, $0.02 surcharge ≈ 0.2 OPG = 2e17 smallest units.
        delta_opg = searched.cost_opg - base.cost_opg
        scale = Decimal(10) ** 18
        delta_usd = (Decimal(delta_opg) / scale) * Decimal("0.10")
        # Allow a tiny rounding tolerance from ceiling rounding on each call.
        self.assertAlmostEqual(delta_usd, Decimal("0.02"), places=6)

    def test_zero_searches_matches_no_web_search(self):
        a = _call(_usage(), "gpt-4.1", web_search_count=0)
        b = _call(_usage(), "gpt-4.1")
        self.assertEqual(a.cost_opg, b.cost_opg)

    def test_unsupported_provider_not_charged_for_search(self):
        # Even if a count slips through, bytedance price is 0 -> no surcharge.
        base = _call(_usage(), "seed-1.6")
        searched = _call(_usage(), "seed-1.6", web_search_count=5)
        self.assertEqual(searched.cost_opg, base.cost_opg)


# ---------------------------------------------------------------------------
# chat_controller integration
# ---------------------------------------------------------------------------


class _MockResponse:
    def __init__(self, content="", tool_calls=None, usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = usage or {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }


def _mock_tee_keys():
    tee = Mock()
    tee.sign_data.return_value = "bW9ja3NpZ25hdHVyZQ=="
    tee.get_tee_id.return_value = "abcdef01" * 8
    return tee


class TestChatControllerWebSearch(unittest.TestCase):
    @patch("tee_gateway.controllers.chat_controller.compute_session_cost")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_web_search_flag_binds_tool_and_bills(
        self, mock_connexion, mock_get_model, mock_get_tee_keys, mock_cost
    ):
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "latest news?"}],
            "web_search": True,
            "stream": False,
        }

        # Anthropic response with a server_tool_use web_search block.
        response = _MockResponse(
            content=[
                {"type": "text", "text": "Here is the news."},
                {"type": "server_tool_use", "name": "web_search", "id": "srv_1"},
            ]
        )
        model = Mock()
        model.invoke.return_value = response
        model.bind_tools.return_value = model
        mock_get_model.return_value = model
        mock_get_tee_keys.return_value = _mock_tee_keys()
        mock_cost.return_value = None

        result = create_chat_completion(None)

        # Model must be constructed with web_search=True.
        self.assertTrue(mock_get_model.call_args.kwargs["web_search"])
        # The anthropic web search tool must be bound.
        bound = model.bind_tools.call_args[0][0]
        self.assertTrue(
            any(
                isinstance(t, dict) and t.get("type") == "web_search_20250305"
                for t in bound
            )
        )
        # Billing must receive the detected search count (1 server_tool_use).
        self.assertEqual(mock_cost.call_args.kwargs["web_search_count"], 1)
        self.assertIn("choices", result)

    @patch("tee_gateway.controllers.chat_controller.compute_session_cost")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_no_web_search_does_not_bind_or_bill_search(
        self, mock_connexion, mock_get_model, mock_get_tee_keys, mock_cost
    ):
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        model = Mock()
        model.invoke.return_value = _MockResponse(content="hi")
        model.bind_tools.return_value = model
        mock_get_model.return_value = model
        mock_get_tee_keys.return_value = _mock_tee_keys()
        mock_cost.return_value = None

        create_chat_completion(None)

        self.assertFalse(mock_get_model.call_args.kwargs["web_search"])
        # No tools and no web search -> bind_tools must not be called.
        model.bind_tools.assert_not_called()
        self.assertEqual(mock_cost.call_args.kwargs["web_search_count"], 0)


class TestChatControllerWebSearchStreaming(unittest.TestCase):
    @staticmethod
    def _collect_sse(response) -> str:
        parts = []
        for part in response.response:
            parts.append(part.decode("utf-8") if isinstance(part, bytes) else part)
        return "".join(parts)

    @patch("tee_gateway.controllers.chat_controller.compute_session_cost")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_streaming_emits_web_search_status_frame(
        self, mock_connexion, mock_get_model, mock_get_tee_keys, mock_cost
    ):
        from langchain_core.messages import AIMessageChunk

        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "latest news?"}],
            "web_search": True,
            "stream": True,
        }

        # The provider streams a web-search block (twice, as input accumulates)
        # before the answer text — the status frame must be emitted only once.
        chunks = [
            AIMessageChunk(
                content=[
                    {
                        "type": "server_tool_use",
                        "name": "web_search",
                        "id": "srv_1",
                        "input": {},
                    }
                ]
            ),
            AIMessageChunk(
                content=[
                    {
                        "type": "server_tool_use",
                        "name": "web_search",
                        "id": "srv_1",
                        "input": {"query": "latest news"},
                    }
                ]
            ),
            AIMessageChunk(content="Here is the news."),
        ]
        model = Mock()
        model.stream.return_value = chunks
        model.bind_tools.return_value = model
        mock_get_model.return_value = model
        mock_get_tee_keys.return_value = _mock_tee_keys()
        mock_cost.return_value = None

        response = create_chat_completion(None)
        body = self._collect_sse(response)

        # Exactly one web-search status frame for the single (deduped) search.
        self.assertEqual(body.count('"web_search"'), 1)
        self.assertIn('"status": "searching"', body)
        # The answer text still streams as a normal content delta.
        self.assertIn("Here is the news.", body)
        self.assertIn("[DONE]", body)

    @patch("tee_gateway.controllers.chat_controller.compute_session_cost")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_streaming_without_web_search_emits_no_status_frame(
        self, mock_connexion, mock_get_model, mock_get_tee_keys, mock_cost
    ):
        from langchain_core.messages import AIMessageChunk

        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
        model = Mock()
        model.stream.return_value = [AIMessageChunk(content="hi")]
        model.bind_tools.return_value = model
        mock_get_model.return_value = model
        mock_get_tee_keys.return_value = _mock_tee_keys()
        mock_cost.return_value = None

        response = create_chat_completion(None)
        body = self._collect_sse(response)

        self.assertNotIn('"web_search"', body)
        self.assertIn("hi", body)


if __name__ == "__main__":
    unittest.main()
