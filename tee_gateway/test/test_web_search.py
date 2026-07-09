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
        # grok-4 -> xAI default ($0.025/search unit)
        self.assertEqual(get_web_search_price_usd("grok-4"), Decimal("0.025"))
        # gemini -> google default ($0.035/grounded request)
        self.assertEqual(get_web_search_price_usd("gemini-3.5-flash"), Decimal("0.035"))

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

    def test_xai_tool(self):
        self.assertEqual(get_web_search_tool("x-ai"), {"type": "web_search"})

    def test_bytedance_has_no_bound_tool(self):
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


if __name__ == "__main__":
    unittest.main()
