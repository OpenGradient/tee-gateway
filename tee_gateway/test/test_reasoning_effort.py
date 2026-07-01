"""
Unit tests for the per-request reasoning-effort control.

Covers:
  - model_registry: which models declare ``supports_reasoning_effort``
  - llm_backend._effective_reasoning_effort: capability + value gating
  - llm_backend.get_chat_model_cached: effort maps to each provider's native
    knob (reasoning_effort / effort / thinking_level) only for capable models
  - chat_controller: reasoning_effort threads to the model and into the signed
    request hash
"""

import unittest
from unittest.mock import patch, Mock

from tee_gateway.config import ProviderConfig
from tee_gateway.model_registry import get_model_config
from tee_gateway import llm_backend
from tee_gateway.llm_backend import (
    _effective_reasoning_effort,
    get_chat_model_cached,
    set_provider_config,
)
from tee_gateway.controllers.chat_controller import (
    create_chat_completion,
    _chat_request_to_dict,
    _parse_chat_request,
)


# ---------------------------------------------------------------------------
# model_registry capability flag
# ---------------------------------------------------------------------------


class TestReasoningEffortCapability(unittest.TestCase):
    def test_flagship_models_support_effort(self):
        for model in (
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-sonnet-4-6",
            "claude-fable-5",
            "gpt-5.5",
            "o4-mini",
            "gemini-3.5-flash",
        ):
            self.assertTrue(
                get_model_config(model).supports_reasoning_effort,
                f"{model} should support reasoning effort",
            )

    def test_non_reasoning_models_do_not_support_effort(self):
        # Older/non-reasoning tiers where the effort field would 400 or is moot.
        for model in (
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "gpt-4.1",
            "gemini-2.5-flash",
            "grok-4",
            "seed-1.6",
            "glm-5.2",
        ):
            self.assertFalse(
                get_model_config(model).supports_reasoning_effort,
                f"{model} should not support reasoning effort",
            )


# ---------------------------------------------------------------------------
# llm_backend._effective_reasoning_effort
# ---------------------------------------------------------------------------


class TestEffectiveReasoningEffort(unittest.TestCase):
    def test_supported_model_valid_value_passes_through(self):
        cfg = get_model_config("claude-opus-4-8")
        for eff in ("low", "medium", "high"):
            self.assertEqual(_effective_reasoning_effort(cfg, eff), eff)

    def test_unsupported_model_ignores_effort(self):
        cfg = get_model_config("gpt-4.1")
        self.assertIsNone(_effective_reasoning_effort(cfg, "high"))

    def test_invalid_value_ignored_even_for_capable_model(self):
        cfg = get_model_config("gpt-5.5")
        for eff in (None, "", "medium-high", "max", "MEDIUM"):
            self.assertIsNone(_effective_reasoning_effort(cfg, eff))


# ---------------------------------------------------------------------------
# llm_backend.get_chat_model_cached provider mapping
# ---------------------------------------------------------------------------


class TestProviderEffortMapping(unittest.TestCase):
    def setUp(self):
        # Inject dummy keys for every provider; also clears the model cache.
        set_provider_config(
            ProviderConfig(
                openai_api_key="k",
                anthropic_api_key="k",
                google_api_key="k",
                xai_api_key="k",
                bytedance_api_key="k",
                nous_api_key="k",
                zai_api_key="k",
            )
        )

    def tearDown(self):
        llm_backend.get_chat_model_cached.cache_clear()

    def test_openai_maps_to_reasoning_effort(self):
        model = get_chat_model_cached("gpt-5.5", 1.0, 4096, reasoning_effort="high")
        self.assertEqual(getattr(model, "reasoning_effort", None), "high")

    def test_anthropic_maps_to_output_config_effort(self):
        model = get_chat_model_cached(
            "claude-opus-4-8", 1.0, 4096, reasoning_effort="low"
        )
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])
        self.assertEqual(payload.get("output_config"), {"effort": "low"})

    def test_google_maps_to_thinking_level(self):
        model = get_chat_model_cached(
            "gemini-3.5-flash", 1.0, 4096, reasoning_effort="medium"
        )
        self.assertEqual(getattr(model, "thinking_level", None), "medium")
        self.assertIsNone(getattr(model, "thinking_budget", None))

    def test_unsupported_model_receives_no_effort(self):
        # gpt-4.1 is not effort-capable: no reasoning_effort should be set.
        model = get_chat_model_cached("gpt-4.1", 0.7, 4096, reasoning_effort="high")
        self.assertIsNone(getattr(model, "reasoning_effort", None))

    def test_capable_model_without_effort_leaves_field_unset(self):
        model = get_chat_model_cached("claude-opus-4-8", 1.0, 4096)
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])
        self.assertNotIn("output_config", payload)


# ---------------------------------------------------------------------------
# chat_controller: parsing + request hash
# ---------------------------------------------------------------------------


class TestChatRequestHashing(unittest.TestCase):
    def test_effort_parsed_from_request_body(self):
        req = _parse_chat_request(
            {
                "model": "claude-opus-4-8",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            }
        )
        self.assertEqual(req.reasoning_effort, "high")

    def test_effort_absent_is_none(self):
        req = _parse_chat_request(
            {
                "model": "claude-opus-4-8",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        self.assertIsNone(req.reasoning_effort)

    def test_effort_included_in_canonical_hash_when_set(self):
        req = _parse_chat_request(
            {
                "model": "claude-opus-4-8",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "low",
            }
        )
        self.assertEqual(_chat_request_to_dict(req)["reasoning_effort"], "low")

    def test_effort_omitted_from_hash_when_unset(self):
        req = _parse_chat_request(
            {
                "model": "claude-opus-4-8",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        self.assertNotIn("reasoning_effort", _chat_request_to_dict(req))


# ---------------------------------------------------------------------------
# chat_controller integration
# ---------------------------------------------------------------------------


class _MockResponse:
    def __init__(self, content=""):
        self.content = content
        self.tool_calls = []
        self.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }


def _mock_tee_keys():
    tee = Mock()
    tee.sign_data.return_value = "bW9ja3NpZ25hdHVyZQ=="
    tee.get_tee_id.return_value = "abcdef01" * 8
    return tee


class TestChatControllerReasoningEffort(unittest.TestCase):
    @patch("tee_gateway.controllers.chat_controller.compute_session_cost")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_effort_threaded_to_model(
        self, mock_connexion, mock_get_model, mock_get_tee_keys, mock_cost
    ):
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = {
            "model": "claude-opus-4-8",
            "messages": [{"role": "user", "content": "think hard"}],
            "reasoning_effort": "high",
            "stream": False,
        }
        model = Mock()
        model.invoke.return_value = _MockResponse(content="ok")
        model.bind_tools.return_value = model
        mock_get_model.return_value = model
        mock_get_tee_keys.return_value = _mock_tee_keys()
        mock_cost.return_value = None

        create_chat_completion(None)

        self.assertEqual(mock_get_model.call_args.kwargs["reasoning_effort"], "high")

    @patch("tee_gateway.controllers.chat_controller.compute_session_cost")
    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_no_effort_passes_none(
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

        self.assertIsNone(mock_get_model.call_args.kwargs["reasoning_effort"])


if __name__ == "__main__":
    unittest.main()
