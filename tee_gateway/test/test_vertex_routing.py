"""Unit tests for GCP Vertex AI routing of Anthropic (Claude) models.

When a GCP service-account key is injected via /v1/keys, Claude models are
served through ``AnthropicVertex``; without it, they fall back to Anthropic's
direct API. Gemini deliberately stays on the direct Gemini API either way (a
paid-tier Gemini key already bills to its GCP project). No network calls are
made — the tests only exercise client construction and provider routing.
"""

import json
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from tee_gateway import llm_backend
from tee_gateway.config import ProviderConfig
from tee_gateway.llm_backend import (
    ChatAnthropicVertex,
    get_chat_model_cached,
    set_provider_config,
    vertex_enabled,
)
from tee_gateway.model_registry import SupportedModel


def _fake_service_account_json(project_id: str = "test-project") -> str:
    """A structurally valid service-account key with a throwaway RSA key."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return json.dumps(
        {
            "type": "service_account",
            "project_id": project_id,
            "private_key_id": "test-key-id",
            "private_key": pem,
            "client_email": f"tee-gateway@{project_id}.iam.gserviceaccount.com",
            "client_id": "1234567890",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )


_FAKE_SA_JSON = _fake_service_account_json()


class VertexRoutingTestCase(unittest.TestCase):
    def tearDown(self) -> None:
        # Reset module state so other test files see an unconfigured backend.
        llm_backend._provider_config = None
        llm_backend._vertex_credentials = None
        llm_backend._vertex_project_id = None
        llm_backend._vertex_location = llm_backend.VERTEX_DEFAULT_LOCATION
        get_chat_model_cached.cache_clear()

    def _configure(self, **kwargs) -> None:
        set_provider_config(ProviderConfig(**kwargs))

    # ── routing with GCP credentials ─────────────────────────────────────

    def test_claude_routed_through_vertex_when_gcp_configured(self) -> None:
        self._configure(gcp_service_account_json=_FAKE_SA_JSON)
        self.assertTrue(vertex_enabled())

        model = get_chat_model_cached("claude-opus-5", 0.7, 256)
        self.assertIsInstance(model, ChatAnthropicVertex)
        self.assertEqual(model.vertex_project_id, "test-project")
        self.assertEqual(model.vertex_region, "global")
        self.assertEqual(model.model, "claude-opus-5")
        # Opus 5 rejects `temperature`; the vertex path must strip it too.
        self.assertIsNone(model.temperature)

        # The underlying SDK client must be the Vertex client, aimed at the
        # global aiplatform endpoint.
        client = model._client
        self.assertEqual(type(client).__name__, "AnthropicVertex")
        self.assertIn("aiplatform.googleapis.com", str(client.base_url))

    def test_dated_claude_snapshots_use_vertex_at_form_ids(self) -> None:
        self._configure(gcp_service_account_json=_FAKE_SA_JSON)

        for user_name, vertex_id in [
            ("claude-sonnet-4-5", "claude-sonnet-4-5@20250929"),
            ("claude-haiku-4-5", "claude-haiku-4-5@20251001"),
            ("claude-opus-4-5", "claude-opus-4-5@20251101"),
        ]:
            with self.subTest(model=user_name):
                model = get_chat_model_cached(user_name, 0.7, 256)
                self.assertIsInstance(model, ChatAnthropicVertex)
                self.assertEqual(model.model, vertex_id)

    def test_current_gen_claude_keeps_bare_ids_on_vertex(self) -> None:
        self._configure(gcp_service_account_json=_FAKE_SA_JSON)
        for name in [
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-fable-5",
        ]:
            with self.subTest(model=name):
                model = get_chat_model_cached(name, 0.7, 256)
                self.assertIsInstance(model, ChatAnthropicVertex)
                self.assertEqual(model.model, name)

    def test_gemini_stays_direct_even_with_gcp_configured(self) -> None:
        # Gemini keeps the direct API: its paid-tier key already bills to a
        # GCP project, so only Claude moves to Vertex.
        self._configure(
            google_api_key="AIza-test", gcp_service_account_json=_FAKE_SA_JSON
        )
        model = get_chat_model_cached("gemini-2.5-flash", 0.7, 256)
        self.assertFalse(model.vertexai)
        self.assertFalse(model.client._api_client.vertexai)

        # And GCP credentials are no substitute for the Gemini key.
        self.tearDown()
        self._configure(gcp_service_account_json=_FAKE_SA_JSON)
        with self.assertRaises(ValueError):
            get_chat_model_cached("gemini-2.5-flash", 0.7, 256)

    def test_project_and_location_overrides(self) -> None:
        self._configure(
            gcp_service_account_json=_FAKE_SA_JSON,
            gcp_project_id="override-project",
            gcp_location="us",
        )
        model = get_chat_model_cached("claude-sonnet-5", 0.7, 256)
        self.assertEqual(model.vertex_project_id, "override-project")
        self.assertEqual(model.vertex_region, "us")

    def test_vertex_preferred_over_direct_anthropic_key(self) -> None:
        self._configure(
            anthropic_api_key="sk-ant-test",
            gcp_service_account_json=_FAKE_SA_JSON,
        )
        self.assertIsInstance(
            get_chat_model_cached("claude-opus-5", 0.7, 256), ChatAnthropicVertex
        )

    # ── fallback without GCP credentials ─────────────────────────────────

    def test_direct_apis_used_without_gcp_credentials(self) -> None:
        self._configure(anthropic_api_key="sk-ant-test", google_api_key="AIza-test")
        self.assertFalse(vertex_enabled())

        claude = get_chat_model_cached("claude-opus-5", 0.7, 256)
        self.assertNotIsInstance(claude, ChatAnthropicVertex)

        gemini = get_chat_model_cached("gemini-2.5-flash", 0.7, 256)
        self.assertFalse(gemini.vertexai)

    def test_missing_key_and_gcp_raises(self) -> None:
        self._configure(openai_api_key="sk-test")
        with self.assertRaises(ValueError):
            get_chat_model_cached("claude-opus-5", 0.7, 256)
        with self.assertRaises(ValueError):
            get_chat_model_cached("gemini-2.5-flash", 0.7, 256)

    # ── config validation and reporting ──────────────────────────────────

    def test_malformed_service_account_json_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._configure(gcp_service_account_json="not-json")
        # The failed injection must not leave stale vertex state behind.
        self.assertFalse(vertex_enabled())

    def test_service_account_without_project_requires_override(self) -> None:
        sa = json.loads(_FAKE_SA_JSON)
        del sa["project_id"]
        with self.assertRaises(ValueError):
            self._configure(gcp_service_account_json=json.dumps(sa))
        # An explicit override makes the same key acceptable.
        self._configure(
            gcp_service_account_json=json.dumps(sa), gcp_project_id="explicit"
        )
        self.assertTrue(vertex_enabled())

    def test_initialized_providers_reflect_vertex(self) -> None:
        cfg = ProviderConfig(gcp_service_account_json=_FAKE_SA_JSON)
        self.assertTrue(cfg.vertex_enabled())
        providers = cfg.initialized_providers()
        self.assertIn("anthropic", providers)
        # Gemini is not served by Vertex routing — it still needs its own key.
        self.assertNotIn("google", providers)
        self.assertNotIn("openai", providers)

    def test_registry_vertex_ids_only_set_for_anthropic_snapshots(self) -> None:
        for supported in SupportedModel:
            cfg = supported.value
            if cfg.vertex_api_name is not None:
                self.assertEqual(cfg.provider, "anthropic")
                self.assertIn("@", cfg.vertex_api_name)


if __name__ == "__main__":
    unittest.main()
