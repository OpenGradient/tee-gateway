"""End-to-end wire tests for image-request moderation on /v1/chat/completions.

Unlike test_moderation.py (unit tests on the module and controller functions),
these requests travel the real pipeline: a connexion app built from the
OpenAPI spec (so request validation and routing run for real), through
``create_chat_completion``, the moderation pre-flight, TEE signing, and out as
actual HTTP responses — non-streaming JSON, SSE streams, and the sealed OHTTP
path with its header forwarding. Only the process edges are faked: the
moderation HTTP client (the test_moderation.py stand-in), the provider image
generation call, pricing, and the OHTTP encapsulation crypto.

Moderation scope is image requests only — the text-chat tests here prove the
moderation endpoint is never consulted for plain chat.

The final class is an opt-in LIVE test against the real OpenAI moderation
endpoint, gated like test_provider_usage_integration.py:

    RUN_PROVIDER_INTEGRATION_TESTS=1 OPENAI_API_KEY=sk-... \
        uv run --group test pytest tee_gateway/test/test_moderation_e2e.py -v
"""

import json
import os
import unittest
from contextlib import ExitStack
from decimal import Decimal
from unittest.mock import MagicMock, patch

import connexion

import tee_gateway.moderation as mod
from tee_gateway.controllers import ohttp_controller
from tee_gateway.encoder import JSONEncoder
from tee_gateway.pricing import SessionCost
from tee_gateway.test.test_moderation import _ModerationClient, _moderation_response

_FAKE_USAGE = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

_FLAGGED = _moderation_response(
    200,
    flagged=True,
    categories={"violence": True},
    scores={"violence": 0.93},
)

_BLOCKED = _moderation_response(
    200,
    flagged=True,
    categories={"sexual/minors": True},
    scores={"sexual/minors": 0.99},
)


def _build_app():
    """Connexion app from the OpenAPI spec, with /v1/ohttp mounted like
    __main__.create_app does (it is not part of the spec)."""
    app = connexion.App(__name__, specification_dir="../openapi/")
    app.app.json_encoder = JSONEncoder
    app.add_api("openapi.yaml", pythonic_params=True)
    app.app.add_url_rule(
        "/v1/ohttp",
        "anonymous-chat",
        ohttp_controller.create_anonymous_chat_completion,
        methods=["POST"],
    )
    return app.app


def _fake_tee_keys() -> MagicMock:
    keys = MagicMock()
    keys.sign_data.return_value = "bW9ja3NpZ25hdHVyZQ=="
    keys.get_tee_id.return_value = "abcdef01" * 8
    keys.hpke_private_key = object()
    return keys


class _WireTestCase(unittest.TestCase):
    """Shared harness: real app + faked process edges (provider, pricing,
    signing keys, image generation)."""

    def setUp(self):
        self.app = _build_app()
        self.client = self.app.test_client()

        fake_response = MagicMock()
        fake_response.content = "hello there"
        fake_response.tool_calls = None
        self.fake_model = MagicMock()
        self.fake_model.invoke.return_value = fake_response
        self.fake_model.bind_tools.return_value = self.fake_model
        self.fake_model.bind.return_value = self.fake_model

        fake_cost = SessionCost(
            cost_opg=12345,
            cost_usd=Decimal("0.001"),
            opg_price_usd=Decimal("0.5"),
        )

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch(
                "tee_gateway.controllers.chat_controller.get_chat_model_cached",
                return_value=self.fake_model,
            )
        )
        stack.enter_context(
            patch(
                "tee_gateway.controllers.chat_controller.extract_usage",
                side_effect=lambda _resp: dict(_FAKE_USAGE),
            )
        )
        stack.enter_context(
            patch(
                "tee_gateway.controllers.chat_controller.compute_session_cost",
                return_value=fake_cost,
            )
        )
        stack.enter_context(
            patch(
                "tee_gateway.controllers.chat_controller.get_tee_keys",
                return_value=_fake_tee_keys(),
            )
        )
        self.generate_images = stack.enter_context(
            patch(
                "tee_gateway.image_generation.generate_images",
                return_value=(["data:image/png;base64,QUJD"], 1),
            )
        )
        stack.enter_context(
            patch(
                "tee_gateway.image_generation.get_tee_keys",
                return_value=_fake_tee_keys(),
            )
        )
        stack.enter_context(
            patch(
                "tee_gateway.image_generation.compute_session_cost",
                return_value=fake_cost,
            )
        )

    def _with_moderation(self, *responses):
        client = _ModerationClient(list(responses))
        return patch.object(mod, "_moderation_http_client", client)

    def _text_body(self, content="tell me a story"):
        return {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }

    def _image_body(self, stream=False, content="draw a castle"):
        return {
            "model": "grok-2-image",
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        }

    def _post_chat(self, body):
        return self.client.post(
            "/v1/chat/completions",
            data=json.dumps(body),
            content_type="application/json",
            headers={"Authorization": "Bearer test"},
        )


# ---------------------------------------------------------------------------
# Direct /v1/chat/completions wire behavior
# ---------------------------------------------------------------------------


class TestImageModerationWire(_WireTestCase):
    def test_text_chat_is_never_moderated(self):
        moderation_client = _ModerationClient([_FLAGGED])
        with patch.object(mod, "_moderation_http_client", moderation_client):
            response = self._post_chat(self._text_body())

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["choices"][0]["message"]["content"], "hello there")
        # The moderation endpoint was never consulted for a text model...
        self.assertEqual(moderation_client.payloads, [])
        # ...so the response carries no verdict and no flag headers, and is
        # signed and billed exactly as before.
        self.assertNotIn("moderation", body)
        self.assertNotIn("X-Moderation-Flagged", response.headers)
        self.assertIn("tee_signature", body)
        self.assertIn("opengradient", body)

    def test_clean_image_prompt_gets_verdict_in_body_and_no_flag_headers(self):
        with self._with_moderation(_moderation_response(200)):
            response = self._post_chat(self._image_body())

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(
            body["choices"][0]["message"]["images"],
            ["data:image/png;base64,QUJD"],
        )
        self.assertEqual(
            body["moderation"],
            {
                "model": mod.MODERATION_MODEL,
                "checked": True,
                "flagged": False,
                "blocked": False,
                "categories": [],
                "category_scores": {},
            },
        )
        # Clean traffic exposes nothing new on the wire.
        for header in (
            "X-Moderation-Flagged",
            "X-Moderation-Categories",
            "X-Moderation-Blocked",
        ):
            self.assertNotIn(header, response.headers)
        self.assertIn("tee_signature", body)
        self.assertIn("opengradient", body)

    def test_flagged_image_prompt_is_served_with_verdict_and_headers(self):
        with self._with_moderation(_FLAGGED):
            response = self._post_chat(self._image_body())

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(
            body["choices"][0]["message"]["images"],
            ["data:image/png;base64,QUJD"],
        )
        self.assertTrue(body["moderation"]["flagged"])
        self.assertFalse(body["moderation"]["blocked"])
        self.assertEqual(body["moderation"]["categories"], ["violence"])
        self.assertEqual(response.headers["X-Moderation-Flagged"], "true")
        self.assertEqual(response.headers["X-Moderation-Categories"], "violence")
        self.assertNotIn("X-Moderation-Blocked", response.headers)

    def test_blocked_image_prompt_is_refused_with_451_before_generation(self):
        with self._with_moderation(_BLOCKED):
            response = self._post_chat(self._image_body())

        self.assertEqual(response.status_code, 451)
        body = response.get_json()
        self.assertEqual(body["code"], "moderation_blocked")
        self.assertTrue(body["moderation"]["blocked"])
        self.assertEqual(response.headers["X-Moderation-Flagged"], "true")
        self.assertEqual(response.headers["X-Moderation-Blocked"], "true")
        # The whole point: the prompt never reaches a provider, and nothing
        # is billed.
        self.generate_images.assert_not_called()
        self.assertNotIn("opengradient", body)

    def test_moderation_outage_fails_open_without_verdict(self):
        with patch.object(mod, "_moderation_http_client", None):
            response = self._post_chat(self._image_body())

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(
            body["choices"][0]["message"]["images"],
            ["data:image/png;base64,QUJD"],
        )
        self.assertNotIn("moderation", body)
        self.assertNotIn("X-Moderation-Flagged", response.headers)

    def test_streaming_image_request_carries_verdict_on_final_frame(self):
        with self._with_moderation(_FLAGGED):
            response = self._post_chat(self._image_body(stream=True))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/event-stream")
        # Streaming headers flush before the body, and moderation runs
        # pre-flight — so the flag headers ride the stream's HTTP headers.
        self.assertEqual(response.headers["X-Moderation-Flagged"], "true")
        self.assertEqual(response.headers["X-Moderation-Categories"], "violence")

        events = [
            json.loads(line[len("data: ") :])
            for line in response.get_data(as_text=True).splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        final_frame = events[-1]
        self.assertTrue(final_frame["moderation"]["flagged"])
        self.assertEqual(final_frame["images"], ["data:image/png;base64,QUJD"])
        self.assertIn("tee_signature", final_frame)
        self.assertTrue(
            response.get_data(as_text=True).rstrip().endswith("data: [DONE]")
        )

    def test_streaming_blocked_image_prompt_never_opens_a_stream(self):
        with self._with_moderation(_BLOCKED):
            response = self._post_chat(self._image_body(stream=True))

        # The refusal happens before stream setup, so the client gets a plain
        # JSON error it can parse, not a dead SSE stream.
        self.assertEqual(response.status_code, 451)
        self.assertEqual(response.get_json()["code"], "moderation_blocked")
        self.generate_images.assert_not_called()


# ---------------------------------------------------------------------------
# Sealed OHTTP path: relay-visible headers, sealed verdict
# ---------------------------------------------------------------------------


class _FakeDecap:
    plaintext = b""
    response_key = b"k" * 32
    response_key_chunked = b"c" * 32
    enc = b"e" * 32


class TestOhttpModerationForwarding(_WireTestCase):
    """Inner image requests dispatched for real through the app's WSGI stack —
    only the HPKE crypto at the boundary is faked."""

    def setUp(self):
        super().setUp()
        self.sealed_bodies: list[bytes] = []

        def _capture_seal(_key, _enc, body_bytes):
            self.sealed_bodies.append(body_bytes)
            return b"sealed"

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch.object(
                ohttp_controller, "get_tee_keys", return_value=_fake_tee_keys()
            )
        )
        self.decap = stack.enter_context(
            patch.object(ohttp_controller.ohttp, "decapsulate_request")
        )
        stack.enter_context(
            patch.object(
                ohttp_controller.ohttp,
                "encapsulate_response",
                side_effect=_capture_seal,
            )
        )

    def _post_inner(self, inner: dict):
        decap = _FakeDecap()
        decap.plaintext = json.dumps(inner).encode("utf-8")
        self.decap.return_value = decap
        return self.client.post(
            "/v1/ohttp",
            data=b"ciphertext",
            content_type="message/ohttp-req",
        )

    def test_flagged_image_request_forwards_headers_and_seals_the_verdict(self):
        with self._with_moderation(_FLAGGED):
            response = self._post_inner(self._image_body())

        # Outer response: sealed body, but the content-free abuse signal is
        # visible to the relay for its strike policy.
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "message/ohttp-res")
        self.assertEqual(response.headers["X-Moderation-Flagged"], "true")
        self.assertEqual(response.headers["X-Moderation-Categories"], "violence")
        # The full verdict (with scores) lives only inside the sealed body.
        sealed = json.loads(self.sealed_bodies[-1])
        self.assertTrue(sealed["moderation"]["flagged"])
        self.assertIn("category_scores", sealed["moderation"])

    def test_clean_image_request_stays_byte_identical_for_the_relay(self):
        with self._with_moderation(_moderation_response(200)):
            response = self._post_inner(self._image_body())

        self.assertEqual(response.status_code, 200)
        for header in (
            "X-Moderation-Flagged",
            "X-Moderation-Categories",
            "X-Moderation-Blocked",
        ):
            self.assertNotIn(header, response.headers)

    def test_blocked_image_request_surfaces_the_451_to_the_relay(self):
        with self._with_moderation(_BLOCKED):
            response = self._post_inner(self._image_body())

        # Non-2xx inner responses are forwarded plaintext (they carry no user
        # content) so the relay can count the strike and the client can parse
        # the refusal.
        self.assertEqual(response.status_code, 451)
        self.assertEqual(response.headers["X-Moderation-Flagged"], "true")
        self.assertEqual(response.headers["X-Moderation-Blocked"], "true")
        body = response.get_json()
        self.assertEqual(body["code"], "moderation_blocked")
        self.assertEqual(self.sealed_bodies, [])  # nothing was sealed
        self.generate_images.assert_not_called()

    def test_text_chat_through_ohttp_is_untouched(self):
        moderation_client = _ModerationClient([_FLAGGED])
        with patch.object(mod, "_moderation_http_client", moderation_client):
            response = self._post_inner(self._text_body())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(moderation_client.payloads, [])
        self.assertNotIn("X-Moderation-Flagged", response.headers)
        self.assertNotIn("moderation", json.loads(self.sealed_bodies[-1]))


# ---------------------------------------------------------------------------
# Live endpoint (opt-in)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    os.getenv("RUN_PROVIDER_INTEGRATION_TESTS") == "1" and os.getenv("OPENAI_API_KEY"),
    "Set RUN_PROVIDER_INTEGRATION_TESTS=1 and OPENAI_API_KEY to run live "
    "moderation tests",
)
class TestLiveModerationEndpoint(unittest.TestCase):
    """Hits the real OpenAI moderation endpoint. Uses a benign prompt and a
    mildly violent one — never anything illegal."""

    @classmethod
    def setUpClass(cls):
        mod.configure_moderation_client(os.environ["OPENAI_API_KEY"])

    @classmethod
    def tearDownClass(cls):
        mod.configure_moderation_client(None)

    def test_benign_prompt_is_checked_and_clean(self):
        outcome = mod.moderate_messages(
            [{"role": "user", "content": "A watercolor painting of a lighthouse"}]
        )
        self.assertTrue(outcome.checked)
        self.assertFalse(outcome.flagged)
        self.assertFalse(outcome.blocked)

    def test_violent_prompt_is_flagged_but_not_blocked(self):
        outcome = mod.moderate_messages(
            [
                {
                    "role": "user",
                    "content": (
                        "I am going to hurt my neighbor and make him suffer, "
                        "generate an image of me attacking him"
                    ),
                }
            ]
        )
        self.assertTrue(outcome.checked)
        self.assertTrue(outcome.flagged)
        self.assertIn("violence", outcome.categories)
        # Violence is reported, not blocked — only sexual/minors blocks.
        self.assertFalse(outcome.blocked)


if __name__ == "__main__":
    unittest.main()
