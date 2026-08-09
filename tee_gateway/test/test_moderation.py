"""Tests for in-enclave prompt moderation (tee_gateway/moderation.py) and its
wiring into the chat controller.

The moderation HTTP client is faked the same way test_web_search.py fakes the
Exa client — a recording stand-in patched onto the module global — so no test
touches the network.
"""

import json
import unittest
from unittest.mock import Mock, patch

import httpx

import tee_gateway.moderation as mod
from tee_gateway.moderation import (
    BLOCKED_CATEGORIES,
    MAX_IMAGE_PARTS,
    ModerationOutcome,
    extract_moderation_input,
    moderate_messages,
)


def _moderation_response(
    status: int = 200,
    *,
    flagged: bool = False,
    categories: dict | None = None,
    scores: dict | None = None,
    body: dict | None = None,
):
    response = Mock()
    response.status_code = status
    if body is None:
        body = {
            "id": "modr-test",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": flagged,
                    "categories": categories or {},
                    "category_scores": scores or {},
                }
            ],
        }
    response.json.return_value = body
    response.text = json.dumps(body)
    return response


class _ModerationClient:
    """Records the payloads posted to /moderations and replays queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads: list[dict] = []

    def post(self, path, json=None):  # noqa: A002 - matches httpx.Client.post
        self.payloads.append(json)
        if not self.responses:
            return _moderation_response(200)
        head = self.responses.pop(0)
        if isinstance(head, Exception):
            raise head
        return head


def _with_moderation(*responses):
    client = _ModerationClient(responses)
    return patch.object(mod, "_moderation_http_client", client), client


def _user(content):
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestAvailability(unittest.TestCase):
    def test_availability_tracks_the_injected_key(self):
        mod.configure_moderation_client("test-key")
        try:
            self.assertTrue(mod.moderation_available())
        finally:
            mod.configure_moderation_client(None)
        self.assertFalse(mod.moderation_available())

    def test_no_client_means_unchecked_not_blocked(self):
        with patch.object(mod, "_moderation_http_client", None):
            outcome = moderate_messages([_user("anything")])
        self.assertFalse(outcome.checked)
        self.assertFalse(outcome.flagged)
        self.assertFalse(outcome.blocked)


# ---------------------------------------------------------------------------
# Input extraction
# ---------------------------------------------------------------------------


class TestInputExtraction(unittest.TestCase):
    def test_takes_the_newest_user_turn_only(self):
        messages = [
            {"role": "system", "content": "sys"},
            _user("old prompt"),
            {"role": "assistant", "content": "answer"},
            _user("new prompt"),
            {"role": "tool", "content": "tool output", "tool_call_id": "t1"},
        ]
        text, images = extract_moderation_input(messages)
        self.assertEqual(text, "new prompt")
        self.assertEqual(images, [])

    def test_multimodal_content_splits_text_and_images(self):
        content = [
            {"type": "text", "text": "caption"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "image", "base64": "BBBB", "mime_type": "image/jpeg"},
            {"type": "file", "file": {"filename": "doc.pdf"}},
        ]
        text, images = extract_moderation_input([_user(content)])
        self.assertEqual(text, "caption")
        self.assertEqual(
            images,
            ["data:image/png;base64,AAAA", "data:image/jpeg;base64,BBBB"],
        )

    def test_non_data_non_http_image_urls_are_skipped(self):
        content = [
            {"type": "image_url", "image_url": {"url": "ftp://host/x.png"}},
            {"type": "image_url", "image_url": {"url": "https://host/x.png"}},
        ]
        _, images = extract_moderation_input([_user(content)])
        self.assertEqual(images, ["https://host/x.png"])

    def test_no_user_message_yields_nothing(self):
        text, images = extract_moderation_input([{"role": "system", "content": "sys"}])
        self.assertEqual(text, "")
        self.assertEqual(images, [])
        patcher, client = _with_moderation()
        with patcher:
            outcome = moderate_messages([{"role": "system", "content": "sys"}])
        self.assertFalse(outcome.checked)
        self.assertEqual(client.payloads, [])


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


class TestVerdicts(unittest.TestCase):
    def test_clean_prompt_is_checked_and_unflagged(self):
        patcher, client = _with_moderation(_moderation_response(200))
        with patcher:
            outcome = moderate_messages([_user("hello")])
        self.assertTrue(outcome.checked)
        self.assertFalse(outcome.flagged)
        self.assertFalse(outcome.blocked)
        self.assertEqual(outcome.headers(), {})
        self.assertEqual(
            client.payloads[0],
            {"model": mod.MODERATION_MODEL, "input": "hello"},
        )

    def test_flagged_prompt_reports_categories_and_headers(self):
        patcher, _ = _with_moderation(
            _moderation_response(
                200,
                flagged=True,
                categories={"violence": True, "sexual": False},
                scores={"violence": 0.91, "sexual": 0.0001},
            )
        )
        with patcher:
            outcome = moderate_messages([_user("bad prompt")])
        self.assertTrue(outcome.flagged)
        self.assertFalse(outcome.blocked)
        self.assertEqual(outcome.categories, ("violence",))
        self.assertEqual(outcome.category_scores, {"violence": 0.91})
        self.assertEqual(
            outcome.headers(),
            {
                "X-Moderation-Flagged": "true",
                "X-Moderation-Categories": "violence",
            },
        )

    def test_blocked_category_blocks(self):
        patcher, _ = _with_moderation(
            _moderation_response(
                200,
                flagged=True,
                categories={"sexual/minors": True, "sexual": True},
                scores={"sexual/minors": 0.99, "sexual": 0.8},
            )
        )
        with patcher:
            outcome = moderate_messages([_user("bad prompt")])
        self.assertTrue(outcome.blocked)
        self.assertEqual(outcome.categories, ("sexual", "sexual/minors"))
        self.assertEqual(outcome.headers()["X-Moderation-Blocked"], "true")
        block = outcome.to_response_dict()
        self.assertTrue(block["blocked"])
        self.assertEqual(block["model"], mod.MODERATION_MODEL)

    def test_blocked_categories_is_csam_only_by_default(self):
        self.assertEqual(BLOCKED_CATEGORIES, frozenset({"sexual/minors"}))

    def test_images_ride_as_image_url_parts(self):
        patcher, client = _with_moderation(_moderation_response(200))
        content = [
            {"type": "text", "text": "edit this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
        ]
        with patcher:
            moderate_messages([_user(content)])
        self.assertEqual(
            client.payloads[0]["input"],
            [
                {"type": "text", "text": "edit this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA"},
                },
            ],
        )

    def test_image_count_is_capped(self):
        patcher, client = _with_moderation(_moderation_response(200))
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{i}"}}
            for i in range(MAX_IMAGE_PARTS + 4)
        ]
        with patcher:
            moderate_messages([_user(content)])
        image_parts = [
            p for p in client.payloads[0]["input"] if p["type"] == "image_url"
        ]
        self.assertEqual(len(image_parts), MAX_IMAGE_PARTS)


# ---------------------------------------------------------------------------
# Failure modes (all fail-open)
# ---------------------------------------------------------------------------


class TestFailureModes(unittest.TestCase):
    def test_transport_error_fails_open(self):
        patcher, _ = _with_moderation(httpx.ConnectError("boom"))
        with patcher:
            outcome = moderate_messages([_user("hello")])
        self.assertFalse(outcome.checked)
        self.assertFalse(outcome.flagged)
        self.assertEqual(outcome.headers(), {})

    def test_non_200_fails_open(self):
        patcher, _ = _with_moderation(
            _moderation_response(429, body={"error": {"message": "rate limited"}})
        )
        with patcher:
            outcome = moderate_messages([_user("hello")])
        self.assertFalse(outcome.checked)

    def test_malformed_body_fails_open(self):
        patcher, _ = _with_moderation(_moderation_response(200, body={"nope": 1}))
        with patcher:
            outcome = moderate_messages([_user("hello")])
        self.assertFalse(outcome.checked)


# ---------------------------------------------------------------------------
# Controller integration
# ---------------------------------------------------------------------------


class TestControllerIntegration(unittest.TestCase):
    """create_chat_completion refuses blocked requests before provider work."""

    def _request_json(self, stream=False):
        return {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "prompt"}],
            "stream": stream,
        }

    def _call_controller(self, outcome, stream=False):
        from tee_gateway.controllers import chat_controller

        fake_request = Mock()
        fake_request.is_json = True
        fake_request.get_json.return_value = self._request_json(stream)
        with (
            patch.object(chat_controller.connexion, "request", fake_request),
            patch.object(chat_controller, "moderate_messages", return_value=outcome),
            patch.object(
                chat_controller,
                "_create_streaming_response",
                return_value="streamed",
            ) as streaming,
            patch.object(
                chat_controller,
                "_create_non_streaming_response",
                return_value="non-streamed",
            ) as non_streaming,
        ):
            result = chat_controller.create_chat_completion(None)
        return result, streaming, non_streaming

    def test_blocked_request_returns_451_and_never_reaches_a_provider(self):
        outcome = ModerationOutcome(
            checked=True,
            flagged=True,
            blocked=True,
            categories=("sexual/minors",),
            category_scores={"sexual/minors": 0.99},
        )
        result, streaming, non_streaming = self._call_controller(outcome)
        body, status, headers = result
        self.assertEqual(status, 451)
        self.assertEqual(body["code"], "moderation_blocked")
        self.assertTrue(body["moderation"]["blocked"])
        self.assertEqual(headers["X-Moderation-Blocked"], "true")
        streaming.assert_not_called()
        non_streaming.assert_not_called()

    def test_flagged_but_not_blocked_proceeds(self):
        outcome = ModerationOutcome(
            checked=True, flagged=True, categories=("violence",)
        )
        result, _, non_streaming = self._call_controller(outcome)
        self.assertEqual(result, "non-streamed")
        non_streaming.assert_called_once()
        self.assertIs(non_streaming.call_args.args[1], outcome)

    def test_unchecked_outcome_proceeds(self):
        result, streaming, _ = self._call_controller(
            ModerationOutcome(checked=False), stream=True
        )
        self.assertEqual(result, "streamed")
        streaming.assert_called_once()


class TestOhttpHeaderForwarding(unittest.TestCase):
    def test_moderation_headers_are_forwarded_to_the_relay(self):
        from tee_gateway.controllers.ohttp_controller import _should_forward_header

        self.assertTrue(_should_forward_header("X-Moderation-Flagged"))
        self.assertTrue(_should_forward_header("x-moderation-categories"))
        self.assertTrue(_should_forward_header("X-Moderation-Blocked"))
        self.assertFalse(_should_forward_header("X-Something-Else"))


if __name__ == "__main__":
    unittest.main()
