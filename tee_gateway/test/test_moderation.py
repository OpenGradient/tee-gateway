"""Unit tests for the in-enclave moderation gate (tee_gateway.moderation)."""

import unittest

from tee_gateway import moderation
from tee_gateway.moderation import (
    ModerationBackend,
    ModerationDecision,
    ModerationUnavailable,
    StubBackend,
)


class _AllowBackend(ModerationBackend):
    name = "allow"

    def check(self, texts, image_data_uris=None):
        return ModerationDecision(allowed=True, backend=self.name)


class _BlockBackend(ModerationBackend):
    name = "block"

    def check(self, texts, image_data_uris=None):
        return ModerationDecision(
            allowed=False, categories=["sexual/minors"], backend=self.name
        )


class _BrokenBackend(ModerationBackend):
    name = "broken"

    def check(self, texts, image_data_uris=None):
        raise ModerationUnavailable("upstream down")


class ModerationEnforceTest(unittest.TestCase):
    def tearDown(self):
        # Reset module state so tests don't leak configuration into each other.
        moderation.configure_moderation(StubBackend(), enabled=False, fail_closed=True)

    def test_disabled_gate_allows(self):
        moderation.configure_moderation(_BlockBackend(), enabled=False)
        self.assertIsNone(moderation.enforce(["anything"]))

    def test_allow_backend_passes(self):
        moderation.configure_moderation(_AllowBackend(), enabled=True)
        self.assertIsNone(moderation.enforce(["hello"]))

    def test_block_returns_403_with_flag_header(self):
        moderation.configure_moderation(_BlockBackend(), enabled=True)
        result = moderation.enforce(["bad content"])
        self.assertIsNotNone(result)
        assert result is not None
        _body, status, headers = result
        self.assertEqual(status, 403)
        # The relay bans off this header; the categories carry a policy-class
        # label only, never client content.
        self.assertEqual(headers.get(moderation.MODERATION_FLAG_HEADER), "1")
        self.assertEqual(
            headers.get(moderation.MODERATION_CATEGORIES_HEADER), "sexual/minors"
        )

    def test_fail_closed_returns_503_without_flag(self):
        moderation.configure_moderation(
            _BrokenBackend(), enabled=True, fail_closed=True
        )
        result = moderation.enforce(["x"])
        self.assertIsNotNone(result)
        assert result is not None
        _body, status, headers = result
        self.assertEqual(status, 503)
        # A screening outage is not a policy hit — must not flag the user.
        self.assertNotIn(moderation.MODERATION_FLAG_HEADER, headers)

    def test_fail_open_allows_on_error(self):
        moderation.configure_moderation(
            _BrokenBackend(), enabled=True, fail_closed=False
        )
        self.assertIsNone(moderation.enforce(["x"]))


class ExtractionTest(unittest.TestCase):
    def test_texts_from_messages_dicts(self):
        messages = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "should be ignored"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part text"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            },
        ]
        texts = moderation.texts_from_messages(messages)
        self.assertIn("be nice", texts)
        self.assertIn("hello", texts)
        self.assertIn("part text", texts)
        self.assertNotIn("should be ignored", texts)

    def test_images_from_messages_inline_only(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/x.png"},
                    },
                ],
            },
        ]
        images = moderation.images_from_messages(messages)
        self.assertEqual(images, ["data:image/png;base64,AAAA"])

    def test_payment_safety_identifier_no_context(self):
        # Outside a request/payment context this must not raise, just return None.
        self.assertIsNone(moderation.payment_safety_identifier())

    def test_flag_header_is_forwarded_to_relay(self):
        # The OHTTP layer must surface the moderation flag to the relay, or the
        # per-user ban signal never leaves the enclave.
        from tee_gateway.controllers.ohttp_controller import _should_forward_header

        self.assertTrue(_should_forward_header(moderation.MODERATION_FLAG_HEADER))
        self.assertTrue(_should_forward_header(moderation.MODERATION_CATEGORIES_HEADER))


if __name__ == "__main__":
    unittest.main()
