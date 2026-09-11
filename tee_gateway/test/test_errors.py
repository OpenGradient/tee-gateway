"""Provider failures must be told apart from gateway bugs, with the status."""

import unittest

import anthropic
import httpx
import openai

from tee_gateway.errors import describe_exception, error_response, http_status_for


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://provider"))


class TestDescribeException(unittest.TestCase):
    def test_provider_overload_is_a_retryable_502(self):
        exc = anthropic.APIStatusError(
            "Error code: 529 - {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}",
            response=_response(529),
            body={"type": "error", "error": {"type": "overloaded_error"}},
        )
        payload, status = error_response(exc)
        self.assertEqual(502, status)
        self.assertEqual("provider", payload["source"])
        self.assertEqual(529, payload["provider_status"])
        self.assertTrue(payload["retryable"])
        self.assertEqual("APIStatusError", payload["exception_type"])
        self.assertIn("overloaded_error", payload["error"])

    def test_provider_400_passes_through_and_is_not_retryable(self):
        exc = openai.BadRequestError(
            "Unsupported parameter: 'temperature'",
            response=_response(400),
            body={"error": {"message": "Unsupported parameter: 'temperature'"}},
        )
        payload, status = error_response(exc)
        self.assertEqual(400, status)
        self.assertEqual("provider", payload["source"])
        self.assertEqual(400, payload["provider_status"])
        self.assertFalse(payload["retryable"])
        self.assertEqual("Unsupported parameter: 'temperature'", payload["error"])

    def test_request_class_provider_4xx_keep_their_status(self):
        for provider_status in (404, 413, 415, 422):
            exc = openai.APIStatusError(
                "nope", response=_response(provider_status), body=None
            )
            self.assertEqual(provider_status, http_status_for(exc), provider_status)

    def test_provider_account_errors_become_502_not_the_clients_problem(self):
        # 401/403: the gateway's key, not the client's credentials. 402: must
        # never reach the relay, whose x402 client would read it as a payment
        # challenge from this gateway.
        for provider_status in (401, 402, 403, 407):
            exc = openai.APIStatusError(
                "denied", response=_response(provider_status), body=None
            )
            payload, status = error_response(exc)
            self.assertEqual(502, status, provider_status)
            self.assertEqual(provider_status, payload["provider_status"])
            self.assertFalse(payload["retryable"])

    def test_provider_rate_limit_is_a_503_and_408_a_504(self):
        exc = openai.RateLimitError(
            "Rate limit reached", response=_response(429), body=None
        )
        self.assertEqual(503, http_status_for(exc))
        exc = openai.APIStatusError("slow", response=_response(408), body=None)
        self.assertEqual(504, http_status_for(exc))

    def test_provider_rate_limit_is_retryable(self):
        exc = openai.RateLimitError(
            "Rate limit reached", response=_response(429), body=None
        )
        payload = describe_exception(exc)
        self.assertEqual(429, payload["provider_status"])
        self.assertTrue(payload["retryable"])

    def test_provider_connection_reset_is_a_retryable_502(self):
        exc = httpx.ReadError("[Errno 104] Connection reset by peer")
        payload, status = error_response(exc)
        self.assertEqual(502, status)
        self.assertEqual("provider", payload["source"])
        self.assertNotIn("provider_status", payload)
        self.assertTrue(payload["retryable"])

    def test_provider_timeout_is_a_504(self):
        self.assertEqual(504, http_status_for(httpx.ReadTimeout("timed out")))
        sdk_timeout = openai.APITimeoutError(
            request=httpx.Request("POST", "https://provider")
        )
        self.assertEqual(504, http_status_for(sdk_timeout))
        self.assertTrue(describe_exception(sdk_timeout)["retryable"])

    def test_gateway_bug_stays_a_500(self):
        payload, status = error_response(ValueError("Unsupported provider: nope"))
        self.assertEqual(500, status)
        self.assertEqual(
            {
                "error": "Unsupported provider: nope",
                "exception_type": "ValueError",
                "source": "gateway",
                "retryable": False,
            },
            payload,
        )

    def test_empty_message_uses_the_fallback(self):
        payload = describe_exception(RuntimeError(), fallback="Stream setup failed")
        self.assertEqual("Stream setup failed", payload["error"])

    def test_long_messages_are_truncated(self):
        payload = describe_exception(RuntimeError("x" * 5000))
        self.assertLess(len(payload["error"]), 700)
        self.assertTrue(payload["error"].endswith("…"))


if __name__ == "__main__":
    unittest.main()
