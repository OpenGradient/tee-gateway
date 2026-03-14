import unittest

from flask import json

from tee_gateway.test import BaseTestCase


class TestCompletionsController(BaseTestCase):
    """CompletionsController integration test stubs"""

    @unittest.skip("Integration test - requires HTTP backend")
    def test_create_completion(self):
        """Test case for create_completion

        Creates a completion for the provided prompt and parameters via HTTP backend.
        Tests the HTTP-based completion endpoint that forwards requests to the TEE server.
        """
        body = {
            "logit_bias": {"key": 1},
            "seed": -2147483648,
            "max_tokens": 16,
            "presence_penalty": 0.25495066265333133,
            "echo": False,
            "suffix": "test.",
            "n": 1,
            "logprobs": 2,
            "top_p": 1,
            "frequency_frequency": 0.4109824732281613,
            "best_of": 1,
            "stop": "\n",
            "stream": False,
            "temperature": 1,
            "model": "CreateCompletionRequest_model",
            "stream_options": {"include_usage": True},
            "prompt": "This is a test.",
            "user": "user-1234",
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer special-key",
        }
        response = self.client.open(
            "/v1/completions",
            method="POST",
            headers=headers,
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assert200(response, "Response body is : " + response.data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
