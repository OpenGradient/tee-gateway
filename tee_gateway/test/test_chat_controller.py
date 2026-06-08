import unittest

import connexion
from flask import json

from tee_gateway.encoder import JSONEncoder
from tee_gateway.test import BaseTestCase


def _build_app():
    """Build a connexion app from the OpenAPI spec for request-validation tests.

    Kept independent of ``BaseTestCase`` (which needs the optional
    ``flask_testing`` dep) so this runs in the lean ``test`` dep group.
    """
    app = connexion.App(__name__, specification_dir="../openapi/")
    app.app.json_encoder = JSONEncoder
    app.add_api("openapi.yaml", pythonic_params=True)
    return app.app


class TestUserMessageContentPartValidation(unittest.TestCase):
    """Schema-validation tests for multimodal user-message content parts.

    Regression guard for the "secure attachments" feature: PDF attachments
    arrive as OpenAI ``file`` content parts and must survive connexion's
    request-body validation (the OHTTP inner request runs through the full
    validation pipeline). Before the ``file`` branch was added to the user
    content-part ``oneOf``, PDFs were rejected with a 400 while images passed.
    """

    def setUp(self):
        self.client = _build_app().test_client()

    def _post(self, part):
        body = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}, part]}
            ],
        }
        return self.client.post(
            "/v1/chat/completions",
            data=json.dumps(body),
            content_type="application/json",
            headers={"Authorization": "Bearer test"},
        )

    def test_image_part_passes_schema_validation(self):
        resp = self._post(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
            }
        )
        # Passes validation; only fails later (500) because no provider keys
        # are injected in this unit test. The point is it is NOT a 400.
        self.assertNotEqual(400, resp.status_code, resp.data.decode("utf-8"))

    def test_pdf_file_part_passes_schema_validation(self):
        resp = self._post(
            {
                "type": "file",
                "file": {
                    "filename": "doc.pdf",
                    "file_data": "data:application/pdf;base64,JVBERi0x",
                },
            }
        )
        self.assertNotEqual(400, resp.status_code, resp.data.decode("utf-8"))


class TestChatController(BaseTestCase):
    """ChatController integration test stubs"""

    @unittest.skip("Integration test - requires HTTP backend")
    def test_create_chat_completion(self):
        """Test case for create_chat_completion

        Creates a model response for the given chat conversation via HTTP backend.
        Tests the HTTP-based chat completion endpoint that forwards requests to the TEE server.
        """
        body = {
            "reasoning_effort": "medium",
            "top_logprobs": 2,
            "metadata": {"key": "metadata"},
            "logit_bias": {"key": 6},
            "seed": 2147483647,
            "functions": [
                {
                    "name": "name",
                    "description": "description",
                    "parameters": {"key": ""},
                },
                {
                    "name": "name",
                    "description": "description",
                    "parameters": {"key": ""},
                },
                {
                    "name": "name",
                    "description": "description",
                    "parameters": {"key": ""},
                },
                {
                    "name": "name",
                    "description": "description",
                    "parameters": {"key": ""},
                },
                {
                    "name": "name",
                    "description": "description",
                    "parameters": {"key": ""},
                },
            ],
            "function_call": "none",
            "presence_penalty": -1.079145645226094,
            "tools": [
                {
                    "function": {
                        "name": "name",
                        "description": "description",
                        "strict": False,
                        "parameters": {"key": ""},
                    },
                    "type": "function",
                },
                {
                    "function": {
                        "name": "name",
                        "description": "description",
                        "strict": False,
                        "parameters": {"key": ""},
                    },
                    "type": "function",
                },
            ],
            "logprobs": False,
            "top_p": 1,
            "max_completion_tokens": 5,
            "frequency_penalty": -1.6796687238155954,
            "modalities": ["text", "text"],
            "response_format": {"type": "text"},
            "stream": False,
            "temperature": 1,
            "tool_choice": "none",
            "model": "gpt-4o",
            "service_tier": "auto",
            "audio": {"voice": "alloy", "format": "wav"},
            "max_tokens": 5,
            "store": False,
            "n": 1,
            "stop": "CreateChatCompletionRequest_stop",
            "parallel_tool_calls": True,
            "prediction": {"type": "content", "content": "PredictionContent_content"},
            "messages": [
                {
                    "role": "developer",
                    "name": "name",
                    "content": "ChatCompletionRequestDeveloperMessage_content",
                },
                {
                    "role": "developer",
                    "name": "name",
                    "content": "ChatCompletionRequestDeveloperMessage_content",
                },
            ],
            "stream_options": {"include_usage": True},
            "user": "user-1234",
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer special-key",
        }
        response = self.client.open(
            "/v1/chat/completions",
            method="POST",
            headers=headers,
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assert200(response, "Response body is : " + response.data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
