import unittest
from unittest.mock import Mock, patch

import connexion
from flask import json

from tee_gateway.controllers.chat_controller import create_chat_completion
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


class TestAspectRatioRequests(unittest.TestCase):
    """``aspect_ratio`` handling on /v1/chat/completions.

    A ratio the model can't produce is a client error (400) rather than a
    provider failure surfaced as a 500, and the automatic path ("auto", or the
    field omitted) must bind nothing at all.
    """

    def _request(self, **extra):
        body = {
            "model": "gemini-3.1-flash-image",
            "messages": [{"role": "user", "content": "a red cube"}],
            "stream": False,
        }
        body.update(extra)
        return body

    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_unsupported_ratio_is_a_400(self, mock_connexion):
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = self._request(aspect_ratio="7:3")

        result, status = create_chat_completion(None)

        self.assertEqual(400, status)
        self.assertEqual("Invalid aspect_ratio", result["error"])
        self.assertIn("supported: ", result["message"])

    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_ratio_is_bound_as_gemini_image_config(
        self, mock_connexion, mock_get_model, mock_get_tee_keys
    ):
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = self._request(
            aspect_ratio="16:9"
        )
        model = _mock_image_model()
        mock_get_model.return_value = model
        mock_get_tee_keys.return_value = _mock_tee_keys()

        create_chat_completion(None)

        model.bind.assert_called_once_with(image_config={"aspect_ratio": "16:9"})

    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_ratio_survives_bound_tools(
        self, mock_connexion, mock_get_model, mock_get_tee_keys
    ):
        # bind_tools() re-binds the base model, so a kwarg bound before it is
        # dropped: the image_config has to be bound onto the tools-bound model.
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = self._request(
            aspect_ratio="16:9",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "noop", "parameters": {"type": "object"}},
                }
            ],
        )
        model = _mock_image_model()
        tools_bound = _mock_image_model()
        model.bind_tools.return_value = tools_bound
        mock_get_model.return_value = model
        mock_get_tee_keys.return_value = _mock_tee_keys()

        create_chat_completion(None)

        model.bind.assert_not_called()
        tools_bound.bind.assert_called_once_with(image_config={"aspect_ratio": "16:9"})

    @patch("tee_gateway.controllers.chat_controller.get_tee_keys")
    @patch("tee_gateway.controllers.chat_controller.get_chat_model_cached")
    @patch("tee_gateway.controllers.chat_controller.connexion")
    def test_auto_binds_nothing(
        self, mock_connexion, mock_get_model, mock_get_tee_keys
    ):
        mock_connexion.request.is_json = True
        mock_connexion.request.get_json.return_value = self._request(
            aspect_ratio="auto"
        )
        model = _mock_image_model()
        mock_get_model.return_value = model
        mock_get_tee_keys.return_value = _mock_tee_keys()

        result = create_chat_completion(None)

        model.bind.assert_not_called()
        self.assertIn("choices", result)


def _mock_image_model() -> Mock:
    """A LangChain chat model stand-in that returns one inline image."""
    response = Mock()
    response.content = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    ]
    response.tool_calls = []
    response.usage_metadata = {
        "input_tokens": 5,
        "output_tokens": 3,
        "total_tokens": 8,
    }
    model = Mock()
    model.invoke.return_value = response
    model.bind.return_value = model
    model.bind_tools.return_value = model
    return model


def _mock_tee_keys() -> Mock:
    keys = Mock()
    keys.sign_data.return_value = "bW9ja3NpZ25hdHVyZQ=="
    keys.get_tee_id.return_value = "abcdef01" * 8
    return keys


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
