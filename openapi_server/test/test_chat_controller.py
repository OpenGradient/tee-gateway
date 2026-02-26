import unittest

from flask import json

from openapi_server.models.create_chat_completion_request import CreateChatCompletionRequest  # noqa: E501
from openapi_server.models.create_chat_completion_response import CreateChatCompletionResponse  # noqa: E501
from openapi_server.test import BaseTestCase


class TestChatController(BaseTestCase):
    """ChatController integration test stubs"""

    @unittest.skip("Integration test - requires HTTP backend")
    def test_create_chat_completion(self):
        """Test case for create_chat_completion

        Creates a model response for the given chat conversation via HTTP backend.
        Tests the HTTP-based chat completion endpoint that forwards requests to the TEE server.
        """
        body = {"reasoning_effort":"medium","top_logprobs":2,"metadata":{"key":"metadata"},"logit_bias":{"key":6},"seed":2147483647,"functions":[{"name":"name","description":"description","parameters":{"key":""}},{"name":"name","description":"description","parameters":{"key":""}},{"name":"name","description":"description","parameters":{"key":""}},{"name":"name","description":"description","parameters":{"key":""}},{"name":"name","description":"description","parameters":{"key":""}}],"function_call":"none","presence_penalty":-1.079145645226094,"tools":[{"function":{"name":"name","description":"description","strict":False,"parameters":{"key":""}},"type":"function"},{"function":{"name":"name","description":"description","strict":False,"parameters":{"key":""}},"type":"function"}],"logprobs":False,"top_p":1,"max_completion_tokens":5,"frequency_penalty":-1.6796687238155954,"modalities":["text","text"],"response_format":{"type":"text"},"stream":False,"temperature":1,"tool_choice":"none","model":"gpt-4o","service_tier":"auto","audio":{"voice":"alloy","format":"wav"},"max_tokens":5,"store":False,"n":1,"stop":"CreateChatCompletionRequest_stop","parallel_tool_calls":True,"prediction":{"type":"content","content":"PredictionContent_content"},"messages":[{"role":"developer","name":"name","content":"ChatCompletionRequestDeveloperMessage_content"},{"role":"developer","name":"name","content":"ChatCompletionRequestDeveloperMessage_content"}],"stream_options":{"include_usage":True},"user":"user-1234"}
        headers = { 
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': 'Bearer special-key',
        }
        response = self.client.open(
            '/v1/chat/completions',
            method='POST',
            headers=headers,
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()