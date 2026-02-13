import connexion
from typing import Dict
from typing import Tuple
from typing import Union

from openapi_server.models.create_completion_request import CreateCompletionRequest  # noqa: E501
from openapi_server.models.create_completion_response import CreateCompletionResponse  # noqa: E501
from openapi_server import util


def create_completion(body):  # noqa: E501
    """Creates a completion for the provided prompt and parameters.

     # noqa: E501

    :param body: 
    :type body: dict | bytes

    :rtype: Union[CreateCompletionResponse, Tuple[CreateCompletionResponse, int], Tuple[CreateCompletionResponse, int, Dict[str, str]]
    """
    body = body
    if connexion.request.is_json:
        body = CreateCompletionRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'
