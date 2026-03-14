from tee_gateway.models.base_model import Model
from tee_gateway import util  # noqa: F401


class _ToolCallFunction:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id=None, type=None, function=None):
        self.id = id
        self.type = type
        self.function = function


class ChatCompletionRequestAssistantMessage(Model):

    def __init__(self, content=None, refusal=None, role=None, name=None, audio=None, tool_calls=None, function_call=None):  # noqa: E501
        self.openapi_types = {
            'content': object,
            'refusal': str,
            'role': str,
            'name': str,
            'audio': object,
            'tool_calls': object,
            'function_call': object,
        }

        self.attribute_map = {
            'content': 'content',
            'refusal': 'refusal',
            'role': 'role',
            'name': 'name',
            'audio': 'audio',
            'tool_calls': 'tool_calls',
            'function_call': 'function_call',
        }

        self._content = content
        self._refusal = refusal
        self._role = role
        self._name = name
        self._audio = audio
        self._tool_calls = tool_calls
        self._function_call = function_call

    @classmethod
    def from_dict(cls, dikt) -> 'ChatCompletionRequestAssistantMessage':
        raw_tool_calls = dikt.get('tool_calls')
        tool_calls = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                func_raw = tc.get('function', {}) if isinstance(tc, dict) else {}
                tool_calls.append(_ToolCall(
                    id=tc.get('id') if isinstance(tc, dict) else None,
                    type=tc.get('type') if isinstance(tc, dict) else None,
                    function=_ToolCallFunction(
                        name=func_raw.get('name'),
                        arguments=func_raw.get('arguments'),
                    ),
                ))
        return cls(
            content=dikt.get('content'),
            refusal=dikt.get('refusal'),
            role=dikt.get('role'),
            name=dikt.get('name'),
            audio=dikt.get('audio'),
            tool_calls=tool_calls,
            function_call=dikt.get('function_call'),
        )

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, content):
        self._content = content

    @property
    def refusal(self) -> str:
        return self._refusal

    @refusal.setter
    def refusal(self, refusal: str):
        self._refusal = refusal

    @property
    def role(self) -> str:
        return self._role

    @role.setter
    def role(self, role: str):
        allowed_values = ["assistant"]  # noqa: E501
        if role not in allowed_values:
            raise ValueError(
                "Invalid value for `role` ({0}), must be one of {1}"
                .format(role, allowed_values)
            )
        self._role = role

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def audio(self):
        return self._audio

    @audio.setter
    def audio(self, audio):
        self._audio = audio

    @property
    def tool_calls(self):
        return self._tool_calls

    @tool_calls.setter
    def tool_calls(self, tool_calls):
        self._tool_calls = tool_calls

    @property
    def function_call(self):
        return self._function_call

    @function_call.setter
    def function_call(self, function_call):
        self._function_call = function_call
