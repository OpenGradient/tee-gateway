class CreateChatCompletionRequest:
    """Minimal data holder for chat completion requests."""

    def __init__(self, messages=None, model=None, store=False, reasoning_effort='medium',
                 metadata=None, frequency_penalty=0, logit_bias=None, logprobs=False,
                 top_logprobs=None, max_tokens=None, max_completion_tokens=None, n=1,
                 modalities=None, prediction=None, audio=None, presence_penalty=0,
                 response_format=None, seed=None, service_tier='auto', stop=None,
                 stream=False, stream_options=None, temperature=1, top_p=1,
                 tools=None, tool_choice=None, parallel_tool_calls=True,
                 user=None, function_call=None, functions=None):
        self.messages = messages
        self.model = model
        self.store = store
        self.reasoning_effort = reasoning_effort
        self.metadata = metadata
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.n = n
        self.modalities = modalities
        self.prediction = prediction
        self.audio = audio
        self.presence_penalty = presence_penalty
        self.response_format = response_format
        self.seed = seed
        self.service_tier = service_tier
        self.stop = stop
        self.stream = stream
        self.stream_options = stream_options
        self.temperature = temperature
        self.top_p = top_p
        self.tools = tools
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.user = user
        self.function_call = function_call
        self.functions = functions

    @classmethod
    def from_dict(cls, dikt) -> 'CreateChatCompletionRequest':
        if not isinstance(dikt, dict):
            return dikt
        known = {
            'messages', 'model', 'store', 'reasoning_effort', 'metadata',
            'frequency_penalty', 'logit_bias', 'logprobs', 'top_logprobs',
            'max_tokens', 'max_completion_tokens', 'n', 'modalities', 'prediction',
            'audio', 'presence_penalty', 'response_format', 'seed', 'service_tier',
            'stop', 'stream', 'stream_options', 'temperature', 'top_p', 'tools',
            'tool_choice', 'parallel_tool_calls', 'user', 'function_call', 'functions',
        }
        return cls(**{k: v for k, v in dikt.items() if k in known})
