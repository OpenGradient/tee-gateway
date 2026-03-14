class CreateCompletionRequest:
    """Stub data holder for completion requests (endpoint not implemented)."""

    def __init__(
        self,
        model=None,
        prompt=None,
        best_of=1,
        echo=False,
        frequency_penalty=0,
        logit_bias=None,
        logprobs=None,
        max_tokens=16,
        n=1,
        presence_penalty=0,
        seed=None,
        stop=None,
        stream=False,
        stream_options=None,
        suffix=None,
        temperature=1,
        top_p=1,
        user=None,
    ):
        self.model = model
        self.prompt = prompt
        self.best_of = best_of
        self.echo = echo
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.logprobs = logprobs
        self.max_tokens = max_tokens
        self.n = n
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.stop = stop
        self.stream = stream
        self.stream_options = stream_options
        self.suffix = suffix
        self.temperature = temperature
        self.top_p = top_p
        self.user = user

    @classmethod
    def from_dict(cls, dikt) -> "CreateCompletionRequest":
        if not isinstance(dikt, dict):
            return dikt
        known = {
            "model",
            "prompt",
            "best_of",
            "echo",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "suffix",
            "temperature",
            "top_p",
            "user",
        }
        return cls(**{k: v for k, v in dikt.items() if k in known})
