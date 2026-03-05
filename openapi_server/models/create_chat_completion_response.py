class CreateChatCompletionResponse:
    """Minimal data holder for chat completion responses (used for validation)."""

    def __init__(self, id=None, choices=None, created=None, model=None,
                 service_tier=None, system_fingerprint=None, object=None, usage=None):
        self.id = id
        self.choices = choices
        self.created = created
        self.model = model
        self.service_tier = service_tier
        self.system_fingerprint = system_fingerprint
        self.object = object
        self.usage = usage

    @classmethod
    def from_dict(cls, dikt) -> 'CreateChatCompletionResponse':
        if not isinstance(dikt, dict):
            return dikt
        return cls(
            id=dikt.get('id'),
            choices=dikt.get('choices'),
            created=dikt.get('created'),
            model=dikt.get('model'),
            service_tier=dikt.get('service_tier'),
            system_fingerprint=dikt.get('system_fingerprint'),
            object=dikt.get('object'),
            usage=dikt.get('usage'),
        )
