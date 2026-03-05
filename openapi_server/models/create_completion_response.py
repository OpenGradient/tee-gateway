class CreateCompletionResponse:
    """Stub data holder for completion responses (endpoint not implemented)."""

    def __init__(self, id=None, choices=None, created=None, model=None,
                 system_fingerprint=None, object=None, usage=None):
        self.id = id
        self.choices = choices
        self.created = created
        self.model = model
        self.system_fingerprint = system_fingerprint
        self.object = object
        self.usage = usage

    @classmethod
    def from_dict(cls, dikt) -> 'CreateCompletionResponse':
        if not isinstance(dikt, dict):
            return dikt
        return cls(
            id=dikt.get('id'),
            choices=dikt.get('choices'),
            created=dikt.get('created'),
            model=dikt.get('model'),
            system_fingerprint=dikt.get('system_fingerprint'),
            object=dikt.get('object'),
            usage=dikt.get('usage'),
        )
