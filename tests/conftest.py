class FakeTokenizer:
    """Character-level tokenizer: enough to test label masking without a model."""

    eos_token = "§"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}

    @staticmethod
    def decode(ids):
        return "".join(chr(i) for i in ids)
