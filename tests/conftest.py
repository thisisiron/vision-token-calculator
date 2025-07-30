import torch
import pytest


class _SimpleTokenizer:
    """A minimal tokenizer mock to support id conversions."""

    def __init__(self, token_to_id=None):
        if token_to_id is None:
            token_to_id = {"[IMAGE]": 42}
        self._token_to_id = token_to_id

    def convert_tokens_to_ids(self, token):
        return self._token_to_id.get(token, -1)

    def decode(self, token_id):
        for token, tid in self._token_to_id.items():
            if tid == token_id:
                return token
        return "<UNK>"


class QwenProcessorMock:
    """A lightweight processor mock emulating Qwen behavior used in tests."""

    def __init__(self):
        self.image_token = "[IMAGE]"
        self.image_token_id = None
        self.tokenizer = _SimpleTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "USER:"

    def __call__(
        self,
        text,
        images,
        videos=None,
        padding=True,
        return_tensors="pt",
    ):
        image_token_index = self.tokenizer.convert_tokens_to_ids(self.image_token)
        # Create an input id sequence containing two image tokens.
        input_ids = torch.tensor([[1, 2, image_token_index, 3, image_token_index, 4]])
        return {"input_ids": input_ids}


@pytest.fixture(autouse=True)
def mock_cli_env(monkeypatch):
    """Mock heavy external dependencies for fast and deterministic tests."""
    import vt_calculator.core.calculator as calc

    # Mock AutoProcessor.from_pretrained to avoid network/model loading.
    monkeypatch.setattr(
        calc.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: QwenProcessorMock(),
        raising=True,
    )

    # Mock process_vision_info to a simple passthrough.
    def _process_vision_info(messages):
        image = messages[0]["content"][0]["image"]
        return [image], None

    monkeypatch.setattr(calc, "process_vision_info", _process_vision_info, raising=True)

    # Ensure a predictable device environment in tests.
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("TRANSFORMERS_VERBOSITY", "error")

    yield
