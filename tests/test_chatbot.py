"""
UTDChatbot unit tests — runs without torch, transformers, or peft installed.
All heavy ML dependencies are mocked at module level before importing chatbot.
"""
import sys
import types


# ---------------------------------------------------------------------------
# Mock all heavy ML dependencies before any src import
# ---------------------------------------------------------------------------
def _mock(name, attrs=None):
    parts = name.split(".")
    for i in range(len(parts)):
        full = ".".join(parts[: i + 1])
        if full not in sys.modules:
            m = types.ModuleType(full)
            sys.modules[full] = m
            if i > 0:
                setattr(sys.modules[".".join(parts[:i])], parts[i], m)
    if attrs:
        for k, v in attrs.items():
            setattr(sys.modules[name], k, v)
    return sys.modules[name]


# --- Fake tensor-like object that supports .to() chaining ---
class _FakeInputs(dict):
    """Dict subclass that supports the .to(device) chaining pattern."""
    def to(self, device): return self
    def __getitem__(self, k):
        class _T:
            shape = (1, 5)
        return _T()

class _FakeTensor:
    def __init__(self, data=None): self.data = data or [[1, 2, 3, 4]]
    def __getitem__(self, idx): return self
    def tolist(self): return [0]

class _FakeTokenizer:
    eos_token_id = 2
    def __call__(self, text, return_tensors=None, **kw):
        return _FakeInputs({"input_ids": _FakeTensor()})
    def decode(self, ids, skip_special_tokens=True):
        return "UTD is a top research university in Texas."
    def apply_chat_template(self, messages, tokenize=False, **kw):
        return "mock_prompt"
    @classmethod
    def from_pretrained(cls, path, **kw): return cls()

class _FakeModel:
    device = "cpu"
    @classmethod
    def from_pretrained(cls, path, **kw): return cls()
    def generate(self, **kw): return _FakeTensor([[1, 2, 3, 4, 5]])
    def to(self, device): return self
    def eval(self): return self

class _FakeStreamer:
    def __init__(self, tokenizer, **kw): pass
    def __iter__(self):
        yield "UTD"
        yield " is great"

_mock("torch", {
    "cuda": types.SimpleNamespace(is_available=lambda: False),
    "float16": "float16",
    "bfloat16": "bfloat16",
    "inference_mode": lambda: __import__("contextlib").nullcontext(),
})
_mock("transformers", {
    "AutoTokenizer": _FakeTokenizer,
    "AutoModelForCausalLM": _FakeModel,
    "BitsAndBytesConfig": type("BnB", (), {"__init__": lambda s, **kw: None}),
    "TextIteratorStreamer": _FakeStreamer,
    "GenerationConfig": type("GC", (), {"__init__": lambda s, **kw: None}),
})
_mock("peft", {
    "PeftModel": type("PM", (), {
        "from_pretrained": classmethod(lambda cls, m, p, **kw: _FakeModel())
    }),
})
_mock("bitsandbytes")

# ---------------------------------------------------------------------------
import pytest
from src.chatbot import UTDChatbot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chatbot_lazy_model_not_loaded():
    """Model must NOT be loaded at construction time."""
    bot = UTDChatbot(model_path="test-model")
    assert bot._model is None
    assert bot._tokenizer is None


def test_chatbot_has_system_prompt():
    bot = UTDChatbot()
    assert hasattr(bot, "SYSTEM_PROMPT")
    assert isinstance(bot.SYSTEM_PROMPT, str)
    assert len(bot.SYSTEM_PROMPT) > 20


def test_chatbot_model_path_stored():
    bot = UTDChatbot(model_path="my/custom/model")
    assert bot.model_path == "my/custom/model"


def test_chatbot_chat_loads_model():
    """First chat() call must load the model (lazy loading)."""
    bot = UTDChatbot(model_path="test-model")
    assert bot._model is None
    response = bot.chat("What is UTD's CS ranking?")
    assert bot._model is not None
    assert isinstance(response, str)


def test_chatbot_chat_returns_nonempty_string():
    bot = UTDChatbot()
    result = bot.chat("Tell me about admission requirements.")
    assert isinstance(result, str)
    assert len(result) > 0


def test_chatbot_multiple_calls_reuse_model():
    """Model should only be loaded once across multiple chat() calls."""
    bot = UTDChatbot()
    bot.chat("first question")
    model_ref = id(bot._model)
    bot.chat("second question")
    assert id(bot._model) == model_ref   # same object, not re-loaded


def test_chatbot_history_grows():
    """Each chat() call should append to conversation history."""
    bot = UTDChatbot()
    bot.chat("Hello")
    assert len(bot.history) >= 1
    bot.chat("Follow-up question")
    assert len(bot.history) >= 2
