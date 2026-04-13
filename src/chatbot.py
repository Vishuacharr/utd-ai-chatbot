"""
UTDChatbot — inference wrapper for the fine-tuned Llama 3.2 3B model.

Features:
  - 4-bit NF4 quantization for efficient inference
  - Streaming token generation via TextIteratorStreamer
  - Multi-turn conversation with 10-turn sliding window
  - Lazy model loading (model only loaded on first chat() call)
  - Supports local weights and HuggingFace Hub models

Usage:
    bot = UTDChatbot()
    print(bot.chat("What GPA do I need for CS honors?"))

    # Streaming
    for token in bot.stream_chat("List the MSBA core courses"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Thread
from typing import Iterator, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the UTD Academic Assistant, an expert on The University of Texas at Dallas. "
    "You help students with course requirements, degree plans, academic policies, and campus resources. "
    "Always be helpful, accurate, and concise. If you are unsure, say so."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str       # "user" | "assistant" | "system"
    content: str


@dataclass
class ChatConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    load_in_4bit: bool = True


# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------

class UTDChatbot:
    """
    Fine-tuned Llama 3.2 chatbot for UTD academic queries.

    Model is loaded lazily on the first call to chat() or stream_chat()
    so the class can be instantiated in any environment without requiring
    GPU/transformers to be immediately available.
    """

    SYSTEM_PROMPT: str = SYSTEM_PROMPT   # exposed as class attribute for testing

    def __init__(
        self,
        model_path: str = "./fine_tuned_model",
        cfg: Optional[ChatConfig] = None,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.cfg = cfg or ChatConfig()
        self.device = device
        self.history: List[Message] = []

        # Lazy-load: model and tokenizer are None until first use
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Single-turn or multi-turn chat (non-streaming).

        Args:
            user_message: The user's input text

        Returns:
            The assistant's response string
        """
        self._ensure_loaded()
        self.history.append(Message(role="user", content=user_message))
        prompt = self._build_prompt()

        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                repetition_penalty=self.cfg.repetition_penalty,
                do_sample=self.cfg.do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        self.history.append(Message(role="assistant", content=response))
        return response

    def stream_chat(self, user_message: str) -> Iterator[str]:
        """
        Streaming chat — yields tokens one by one for real-time UIs.

        Args:
            user_message: The user's input text

        Yields:
            Individual decoded tokens as they are generated
        """
        self._ensure_loaded()
        from transformers import TextIteratorStreamer

        self.history.append(Message(role="user", content=user_message))
        prompt = self._build_prompt()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            do_sample=self.cfg.do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        full_response = ""
        for token in streamer:
            full_response += token
            yield token

        self.history.append(Message(role="assistant", content=full_response.strip()))

    def reset(self) -> None:
        """Clear conversation history (model stays loaded)."""
        self.history.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model and tokenizer on first use (lazy initialization)."""
        if self._model is not None:
            return
        self._load()

    def _load(self) -> None:
        """Load the fine-tuned model with 4-bit NF4 quantization."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        print(f"[UTDChatbot] Loading model from: {self.model_path}")

        bnb = BitsAndBytesConfig(
            load_in_4bit=self.cfg.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if self.cfg.load_in_4bit else None

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()
        print("[UTDChatbot] Model loaded successfully")

    def _build_prompt(self) -> str:
        """Build the chat prompt using the model's chat template."""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        for msg in self.history[-10:]:      # sliding window: last 10 turns
            messages.append({"role": msg.role, "content": msg.content})
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
