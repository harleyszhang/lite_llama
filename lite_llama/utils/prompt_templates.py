"""Chat prompt formatting: wrap a user turn with the checkpoint's own chat template.

The vLLM approach, and the only one kept: an instruct checkpoint ships its chat
template inside the tokenizer, so formatting a turn is just
``tokenizer.apply_chat_template(messages, add_generation_prompt=True)`` — always the
model's official format, zero per-family maintenance, and identical to the
multimodal path. Base (non-instruct) checkpoints carry no template and are sent
verbatim by the caller, so they never reach this module.

Usage:
    prompter = get_prompter(tokenizer)      # None when the checkpoint has no template
    prompter.insert_prompt("hello")
    text = prompter.model_input             # official-format prompt string
"""

from __future__ import annotations

from typing import Any


class ChatPrompter:
    """Format one user turn with the tokenizer's own chat template.

    Args:
        tokenizer: A HuggingFace tokenizer exposing ``chat_template`` and
            ``apply_chat_template`` (every modern instruct checkpoint has one).
        system_prompt: Optional system message prepended to the conversation.
    """

    def __init__(self, tokenizer: Any, system_prompt: str | None = None) -> None:
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.model_input: str | None = None

    def apply(self, messages: list[dict[str, str]]) -> str:
        """Format a whole conversation and return the prompt string.

        The multi-turn entry point the OpenAI ``/v1/chat/completions`` endpoint
        needs; :meth:`insert_prompt` is the single-turn shorthand built on it, so
        both paths format through one call and cannot drift apart.

        Args:
            messages: ``{"role": ..., "content": ...}`` dicts in order. A system
                prompt configured on this prompter is prepended unless the caller
                already supplied one.

        Returns:
            The templated prompt, with the assistant turn opened.
        """
        if self.system_prompt and not (messages and messages[0]["role"] == "system"):
            messages = [{"role": "system", "content": self.system_prompt}, *messages]
        self.model_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.model_input

    def insert_prompt(self, prompt: str) -> str:
        """Format ``prompt`` into ``model_input`` (and return it) as a single-turn chat."""
        return self.apply([{"role": "user", "content": prompt}])


def has_chat_template(tokenizer: Any) -> bool:
    """Whether ``tokenizer`` can format chats itself (i.e. an instruct checkpoint)."""
    return getattr(tokenizer, "chat_template", None) is not None


def get_prompter(tokenizer: Any, system_prompt: str | None = None) -> ChatPrompter | None:
    """Return a :class:`ChatPrompter`, or ``None`` when the tokenizer has no template.

    ``None`` means "send the prompt verbatim" — the correct behaviour for base
    (non-instruct) checkpoints, which have no chat template to apply.
    """
    if not has_chat_template(tokenizer):
        return None
    return ChatPrompter(tokenizer, system_prompt)
