"""Chat prompt formatting: wrap a user turn with the checkpoint's chat template.

:func:`get_prompter` returns a :class:`ChatPrompter` when the tokenizer
ships a chat template and None otherwise, so callers choose between
chat formatting and raw prompts without template soup.

Usage:
    prompter = get_prompter(tokenizer)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from ..models.config import read_model_type


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


class PrompterResolver:
    """Decide whether a checkpoint's prompts go through a chat template.

    Sending a *base* model a templated prompt is harmful — base Qwen2.5 echoes
    ``<|im_start|>assistant`` back and degrades into repetition, and feeding a
    *chat* model a bare prompt is just as wrong — so the two checkpoint kinds
    must be told apart reliably. The template itself always comes from the
    tokenizer (the vLLM approach, zero per-family maintenance); this class
    only makes the base-vs-instruct call, by these rules in order:

    1. a directory name containing ``base`` (Qwen3-0.6B-Base) is a base
       checkpoint — the tokenizer ships a template anyway, which is exactly
       the trap this rule exists to catch;
    2. a name containing ``instruct`` / ``chat`` / ``-it`` is an instruct one
       (LLaMA, Qwen2.5);
    3. otherwise ``model_type`` decides: families whose non-Base releases are
       all chat models (``qwen3``, ``qwen3_vl``) default to instruct, the rest
       to base — the conservative direction, since a template a base model
       never saw is worse than a bare prompt.

    The per-family prompter name maps this class used to carry (``"qwen2"``,
    ``"llama3"``, ...) are gone: the tokenizer's own template made them
    vestigial, and their silent fallback sent unregistered instruct families
    (DeepSeek) to verbatim. New model families need no code here.

    Lives in this module — not the CLI — so every entrypoint (``chat``,
    ``batch``, ``serve``) resolves prompting through one class instead of the
    CLI applying one policy and the server another.
    """

    _INSTRUCT_NAME_HINTS: ClassVar[tuple[str, ...]] = ("instruct", "chat", "-it")
    _CHAT_BY_DEFAULT_TYPES: ClassVar[tuple[str, ...]] = ("qwen3", "qwen3_vl")

    @staticmethod
    def read_model_type(model_dir: str) -> str:
        """``model_type`` from the checkpoint's config.json, ``""`` when unreadable.

        Reading delegates to :func:`lite_llama.models.config.read_model_type`
        (the config SSOT); this side only tolerates a missing or broken config
        by degrading to name-based detection.
        """
        try:
            return read_model_type(model_dir)
        except (OSError, ValueError):
            return ""

    @classmethod
    def is_instruct(cls, model_dir: str) -> bool:
        """Whether the checkpoint is instruction-tuned and wants a template."""
        name = Path(model_dir).name.lower()
        if "base" in name:
            return False
        if any(hint in name for hint in cls._INSTRUCT_NAME_HINTS):
            return True
        return cls.read_model_type(model_dir) in cls._CHAT_BY_DEFAULT_TYPES

    @classmethod
    def build(
        cls,
        model_dir: str,
        tokenizer: Any,
        *,
        use_template: bool = True,
    ) -> ChatPrompter | None:
        """The prompter for this checkpoint, or ``None`` for verbatim prompts.

        ``None`` means "send the prompt as-is": base checkpoints (by the rules
        above), or instruct ones whose tokenizer ships no template
        (:func:`get_prompter` already answers ``None`` for those).
        ``use_template=False`` is the explicit override behind
        ``--no-chat-template``.
        """
        if not use_template or not cls.is_instruct(model_dir):
            return None
        return get_prompter(tokenizer)
