"""Output containers for the user-facing :class:`~lite_llama.engine.llm.LLM` API.

Mirrors the shape of ``vllm/outputs.py``: a request yields one
:class:`RequestOutput` carrying one or more :class:`CompletionOutput`
instances (n-best sampling is a future extension; today ``outputs`` always
has one entry, ``index == 0``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompletionOutput:
    """One generated completion for a request.

    Attributes:
        index: Position in the n-best list; always ``0`` until n>1 sampling exists.
        text: The generated text (detokenised, special tokens stripped).
        finish_reason: Why decoding stopped — ``"eos"``, ``"length"`` or
            ``"repeat"`` (the engine's repetition guard fired).
    """

    index: int
    text: str
    finish_reason: str | None


@dataclass
class RequestOutput:
    """Everything produced for one prompt.

    Attributes:
        prompt: The prompt as passed in (pre-template).
        outputs: Generated completions; ``outputs[0]`` is the primary one.
    """

    prompt: str
    outputs: list[CompletionOutput]

    @property
    def text(self) -> str:
        """Shortcut for ``outputs[0].text``."""
        return self.outputs[0].text
