"""Output containers for the user-facing :class:`~lite_llama.engine.llm.LLM` API.

Mirrors ``vllm/outputs.py``: one request yields one :class:`RequestOutput` carrying
one or more :class:`CompletionOutput` instances (n-best is a future extension; today
``outputs`` always has one entry, ``index == 0``).

Usage:
    out = llm.generate(prompts, params)[0]
    text, reason = out.text, out.outputs[0].finish_reason
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from .sampler import PositionLogprobs


@dataclass
class CompletionOutput:
    """One generated completion for a request.

    Attributes:
        index: Position in the n-best list; always ``0`` until n>1 sampling exists.
        text: The generated text (detokenised, special tokens stripped).
        finish_reason: Why decoding stopped — ``"eos"``, ``"length"`` or
            ``"repeat"`` (the engine's repetition guard fired).
        logprobs: Per-token logprob records, parallel to the generated tokens;
            ``None`` unless the request set :attr:`SamplingParams.logprobs`.
    """

    index: int
    text: str
    finish_reason: str | None
    logprobs: list[PositionLogprobs] | None = None


@dataclass
class RequestOutput:
    """Everything produced for one prompt.

    Attributes:
        prompt: The prompt as passed in (pre-template).
        outputs: Generated completions; ``outputs[0]`` is the primary one.
        prompt_logprobs: Per-position records for the prompt, position 0 and
            prefix-cache hits being ``None``; ``None`` unless the request set
            :attr:`SamplingParams.prompt_logprobs`.
    """

    prompt: str
    outputs: list[CompletionOutput]
    prompt_logprobs: list[PositionLogprobs | None] | None = None

    @property
    def text(self) -> str:
        """Shortcut for ``outputs[0].text``."""
        return self.outputs[0].text
