"""Public API: :class:`LLM` (vLLM-style entry), :class:`LLMEngine`, sampling.

Typical usage::

    from lite_llama import LLM, SamplingParams

    llm = LLM(model="my_weight/Qwen2.5-0.5B")
    outputs = llm.generate(
        ["What is the capital of France?"], SamplingParams(temperature=0.0)
    )
    print(outputs[0].outputs[0].text)

``TextGenerator`` / ``VisionGenerator`` remain as backward-compatible wrappers.
"""

from .engine import (
    LLM,
    CompletionOutput,
    LLMEngine,
    RequestOutput,
    Sampler,
    SamplingParams,
    TextGenerator,
    VisionGenerator,
    sample_top_p,
)

__version__ = "0.2.0"

__all__ = [
    "LLM",
    "LLMEngine",
    "Sampler",
    "SamplingParams",
    "TextGenerator",
    "VisionGenerator",
    "CompletionOutput",
    "RequestOutput",
    "sample_top_p",
    "__version__",
]
