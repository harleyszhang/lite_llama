"""Public API: :class:`LLM` (vLLM-style entry), :class:`LLMEngine`, sampling.

Typical usage::

    from lite_llama import LLM, SamplingParams

    llm = LLM(model="my_weight/Qwen2.5-0.5B")
    outputs = llm.generate(
        ["What is the capital of France?"], SamplingParams(temperature=0.0)
    )
    print(outputs[0].outputs[0].text)

``TextGenerator`` / ``VisionGenerator`` remain as backward-compatible wrappers.

For online serving, where requests arrive independently and should share a batch,
use the continuous-batching engine instead::

    from lite_llama import ContinuousBatchingEngine

    engine = ContinuousBatchingEngine.from_pretrained("my_weight/Qwen2.5-0.5B")
    print(engine.generate(["Hello", "Bonjour"], SamplingParams())[0].text)

``AsyncLLMEngine`` is the asyncio wrapper behind ``lite-llama serve``.
"""

from .engine import (
    LLM,
    AsyncLLMEngine,
    CompletionOutput,
    ContinuousBatchingEngine,
    LLMEngine,
    Request,
    RequestOutput,
    Sampler,
    SamplingParams,
    SchedulerConfig,
    StreamedOutput,
    TextGenerator,
    VisionGenerator,
    sample_top_p,
)

__version__ = "0.3.0"

__all__ = [
    "LLM",
    "LLMEngine",
    "AsyncLLMEngine",
    "ContinuousBatchingEngine",
    "Request",
    "SchedulerConfig",
    "StreamedOutput",
    "Sampler",
    "SamplingParams",
    "TextGenerator",
    "VisionGenerator",
    "CompletionOutput",
    "RequestOutput",
    "sample_top_p",
    "__version__",
]
