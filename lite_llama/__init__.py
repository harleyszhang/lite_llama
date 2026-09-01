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

One GPU too few for the request rate? :class:`DataParallelEngine` takes the same
arguments plus ``data_parallel_size`` and routes the prompts across that many whole-
model replicas, one process per GPU. For serving, ``AsyncDataParallelEngine`` adds
streaming on top — ``lite-llama serve --data-parallel-size N`` is it.
"""

from .engine import (
    LLM,
    AsyncDataParallelEngine,
    AsyncLLMEngine,
    CompletionOutput,
    ContinuousBatchingEngine,
    DataParallelEngine,
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

__version__ = "0.10.0"

__all__ = [
    "LLM",
    "AsyncDataParallelEngine",
    "AsyncLLMEngine",
    "CompletionOutput",
    "ContinuousBatchingEngine",
    "DataParallelEngine",
    "LLMEngine",
    "Request",
    "RequestOutput",
    "Sampler",
    "SamplingParams",
    "SchedulerConfig",
    "StreamedOutput",
    "TextGenerator",
    "VisionGenerator",
    "__version__",
    "sample_top_p",
]
