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

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine.async_data_parallel import AsyncDataParallelEngine
    from .engine.async_engine import AsyncLLMEngine, StreamedOutput
    from .engine.continuous_engine import ContinuousBatchingEngine
    from .engine.data_parallel import DataParallelEngine
    from .engine.generator import TextGenerator, VisionGenerator
    from .engine.llm import LLM
    from .engine.llm_engine import LLMEngine
    from .engine.outputs import CompletionOutput, RequestOutput
    from .engine.sampler import Sampler, SamplingParams, sample_top_p
    from .engine.scheduler import Request, SchedulerConfig


# Importing ``lite_llama`` is common in CLI discovery, test collection and worker
# process startup.  Keep that operation CPU-only: engine implementations import
# the executor and Triton kernels, while lightweight values such as
# ``SamplingParams`` do not need either.  PEP 562 gives the public facade the
# same API without eagerly importing every implementation behind it.
_EXPORTS: dict[str, tuple[str, str]] = {
    "LLM": (".engine.llm", "LLM"),
    "AsyncDataParallelEngine": (".engine.async_data_parallel", "AsyncDataParallelEngine"),
    "AsyncLLMEngine": (".engine.async_engine", "AsyncLLMEngine"),
    "CompletionOutput": (".engine.outputs", "CompletionOutput"),
    "ContinuousBatchingEngine": (".engine.continuous_engine", "ContinuousBatchingEngine"),
    "DataParallelEngine": (".engine.data_parallel", "DataParallelEngine"),
    "LLMEngine": (".engine.llm_engine", "LLMEngine"),
    "Request": (".engine.scheduler", "Request"),
    "RequestOutput": (".engine.outputs", "RequestOutput"),
    "Sampler": (".engine.sampler", "Sampler"),
    "SamplingParams": (".engine.sampler", "SamplingParams"),
    "SchedulerConfig": (".engine.scheduler", "SchedulerConfig"),
    "StreamedOutput": (".engine.async_engine", "StreamedOutput"),
    "TextGenerator": (".engine.generator", "TextGenerator"),
    "VisionGenerator": (".engine.generator", "VisionGenerator"),
    "sample_top_p": (".engine.sampler", "sample_top_p"),
}


def __getattr__(name: str) -> Any:
    """Load a public symbol on first use and cache it in this module."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORTS.keys())

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
