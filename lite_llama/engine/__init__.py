"""Generation engine: two batching strategies over one shared executor.

:class:`LLM` / :class:`LLMEngine` run a *one-shot* batch — every prompt starts on
the same step and the batch keeps its width until the longest sequence ends. That
is the right shape for offline generation, where all the prompts are known up
front.

:class:`ContinuousBatchingEngine` decides the batch per step instead, so requests
join and leave mid-flight; :class:`AsyncLLMEngine` puts an asyncio face on it for
online serving.

:class:`DataParallelEngine` is orthogonal to both: it runs several whole-model
replicas in separate processes and routes requests between them, for throughput once
one GPU is saturated. :class:`AsyncDataParallelEngine` puts an asyncio face on that
for online serving, the way :class:`AsyncLLMEngine` does for a single replica.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .async_data_parallel import AsyncDataParallelEngine
    from .async_engine import AsyncLLMEngine, StreamedOutput
    from .continuous_engine import ContinuousBatchingEngine
    from .data_parallel import DataParallelEngine
    from .generator import TextGenerator, VisionGenerator
    from .llm import LLM
    from .llm_engine import LLMEngine
    from .outputs import CompletionOutput, RequestOutput
    from .sampler import BatchedSamplingParams, Sampler, SamplingParams, sample_top_p
    from .scheduler import Request, RequestStatus, Scheduler, SchedulerConfig


# The submodules pull the executor and the Triton kernels; values such as
# ``SamplingParams`` or ``SchedulerConfig`` do not need either. Resolve the
# facade lazily so importing one lightweight symbol stays CPU-only.
_EXPORTS: dict[str, tuple[str, str]] = {
    "LLM": (".llm", "LLM"),
    "AsyncDataParallelEngine": (".async_data_parallel", "AsyncDataParallelEngine"),
    "AsyncLLMEngine": (".async_engine", "AsyncLLMEngine"),
    "BatchedSamplingParams": (".sampler", "BatchedSamplingParams"),
    "CompletionOutput": (".outputs", "CompletionOutput"),
    "ContinuousBatchingEngine": (".continuous_engine", "ContinuousBatchingEngine"),
    "DataParallelEngine": (".data_parallel", "DataParallelEngine"),
    "LLMEngine": (".llm_engine", "LLMEngine"),
    "Request": (".scheduler", "Request"),
    "RequestOutput": (".outputs", "RequestOutput"),
    "RequestStatus": (".scheduler", "RequestStatus"),
    "Sampler": (".sampler", "Sampler"),
    "SamplingParams": (".sampler", "SamplingParams"),
    "Scheduler": (".scheduler", "Scheduler"),
    "SchedulerConfig": (".scheduler", "SchedulerConfig"),
    "StreamedOutput": (".async_engine", "StreamedOutput"),
    "TextGenerator": (".generator", "TextGenerator"),
    "VisionGenerator": (".generator", "VisionGenerator"),
    "sample_top_p": (".sampler", "sample_top_p"),
}


def __getattr__(name: str) -> Any:
    """Resolve the engine facade without importing unrelated GPU modules."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORTS.keys())

__all__ = [
    "LLM",
    "AsyncDataParallelEngine",
    "AsyncLLMEngine",
    "BatchedSamplingParams",
    "CompletionOutput",
    "ContinuousBatchingEngine",
    "DataParallelEngine",
    "LLMEngine",
    "Request",
    "RequestOutput",
    "RequestStatus",
    "Sampler",
    "SamplingParams",
    "Scheduler",
    "SchedulerConfig",
    "StreamedOutput",
    "TextGenerator",
    "VisionGenerator",
    "sample_top_p",
]
