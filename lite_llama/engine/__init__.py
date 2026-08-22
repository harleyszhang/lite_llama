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
one GPU is saturated.
"""

from .async_engine import AsyncLLMEngine, StreamedOutput
from .continuous_engine import ContinuousBatchingEngine
from .data_parallel import DataParallelEngine
from .generator import TextGenerator, VisionGenerator
from .llm import LLM
from .llm_engine import LLMEngine
from .outputs import CompletionOutput, RequestOutput
from .sampler import BatchedSamplingParams, Sampler, SamplingParams, sample_top_p
from .scheduler import Request, RequestStatus, Scheduler, SchedulerConfig

__all__ = [
    "LLM",
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
