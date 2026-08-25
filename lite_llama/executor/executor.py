"""Executor: the one interface an engine has for running a model pass.

Core design
-----------
An engine should not know whether the model it drives lives in this process, in
eight of them, or behind a network. It hands an :class:`~lite_llama.executor.worker.ModelInput`
to an :class:`Executor` and gets sampled tokens back — three methods, no tensors in
the signature except the result.

Two implementations, differing only in *where the plan comes from*:

* :class:`UniProcExecutor` calls the local :class:`~lite_llama.executor.worker.ModelWorker`
  directly. Single process, so a breakpoint in the engine loop is a breakpoint in
  the kernel; this is the default for one GPU and the reason the plan-building
  code stays debuggable.
* ``MultiprocExecutor`` (tensor parallelism) broadcasts the plan over a CPU
  collective and then runs *the same* worker method. Because the plan is pure
  data and every rank derives layout from it identically, the driver and the
  workers execute one code path, not two.

Usage:
    executor = UniProcExecutor(llm_engine, max_num_seqs=32, max_seq_len=2048)
    tokens = executor.execute(plan)
    executor.shutdown()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from .worker import ModelInput, ModelWorker

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..engine.llm_engine import LLMEngine


class Executor(ABC):
    """Runs model passes on behalf of an engine.

    Implementations own the model, the KV cache and however many processes it
    takes to hold them. The engine's side of the contract is narrow on purpose:
    it may ask how many cache slots exist, submit a plan, and shut the executor
    down.
    """

    @property
    @abstractmethod
    def num_slots(self) -> int:
        """How many cache slots plans may address, i.e. the concurrency ceiling."""

    @abstractmethod
    def execute(self, model_input: ModelInput) -> torch.Tensor:
        """Run one pass and return its sampled token ids, one per sampled row."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release whatever the executor owns beyond this object's lifetime."""


class UniProcExecutor(Executor):
    """One process, one model, no message passing.

    Args:
        engine: A built :class:`~lite_llama.engine.llm_engine.LLMEngine`; the
            executor takes its KV cache over.
        max_num_seqs: Concurrency ceiling.
        max_seq_len: Context bound.
    """

    def __init__(self, engine: LLMEngine, max_num_seqs: int, max_seq_len: int) -> None:
        self._worker = ModelWorker(engine, max_num_seqs, max_seq_len)

    @property
    def num_slots(self) -> int:
        return self._worker.num_slots

    def execute(self, model_input: ModelInput) -> torch.Tensor:
        return self._worker.execute(model_input)

    def shutdown(self) -> None:
        """Nothing to tear down: the caller still owns the engine it passed in."""
