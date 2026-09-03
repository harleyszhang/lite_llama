"""Executor: the one interface an engine has for running a model pass.

:class:`Executor` is the abstract seam — ``execute`` a :class:`ModelInput`
in, sampled tokens out; :class:`UniProcExecutor` runs the worker in-process
while :class:`MultiprocExecutor` forwards plans to TP follower ranks.

Usage:
    executor = UniProcExecutor(engine, max_num_seqs, max_seq_len)
    tokens, logprobs = executor.execute(model_input)
"""

from __future__ import annotations

import multiprocessing as mp
import socket
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from ..distributed.parallel_state import tensor_model_parallel_broadcast_object_list
from ..utils.logger import get_logger
from .worker import ModelInput, ModelWorker, PassLogprobs

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..engine.llm_engine import LLMEngine

_log = get_logger(__name__)

#: How long :meth:`MultiprocExecutor.shutdown` waits for a follower to notice the
#: stop signal before it is killed. A follower's remaining work is one forward
#: pass, so this is generous.
SHUTDOWN_TIMEOUT_S = 30.0

#: How long a teardown waits for ``destroy_process_group`` before giving up on
#: it. Eager groups destroy in milliseconds; the wait exists solely for the
#: graph-captured case, where NCCL's abort can sit in a futex forever.
DESTROY_DEADLINE_S = 15.0


def _destroy_with_deadline(destroy: Callable[[], None], abandon: Callable[[], None]) -> None:
    """Tear the group down, but never park the caller on a wedged abort.

    ``destroy_process_group`` runs on a daemon thread with a deadline: eager
    engines (the overwhelmingly common teardown) destroy promptly and notice
    nothing; a communicator whose collectives were captured into a CUDA graph
    can block the abort indefinitely — a PyTorch/NCCL interaction that no
    amount of syncing or re-ordering clears from the outside. On the deadline
    the grid is abandoned instead: the parallel-state globals reset to a world
    of one so this process stops speaking for the group, and the wedged
    communicator dies with the process and its CUDA context.
    """
    import threading

    teardown = threading.Thread(target=destroy, daemon=True)
    teardown.start()
    teardown.join(DESTROY_DEADLINE_S)
    if teardown.is_alive():
        _log.warning(
            "group teardown did not finish within %.0fs (a graph-captured "
            "NCCL communicator can wedge its abort); abandoning the group — "
            "it dies with this process",
            DESTROY_DEADLINE_S,
        )
        abandon()


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

    @property
    def num_kv_blocks(self) -> int:
        """Cache blocks the scheduler may hand out, or ``0`` if the executor cannot say.

        Zero leaves the scheduler to size its block pool from the slot geometry it
        already knows, which is what a fake executor in a test wants; a real one
        reports what its profiled cache actually holds.
        """
        return 0

    @abstractmethod
    def execute(self, model_input: ModelInput) -> tuple[torch.Tensor, PassLogprobs | None]:
        """Run one pass: its sampled token ids, one per sampled row, and any
        logprob records the plan asked for (``None`` when none did)."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release whatever the executor owns beyond this object's lifetime."""

    def readback_async(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """Stage sampled tokens for the host without waiting for their pass.

        The default is the blocking degradation — ``.cpu()`` and no event —
        which keeps the engine's launch/harvest contract for executors with
        no copy stream to ride (fakes in tests, CPU-only workers). Real
        executors forward to their worker's pool and return a pinned view
        whose copy lands behind the pass that produced it.
        """
        return tokens.cpu(), None

    def timeline_summary(self) -> str:
        """Region table of the streams this executor ran on, for overlap diagnostics.

        Empty unless stream tracing is enabled (``LITE_LLAMA_OVERLAP_TIMELINE``), so
        callers can print it unconditionally; an executor that owns no streams keeps
        this default.
        """
        return ""


class UniProcExecutor(Executor):
    """One process, one model, no message passing.

    Args:
        engine: A built :class:`~lite_llama.engine.llm_engine.LLMEngine`; the
            executor takes its KV cache over.
        max_num_seqs: Concurrency ceiling.
        max_seq_len: Context bound.
        pipeline: Whether the worker feeds decode inputs back on the device
            (the O2 launch/harvest engine); ``None`` defers to
            :data:`~lite_llama.executor.worker.PIPELINE_ENV`.
    """

    def __init__(
        self,
        engine: LLMEngine,
        max_num_seqs: int,
        max_seq_len: int,
        *,
        pipeline: bool | None = None,
    ) -> None:
        self._worker = ModelWorker(engine, max_num_seqs, max_seq_len, pipeline=pipeline)

    @property
    def num_slots(self) -> int:
        return self._worker.num_slots

    @property
    def num_kv_blocks(self) -> int:
        return self._worker.num_kv_blocks

    def execute(self, model_input: ModelInput) -> tuple[torch.Tensor, PassLogprobs | None]:
        return self._worker.execute(model_input)

    def readback_async(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        return self._worker.readback(tokens)

    def timeline_summary(self) -> str:
        return self._worker.timeline.summary()

    def shutdown(self) -> None:
        """Nothing to tear down: the caller still owns the engine it passed in."""


class MultiprocExecutor(Executor):
    """Tensor parallelism: this rank plans, every rank runs.

    The driver is rank 0 of its replica *and* a worker, so a TP size of two costs
    two processes rather than three. Each :meth:`execute` publishes the plan on
    the CPU group and then does its own share of the forward; the collectives
    inside the model and the sampler line the ranks up from there.

    Args:
        engine: This rank's :class:`~lite_llama.engine.llm_engine.LLMEngine`,
            holding its shard of the weights.
        max_num_seqs: Concurrency ceiling.
        max_seq_len: Context bound.
        followers: Processes running ranks 1.. of this replica, as returned by
            :func:`launch_tensor_parallel`. Empty when someone else owns them
            (the CLI, a DP controller), in which case shutdown only sends the
            stop signal.
        pipeline: Whether the worker feeds decode inputs back on the device
            (the O2 launch/harvest engine); ``None`` defers to
            :data:`~lite_llama.executor.worker.PIPELINE_ENV`, which is also
            how the follower ranks learn the driver's choice.
    """

    def __init__(
        self,
        engine: LLMEngine,
        max_num_seqs: int,
        max_seq_len: int,
        followers: Sequence[mp.process.BaseProcess] = (),
        *,
        pipeline: bool | None = None,
    ) -> None:
        self._worker = ModelWorker(engine, max_num_seqs, max_seq_len, pipeline=pipeline)
        self._followers = tuple(followers)
        self._live = True

    @property
    def num_slots(self) -> int:
        return self._worker.num_slots

    @property
    def num_kv_blocks(self) -> int:
        return self._worker.num_kv_blocks

    def execute(self, model_input: ModelInput) -> tuple[torch.Tensor, PassLogprobs | None]:
        ensure_followers_alive(self._followers)
        tensor_model_parallel_broadcast_object_list(model_input)
        return self._worker.execute(model_input)

    def readback_async(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        # Rank 0's copy is the only one that matters: followers discard their
        # tokens exactly as they discard their sampled results.
        return self._worker.readback(tokens)

    def timeline_summary(self) -> str:
        """Only this rank's regions; the followers trace their own streams."""
        return self._worker.timeline.summary()

    def shutdown(self) -> None:
        """Tell the followers to leave their loop, then reap them.

        Idempotent, and conditional on the group still being whole: broadcasting
        the stop signal to a rank that has already died would block forever, so a
        crashed follower is joined without being asked. An empty follower tuple
        means somebody else owns the processes, and the signal is still theirs to
        receive.

        Owning the followers means owning the rank-0 half of their group too, so
        a non-empty tuple tears the group down with them: left standing, it
        re-shards the next engine this process builds, and even changes how an
        unrelated transformers load behaves.
        """
        if not self._live:
            return
        self._live = False
        if all(process.is_alive() for process in self._followers):
            tensor_model_parallel_broadcast_object_list(None)
        if self._followers:
            # Our half of the group goes down BEFORE the reap, not after: the
            # followers' own teardown only completes when every rank destroys
            # with them, so a follower whose rank 0 is parked in ``join`` is
            # stuck inside its destructor waiting for a communicator we are
            # holding. The deadlock is invisible to an eager run — a plain
            # communicator tears down without the rendezvous — but one whose
            # collectives were recorded into a CUDA graph has unfulfilled
            # device-side state that only the group-wide destroy can release.
            # Killing that follower then breaks OUR destroy too, which is the
            # hang this ordering exists to prevent. The barrier lines every
            # rank up at the destroy itself, because ``ncclCommAbort`` is
            # collective in some NCCL versions: one rank aborting alone parks
            # the peer's abort forever.
            from ..distributed.parallel_state import (
                abandon_parallel,
                destroy_parallel,
                tensor_model_parallel_barrier,
            )

            tensor_model_parallel_barrier()
            _destroy_with_deadline(destroy_parallel, abandon_parallel)
        for process in self._followers:
            process.join(timeout=SHUTDOWN_TIMEOUT_S)
            if process.is_alive():
                _log.warning("tp follower pid %s did not exit; terminating", process.pid)
                process.terminate()


def ensure_followers_alive(followers: Sequence[mp.process.BaseProcess]) -> None:
    """Raise if any follower has exited, before a collective can hang on it.

    Every collective assumes all ranks arrive. When one has died the rest simply
    wait, and a silent hang is the worst failure mode multi-process execution
    has — so the driver checks the cheap local fact (is the process alive) before
    committing to the expensive global one. Rank numbering starts at 1: rank 0 is
    the process doing the checking.
    """
    for rank, process in enumerate(followers, start=1):
        if not process.is_alive():
            raise RuntimeError(
                f"tensor-parallel rank {rank} (pid {process.pid}) exited with code "
                f"{process.exitcode}; see its traceback above"
            )


def free_port() -> int:
    """A port the OS says is free, so a rendezvous never inherits a stale one.

    A fixed default (29500) makes two engines on one machine collide, and makes a
    crashed run's lingering socket break the next one — both of which surface as a
    hang at rendezvous rather than as an error.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_tensor_parallel(
    tp_size: int,
    engine_kwargs: dict[str, Any],
    max_num_seqs: int,
    master_port: int | None = None,
) -> tuple[mp.process.BaseProcess, ...]:
    """Start ranks 1..``tp_size``-1 and join this process as rank 0.

    Blocks until the whole group has rendezvoused, so on return the caller may
    build its own :class:`~lite_llama.engine.llm_engine.LLMEngine` and the shard
    widths will already be right — layers read the TP size from
    :mod:`lite_llama.distributed.parallel_state`, not from an argument.

    Args:
        tp_size: Ranks in the group, including this one.
        engine_kwargs: Constructor arguments every rank builds its engine from,
            minus ``device``, which is the rank's own GPU. Must be picklable.
        max_num_seqs: Concurrency ceiling, so followers size their scratch to
            match.
        master_port: Rendezvous port; rank 0 listens. Defaults to a free one.

    Returns:
        The follower processes, in rank order.
    """
    from ..distributed.parallel_state import init_tensor_parallel

    master_port = free_port() if master_port is None else master_port
    context = mp.get_context("spawn")
    followers = [
        context.Process(
            target=run_follower,
            args=(rank, tp_size, engine_kwargs, max_num_seqs, master_port),
            name=f"lite-llama-tp{rank}",
            daemon=True,
        )
        for rank in range(1, tp_size)
    ]
    for process in followers:
        process.start()
    init_tensor_parallel(rank=0, world_size=tp_size, master_port=master_port)
    return tuple(followers)


def serve_plans(engine: LLMEngine, max_num_seqs: int) -> None:
    """Run broadcast plans until the driver sends ``None``. The whole of a follower.

    A follower rank holds no scheduler, no queue and no stop criteria, and it
    discards the tokens it samples — rank 0 sampled the same ones and is the one
    who has to detokenise them. What keeps the ranks in step is that they run
    identical code over an identical plan.

    Separate from :func:`run_follower` because who *starts* a follower varies —
    this module spawns it for a lone replica, the data-parallel controller spawns
    it as one cell of its grid — while what a follower *does* must not.

    Args:
        engine: This rank's engine, holding its shard of the weights.
        max_num_seqs: Concurrency ceiling, so the scratch matches the driver's.
            ``max_seq_len`` is taken from the engine instead: it only sizes local
            scratch, so reading it here cannot desynchronise anything.
    """
    worker = ModelWorker(engine, max_num_seqs, engine.max_seq_len)
    while (plan := tensor_model_parallel_broadcast_object_list()) is not None:
        # The records are discarded exactly as the tokens are: every rank
        # computed identical ones, and rank 0 is the one that reports them.
        worker.execute(plan)


def run_follower(
    rank: int,
    tp_size: int,
    engine_kwargs: dict[str, Any],
    max_num_seqs: int,
    master_port: int,
) -> None:
    """Body of a non-driver tensor-parallel rank: rendezvous, build, serve plans.

    Module-level so that ``spawn`` can pickle it by name.
    """
    from ..distributed.parallel_state import destroy_parallel, init_tensor_parallel
    from ..engine.llm_engine import LLMEngine

    torch.cuda.set_device(rank)
    init_tensor_parallel(rank=rank, world_size=tp_size, master_port=master_port)
    try:
        engine = LLMEngine(device=f"cuda:{rank}", tensor_parallel_size=tp_size, **engine_kwargs)
        _log.info("tp rank %d ready on cuda:%d", rank, rank)
        serve_plans(engine, max_num_seqs)
    finally:
        # Meet rank 0 at the destroy before running it: ncclCommAbort is
        # collective in some NCCL versions, so the ranks have to abort
        # together — a lone abort leaves the peer parked forever. The deadline
        # is the belt to that braces: a communicator whose collectives were
        # captured into a CUDA graph can park the abort itself in a futex
        # (a PyTorch/NCCL interaction), and a follower that cannot leave its
        # teardown never exits to be reaped.
        from ..distributed.parallel_state import (
            abandon_parallel,
            tensor_model_parallel_barrier,
        )

        tensor_model_parallel_barrier()
        _destroy_with_deadline(destroy_parallel, abandon_parallel)
