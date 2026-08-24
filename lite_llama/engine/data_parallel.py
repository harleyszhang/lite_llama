"""Data parallelism: N whole-model replicas, one process each, requests routed between them.

Where tensor parallelism splits a *weight matrix* and pays an all-reduce per block,
data parallelism splits the *request stream* and pays nothing in the forward: a
replica holds the entire model, so it needs no collective at all. The only machinery
is therefore routing pick a replica per request, collect its answer  which is why
the replicas here are OS processes talking over ``multiprocessing`` queues rather than
NCCL ranks, and why each worker profiles and sizes its own KV cache against its card.

The structure mirrors how vLLM and SGLang lay this out, scaled down to lite_llama's
synchronous batch API:

* a **worker** (:func:`_dp_worker`) is a rank-aware process that builds one
  :class:`~lite_llama.engine.llm.LLM` on its own GPU and serves requests off a queue
  the role of vLLM's ``DPEngineCoreProc`` and SGLang's scheduler process;
* a **load balancer** (:mod:`lite_llama.engine.dp_load_balancer`) decides which replica
  each request goes to  SGLang's ``LoadBalanceMethod``, vLLM's ``DPLBAsyncMPClient``;
* the **coordinator** (:class:`DataParallelEngine`) owns the worker processes, routes
  through the balancer, and reassembles results in the caller's order  SGLang's
  ``DataParallelController``.

The cost model that follows: DP multiplies throughput (each replica decodes an
independent batch) while leaving per-token latency alone, and needs the weights
resident once per GPU. TP is the opposite trade  it splits the weights so a model too
large for one card fits, and pays latency for the collectives. They compose:
``dp_size`` replicas of ``tp_size`` ranks each, the grid
:func:`~lite_llama.distributed.parallel_state.init_parallel` describes.

Usage:
    with DataParallelEngine(model="my_weight/Qwen2.5-0.5B", data_parallel_size=2) as engine:
        outputs = engine.generate(prompts, SamplingParams(temperature=0.0))
"""

from __future__ import annotations

import contextlib
import queue
import time
import traceback
from typing import Any

import torch.multiprocessing as mp

from ..utils.logger import get_logger
from .dp_load_balancer import LOAD_BALANCERS, make_load_balancer
from .outputs import CompletionOutput, RequestOutput
from .sampler import SamplingParams

_log = get_logger(__name__)

#: Seconds a replica is given to import torch, load the checkpoint and profile its
#: KV cache. Loading a 30B FP8 checkpoint from a cold page cache is minutes, not
#: seconds, so this is deliberately generous — it exists to turn a wedged worker
#: into an error rather than a hang.
STARTUP_TIMEOUT_S = 900.0

#: How often a blocked wait looks up from the queue to check the workers are alive.
#: Generation has no timeout — a long prompt legitimately takes as long as it takes —
#: so liveness, not the clock, is what distinguishes "still working" from "died".
_LIVENESS_POLL_S = 5.0

#: Sentinel that tells a worker to leave its request loop and exit.
_SHUTDOWN = None


def _dp_worker(
    global_rank: int,
    dp_rank: int,
    tp_rank: int,
    dp_size: int,
    tp_size: int,
    engine_kwargs: dict[str, Any],
    request_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """One rank of the grid: build an engine on its own GPU, then serve until told to stop.

    There is one process per *cell* of the ``dp_size x tp_size`` grid, not one per
    replica. That is forced by ``init_parallel``: with ``tp_size > 1`` it rendezvouses a
    world of ``dp_size * tp_size`` ranks, so spawning only ``dp_size`` processes hangs
    forever waiting for ranks nobody started. Each cell occupies device ``global_rank``.

    Within a replica the ranks are *mirrors*: they receive the same request message and
    run the same forward, staying in step through the TP collectives (the sampled token
    is broadcast from TP rank 0). Only the replica's leader (``tp_rank == 0``) reports a
    result, because the coordinator expects exactly one answer per replica.

    Runs in a spawned process, so it takes only picklable arguments and imports torch
    itself. Every startup outcome is reported through ``result_queue`` — including a
    failed build, which arrives as an ``"error"`` message so the coordinator raises
    instead of blocking on a worker that will never answer.

    Args:
        global_rank: This process's rank in ``[0, dp_size * tp_size)``, and its device.
        dp_rank: Which replica this process belongs to.
        tp_rank: Which rank inside that replica; ``0`` is the reporting leader.
        dp_size: Total replicas (for the parallel-state grid).
        tp_size: TP ranks per replica (1 for pure DP).
        engine_kwargs: Forwarded verbatim to :class:`LLM`, minus ``device``.
        request_queue: ``(batch_id, indices, prompts, params)`` tuples, or
            :data:`_SHUTDOWN`.
        result_queue: Where ``("ready" | "done" | "error", ...)`` messages go back.
    """
    import torch

    from ..distributed.parallel_state import init_parallel
    from .llm import LLM

    is_leader = tp_rank == 0
    try:
        torch.cuda.set_device(global_rank)
        init_parallel(global_rank=global_rank, tp_size=tp_size, dp_size=dp_size)
        llm = LLM(device=f"cuda:{global_rank}", **engine_kwargs)
        result_queue.put(("ready", global_rank, None))
    except Exception:
        result_queue.put(("error", global_rank, traceback.format_exc()))
        return

    while True:
        message = request_queue.get()
        if message is _SHUTDOWN:
            break
        batch_id, indices, prompts, params = message
        try:
            outputs = llm.generate(prompts, params)
        except Exception:
            detail = traceback.format_exc()
            if is_leader:
                result_queue.put(("error", batch_id, detail))
                continue
            # A follower must not put a second message on the queue for one dispatch:
            # the coordinator counts messages, and the extra one would be misread as a
            # result by the next generate(). Exiting is the signal instead — the
            # coordinator's liveness poll turns the leader's stalled collective into a
            # RuntimeError naming this dead process.
            _log.error("tp rank %d of replica %d failed:\n%s", tp_rank, dp_rank, detail)
            return
        if not is_leader:
            continue
        result_queue.put(
            (
                "done",
                batch_id,
                [
                    (i, out.text, out.outputs[0].finish_reason)
                    for i, out in zip(indices, outputs, strict=True)
                ],
            )
        )


class DataParallelEngine:
    """Route generation across ``data_parallel_size`` model replicas.

    Mirrors :meth:`~lite_llama.engine.llm.LLM.generate` so a caller can swap one for
    the other; what differs is that nothing is loaded in *this* process — the model
    lives in the workers, and this object only routes. Deliberately not an ``LLM``
    subclass for that reason: it owns no model, no KV cache and no sampler, only the
    worker processes and a :class:`~lite_llama.engine.dp_load_balancer.LoadBalancer`.

    Replica ``i`` spans devices ``[i * tp, (i+1) * tp)`` and is served by that many
    processes — one per cell of the rank grid in
    :mod:`lite_llama.distributed.parallel_state`. The replica's leader is the process on
    its first device; it is the one that answers.

    Args:
        model: HuggingFace checkpoint directory, as for :class:`LLM`.
        data_parallel_size: Number of replicas; must not exceed the visible GPUs.
        tensor_parallel_size: TP ranks *within* each replica (1 = pure DP).
        load_balancer: Routing policy name, one of
            :data:`~lite_llama.engine.dp_load_balancer.LOAD_BALANCERS`.
        **engine_kwargs: Forwarded verbatim to each replica's :class:`LLM`
            (``max_seq_len``, ``quantization``, ``use_cuda_graph``, ...). ``device``
            is not accepted: it is derived from the replica's position in the grid.

    Raises:
        ValueError: If ``data_parallel_size`` is below 1, exceeds the visible GPU
            count, ``device`` was passed, or the policy name is unknown.
        RuntimeError: If a replica fails to build.
    """

    def __init__(
        self,
        model: str,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        load_balancer: str = "round_robin",
        **engine_kwargs: Any,
    ) -> None:
        import torch

        if data_parallel_size < 1:
            raise ValueError(f"data_parallel_size must be >= 1, got {data_parallel_size}")
        if "device" in engine_kwargs:
            raise ValueError(
                "device is derived from the replica's rank; pass data_parallel_size "
                "and tensor_parallel_size instead"
            )
        if load_balancer not in LOAD_BALANCERS:
            raise ValueError(
                f"unknown load_balancer {load_balancer!r}; choose from {LOAD_BALANCERS}"
            )
        needed = data_parallel_size * tensor_parallel_size
        visible = torch.cuda.device_count()
        if needed > visible:
            raise ValueError(
                f"data_parallel_size={data_parallel_size} x "
                f"tensor_parallel_size={tensor_parallel_size} needs {needed} GPUs, "
                f"but only {visible} are visible"
            )

        self.model = model
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.world_size = needed
        self._balancer = make_load_balancer(load_balancer, data_parallel_size)
        self._engine_kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            **engine_kwargs,
        }
        self._tokenizer = None
        self._next_batch_id = 0
        self._closed = False

        # "spawn" is required: a forked child inherits a CUDA context that cannot be
        # re-initialised on another device.
        ctx = mp.get_context("spawn")
        # One queue per *grid cell*, not per replica: the followers of a TP replica must
        # each receive the request message to run the same forward as their leader.
        self._request_queues = [ctx.Queue() for _ in range(needed)]
        self._result_queue: mp.Queue = ctx.Queue()
        self._workers = [
            ctx.Process(
                target=_dp_worker,
                args=(
                    global_rank,
                    global_rank // tensor_parallel_size,
                    global_rank % tensor_parallel_size,
                    data_parallel_size,
                    tensor_parallel_size,
                    self._engine_kwargs,
                    self._request_queues[global_rank],
                    self._result_queue,
                ),
                daemon=True,
            )
            for global_rank in range(needed)
        ]
        for worker in self._workers:
            worker.start()
        self._await_ready()

    def _await_message(self, timeout_s: float | None = None) -> tuple:
        """Take one message off the result queue, or raise if a replica died.

        A plain blocking ``get`` would wait forever on a worker killed by the OOM
        killer or a segfaulting kernel — there is no message coming, and the caller
        cannot tell that from a slow generation. Polling and checking ``is_alive``
        turns that silence into an error.

        Args:
            timeout_s: Overall deadline, or ``None`` to wait as long as the workers
                stay alive.

        Raises:
            RuntimeError: If a replica exited without reporting, or the deadline passed.
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            try:
                return self._result_queue.get(timeout=_LIVENESS_POLL_S)
            except queue.Empty:
                dead = [w for w in self._workers if not w.is_alive()]
                if dead:
                    raise RuntimeError(
                        "data-parallel replica(s) "
                        f"{[w.pid for w in dead]} exited without reporting a result "
                        f"(exit codes {[w.exitcode for w in dead]}); "
                        "the usual cause is the host or GPU running out of memory"
                    ) from None
                if deadline is not None and time.monotonic() > deadline:
                    raise RuntimeError(
                        f"data-parallel replicas did not respond within {timeout_s:.0f}s"
                    ) from None

    def _await_ready(self) -> None:
        """Block until every rank of the grid reports a built engine, or raise."""
        pending = set(range(self.world_size))
        while pending:
            kind, global_rank, detail = self._await_message(STARTUP_TIMEOUT_S)
            if kind == "error":
                self.shutdown()
                raise RuntimeError(f"data-parallel rank {global_rank} failed to start:\n{detail}")
            pending.discard(global_rank)
        _log.info(
            "data parallel ready: %d replicas x TP %d = %d ranks (%s)",
            self.data_parallel_size,
            self.tensor_parallel_size,
            self.world_size,
            type(self._balancer).__name__,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @property
    def tokenizer(self):
        """Tokenizer of the served checkpoint, loaded on first use.

        The replicas own the ones that matter; this is a convenience for callers that
        count tokens (benchmarks, evals) and would otherwise reach into a worker.
        """
        if self._tokenizer is None:
            from .llm_engine import LLMEngine

            self._tokenizer = LLMEngine._load_tokenizer(self.model)
        return self._tokenizer

    def _estimate_tokens(self, prompts: list[str]) -> list[int]:
        """Token count per prompt — the unit a token-aware balancer needs.

        Prompt *characters* are not prompt tokens: the ratio swings from ~1 for CJK to
        ~6 for whitespace-heavy code, so routing on ``len(prompt)`` silently mis-sizes
        every non-English batch. The tokenizer is the only honest answer, and it is
        called once for the whole batch.

        Returns zeros — and never touches the tokenizer — when the configured policy
        does not read the estimate, so ``round_robin`` stays free.
        """
        if not self._balancer.needs_token_estimate:
            return [0] * len(prompts)
        encoded = self.tokenizer(prompts, add_special_tokens=True)["input_ids"]
        return [len(ids) for ids in encoded]

    def _route(self, estimates: list[int]) -> list[list[int]]:
        """Ask the balancer which replica each prompt goes to.

        Returns one index list per replica: replica ``r`` should generate
        ``[prompts[i] for i in result[r]]``. Grouping the per-request decisions back
        into one sub-batch per replica keeps each replica doing a single efficient
        batched forward, while the *choice* stays a per-request policy — so swapping in
        a load-aware balancer changes the split without touching this code.

        Args:
            estimates: Prompt lengths in tokens, from :meth:`_estimate_tokens`.
        """
        buckets: list[list[int]] = [[] for _ in range(self.data_parallel_size)]
        for index, estimated_tokens in enumerate(estimates):
            replica = self._balancer.select(estimated_tokens=estimated_tokens)
            buckets[replica].append(index)
        return buckets

    def _replica_queues(self, replica: int) -> list[mp.Queue]:
        """Every rank of ``replica``, leader first — all of them must get the request."""
        base = replica * self.tensor_parallel_size
        return self._request_queues[base : base + self.tensor_parallel_size]

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Generate one completion per prompt, in the caller's order.

        The call blocks until every replica has finished its share, so the batch
        latency is the slowest replica's — with one prompt and four replicas, three of
        them sit idle. DP buys throughput on many requests, not latency on few.

        Args:
            prompts: A single prompt or a batch.
            sampling_params: Defaults to :class:`SamplingParams` defaults.

        Returns:
            One :class:`RequestOutput` per prompt, ordered as ``prompts`` was.

        Raises:
            RuntimeError: If the engine is closed, a replica raised while generating,
                or a replica died without answering.
        """
        if self._closed:
            raise RuntimeError("this DataParallelEngine has been shut down")

        params = sampling_params or SamplingParams()
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if not prompts:
            return []

        estimates = self._estimate_tokens(prompts)
        buckets = self._route(estimates)
        dispatched = 0
        for replica, indices in enumerate(buckets):
            if not indices:
                continue
            batch_id = self._next_batch_id
            self._next_batch_id += 1
            message = (batch_id, indices, [prompts[i] for i in indices], params)
            for request_queue in self._replica_queues(replica):
                request_queue.put(message)
            dispatched += 1

        texts: list[str | None] = [None] * len(prompts)
        reasons: list[str | None] = [None] * len(prompts)
        failure: RuntimeError | None = None
        for _ in range(dispatched):
            kind, batch_id, payload = self._await_message()
            if kind == "error":
                # Drain the rest of this round before raising: the sibling replicas
                # have already (or are about to) put their "done" messages into the
                # shared queue, and anything left there would be misread as a result
                # by the next generate() call.
                if failure is None:
                    failure = RuntimeError(f"data-parallel replica failed on request:\n{payload}")
                continue
            if failure is not None:
                continue  # a sibling's results; this call is failing anyway
            for index, text, reason in payload:
                texts[index] = text
                reasons[index] = reason

        # Every request on a replica is done, so let a load-aware balancer forget
        # them — subtracting the same estimate that was added. Runs on the error path
        # too: skipping it would leak in-flight load forever.
        for replica, indices in enumerate(buckets):
            for index in indices:
                self._balancer.release(replica, estimated_tokens=estimates[index])

        if failure is not None:
            raise failure

        return [
            RequestOutput(prompt=prompt, outputs=[CompletionOutput(0, text or "", reason)])
            for prompt, text, reason in zip(prompts, texts, reasons, strict=True)
        ]

    def shutdown(self) -> None:
        """Stop every replica and release its GPU memory. Idempotent.

        Workers are asked to leave their loop first and only killed if they do not: a
        replica in the middle of a forward would otherwise leave its CUDA context for
        the driver to clean up.
        """
        if self._closed:
            return
        self._closed = True
        for request_queue in self._request_queues:
            # The queue is already closed if its worker died with it.
            with contextlib.suppress(ValueError, OSError):
                request_queue.put(_SHUTDOWN)
        for worker in self._workers:
            worker.join(timeout=30)
            if worker.is_alive():
                _log.warning("data-parallel replica %s did not exit; terminating", worker.pid)
                worker.terminate()
                worker.join(timeout=5)

    def __enter__(self) -> DataParallelEngine:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.shutdown()

    def __del__(self) -> None:
        # Best effort: interpreter teardown may already have dropped what shutdown
        # needs, and a failure here would only mask the real exit path.
        with contextlib.suppress(Exception):
            self.shutdown()
