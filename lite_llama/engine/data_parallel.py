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
    dp_rank: int,
    dp_size: int,
    tp_size: int,
    engine_kwargs: dict[str, Any],
    request_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """One replica: build an engine on its own GPU, then serve requests until told to stop.

    Runs in a spawned process, so it takes only picklable arguments and imports torch
    itself. It occupies device ``dp_rank * tp_size`` (the first rank of its slice of
    the grid). Every outcome is reported through ``result_queue`` — including a failed
    build, which arrives as an ``"error"`` message so the coordinator raises instead of
    blocking on a worker that will never answer.

    Args:
        dp_rank: Which replica this process is.
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

    try:
        device_index = dp_rank * tp_size
        torch.cuda.set_device(device_index)
        init_parallel(global_rank=device_index, tp_size=tp_size, dp_size=dp_size)
        llm = LLM(device=f"cuda:{device_index}", **engine_kwargs)
        result_queue.put(("ready", dp_rank, None))
    except Exception:
        result_queue.put(("error", dp_rank, traceback.format_exc()))
        return

    while True:
        message = request_queue.get()
        if message is _SHUTDOWN:
            break
        batch_id, indices, prompts, params = message
        try:
            outputs = llm.generate(prompts, params)
            payload = [
                (i, out.text, out.outputs[0].finish_reason)
                for i, out in zip(indices, outputs, strict=True)
            ]
            result_queue.put(("done", batch_id, payload))
        except Exception:
            result_queue.put(("error", batch_id, traceback.format_exc()))


class DataParallelEngine:
    """Route generation across ``data_parallel_size`` model replicas.

    Mirrors :meth:`~lite_llama.engine.llm.LLM.generate` so a caller can swap one for
    the other; what differs is that nothing is loaded in *this* process — the model
    lives in the workers, and this object only routes. Deliberately not an ``LLM``
    subclass for that reason: it owns no model, no KV cache and no sampler, only the
    worker processes and a :class:`~lite_llama.engine.dp_load_balancer.LoadBalancer`.

    Replica ``i`` takes device ``cuda:{i * tensor_parallel_size}``, matching the rank
    grid in :mod:`lite_llama.distributed.parallel_state`.

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
        self._request_queues = [ctx.Queue() for _ in range(data_parallel_size)]
        self._result_queue: mp.Queue = ctx.Queue()
        self._workers = [
            ctx.Process(
                target=_dp_worker,
                args=(
                    dp_rank,
                    data_parallel_size,
                    tensor_parallel_size,
                    self._engine_kwargs,
                    self._request_queues[dp_rank],
                    self._result_queue,
                ),
                daemon=True,
            )
            for dp_rank in range(data_parallel_size)
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
        """Block until every replica reports a built engine, or raise."""
        pending = set(range(self.data_parallel_size))
        while pending:
            kind, dp_rank, detail = self._await_message(STARTUP_TIMEOUT_S)
            if kind == "error":
                self.shutdown()
                raise RuntimeError(f"data-parallel replica {dp_rank} failed to start:\n{detail}")
            pending.discard(dp_rank)
        _log.info(
            "data parallel ready: %d replicas x TP %d (%s)",
            self.data_parallel_size,
            self.tensor_parallel_size,
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

    def _route(self, prompts: list[str]) -> list[list[int]]:
        """Ask the balancer which replica each prompt goes to.

        Returns one index list per replica: replica ``r`` should generate
        ``[prompts[i] for i in result[r]]``. Grouping the per-request decisions back
        into one sub-batch per replica keeps each replica doing a single efficient
        batched forward, while the *choice* stays a per-request policy — so swapping in
        the least-loaded balancer changes the split without touching this code.
        """
        buckets: list[list[int]] = [[] for _ in range(self.data_parallel_size)]
        for index, prompt in enumerate(prompts):
            replica = self._balancer.select(estimated_tokens=len(prompt))
            buckets[replica].append(index)
        return buckets

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

        buckets = self._route(prompts)
        dispatched = 0
        for replica, indices in enumerate(buckets):
            if not indices:
                continue
            batch_id = self._next_batch_id
            self._next_batch_id += 1
            self._request_queues[replica].put(
                (batch_id, indices, [prompts[i] for i in indices], params)
            )
            dispatched += 1

        texts: list[str | None] = [None] * len(prompts)
        reasons: list[str | None] = [None] * len(prompts)
        for _ in range(dispatched):
            kind, batch_id, payload = self._await_message()
            if kind == "error":
                raise RuntimeError(f"data-parallel replica failed on request:\n{payload}")
            for index, text, reason in payload:
                texts[index] = text
                reasons[index] = reason

        # Every request on a replica is done, so let a load-aware balancer forget them.
        for replica, indices in enumerate(buckets):
            for _ in indices:
                self._balancer.release(replica)

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
