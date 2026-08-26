"""Asyncio front end for the data-parallel coordinator.

:meth:`DataParallelEngine.generate <lite_llama.engine.data_parallel.DataParallelEngine.generate>`
is synchronous and batch-shaped: it dispatches a whole prompt list, blocks until
every replica answers, and returns complete texts. An HTTP server needs the
opposite on all three counts — requests arrive one at a time, a response must
start streaming long before the answer exists, and blocking the event loop
stalls every other connection. This class is therefore to
:class:`~lite_llama.engine.data_parallel.DataParallelEngine` what
:class:`~lite_llama.engine.async_engine.AsyncLLMEngine` is to a single
:class:`~lite_llama.engine.continuous_engine.ContinuousBatchingEngine`, and
exposes the same ``generate`` / ``generate_text`` surface so the OpenAI layer
serves from either without knowing which.

The coordinator process stays free of CUDA and of the replicas' engines — the
workers own those, as before. What is added is one *pump* thread: it drains the
result queue (an ``mp.Queue`` whose blocking ``get`` no event loop can await)
and schedules each message onto the stream of the loop that created it, which is
exactly the role the worker thread's publish half plays in the single-engine
front end. Coroutines never touch the queue; the pump never touches a loop
directly.

Usage:
    async with AsyncDataParallelEngine.from_pretrained(
        "my_weight/Qwen2.5-0.5B", data_parallel_size=2
    ) as engine:
        async for chunk in engine.generate("Hello", SamplingParams()):
            print(chunk.delta, end="")
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import threading
from collections.abc import AsyncIterator
from typing import Any

from ..utils.logger import get_logger
from .async_engine import StreamedOutput, _RequestStream
from .data_parallel import DataParallelEngine
from .sampler import SamplingParams
from .scheduler import DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS

logger = get_logger(__name__)

#: Message tag that ends the pump thread. A tag rather than the ``None``
#: sentinel (which stops the *replicas* on the request side) so the two queue
#: directions cannot be confused by a reader of either.
_PUMP_STOP = "pump-stop"

#: How long shutdown waits for the pump to notice the stop tag before giving up
#: on a clean join. The pump is mid-``get`` at worst, so this is generous.
_PUMP_JOIN_TIMEOUT_S = 30.0


class AsyncDataParallelEngine(DataParallelEngine):
    """Serves many concurrent coroutines from ``data_parallel_size`` replicas.

    Construction starts the replica processes exactly as the synchronous engine
    does (and raises the same errors); ``start()`` then adds the pump thread,
    and every ``generate()`` coroutine routes through the load balancer to one
    replica's queue, streaming that request's chunks back as the replica
    reports them. A consumer that abandons its stream aborts the request on the
    replica, so an HTTP connection that drops frees its KV slot instead of
    decoding to the length cap.

    Args:
        model: HuggingFace checkpoint directory, as for :class:`LLM`.
        data_parallel_size: Number of replicas; must not exceed the visible GPUs.
        tensor_parallel_size: TP ranks *within* each replica (1 = pure DP).
        load_balancer: Routing policy, one of
            :data:`~lite_llama.engine.dp_load_balancer.LOAD_BALANCERS`. The
            load-aware names finally have someone to serve here: with the batch
            API every prompt arrived at once, so "least in-flight" had nothing
            to measure.
        max_num_seqs: Requests each replica keeps in flight. Per replica: DP
            multiplies concurrency along with throughput.
        max_num_batched_tokens: Padded token budget for one prefill group.
        enable_prefix_cache: Give every replica a prefix cache. Pairs with
            ``load_balancer="cache_aware"``, which is what stops the caches from
            being ``data_parallel_size`` unrelated ones — and where the online path
            has the advantage over the batch API, since requests arrive over time
            and a prefix populated by one can still be hot for the next.
        **engine_kwargs: Forwarded verbatim to each replica's engine, as for
            :class:`~lite_llama.engine.data_parallel.DataParallelEngine`.
            ``device`` is not accepted; it is derived from the grid position.
    """

    def __init__(
        self,
        model: str,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        load_balancer: str = "round_robin",
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
        enable_prefix_cache: bool = False,
        **engine_kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            load_balancer=load_balancer,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_cache=enable_prefix_cache,
            **engine_kwargs,
        )
        self._streams: dict[str, _RequestStream] = {}
        self._request_ids = itertools.count()
        self._pump: threading.Thread | None = None
        # Guards ``_streams`` only: the pump reads it while coroutines register
        # and drop entries. Everything else the pump touches — the result queue,
        # the workers — it shares with the constructor, which has returned.
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> AsyncDataParallelEngine:
        """Load a checkpoint and wrap it for async serving.

        The same spelling :meth:`~lite_llama.engine.async_engine.AsyncLLMEngine.from_pretrained`
        uses, so an entrypoint that builds one engine kind can build the other
        without changing the shape of its call.
        """
        return cls(model=model, **kwargs)

    # ------------------------------------------------------------- lifecycle #
    def start(self) -> None:
        """Start the result pump. Idempotent, and safe to call from any loop."""
        if self._pump is not None:
            return
        self._pump = threading.Thread(
            target=self._pump_results, name="lite-llama-dp-pump", daemon=True
        )
        self._pump.start()

    async def shutdown(self) -> None:
        """Stop the pump, then the replicas. Idempotent.

        The pump stops *first*: once it has joined, no message it would read can
        matter, so the replicas may be told to stop without anybody listening.
        Both stages block (a replica drains its in-flight batch before exiting),
        so both run off the event loop.
        """
        if self._pump is not None:
            with contextlib.suppress(ValueError, OSError):
                self._result_queue.put((_PUMP_STOP, None, None))
            await asyncio.get_running_loop().run_in_executor(
                None, self._pump.join, _PUMP_JOIN_TIMEOUT_S
            )
            self._pump = None
        with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()
        for stream in streams:
            stream.push(None)
        await asyncio.get_running_loop().run_in_executor(None, DataParallelEngine.shutdown, self)

    async def __aenter__(self) -> AsyncDataParallelEngine:
        self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.shutdown()

    def __del__(self) -> None:
        # Not the async shutdown(): interpreter teardown cannot await a
        # coroutine, and the parent's synchronous shutdown is what actually
        # reaps the replica processes. The pump is a daemon thread and dies
        # with the process.
        with contextlib.suppress(Exception):
            DataParallelEngine.shutdown(self)

    # ------------------------------------------------------------ public API #
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[StreamedOutput]:
        """Stream one request's completion from whichever replica the balancer picks.

        The request is submitted on first iteration and aborted if the consumer
        stops early — an abandoned HTTP connection frees its replica's cache slot
        on the next step rather than decoding to its length cap.

        Args:
            prompt: Prompt text, already chat-templated if the model expects that.
            sampling_params: Per-request knobs.
            request_id: Caller-supplied id; generated when omitted.

        Yields:
            :class:`~lite_llama.engine.async_engine.StreamedOutput` chunks, the
            last one carrying a finish reason.

        Raises:
            RuntimeError: If the engine is shut down, or the replica failed or
                died while serving this request.
        """
        if self._closed:
            raise RuntimeError("this AsyncDataParallelEngine has been shut down")
        self.start()
        request_id = request_id or f"dp-{next(self._request_ids)}"
        stream = _RequestStream(request_id, asyncio.get_running_loop())
        with self._lock:
            self._streams[request_id] = stream

        token_ids = self._tokenize_for_routing([prompt])
        prompt_ids = None if token_ids is None else token_ids[0]
        estimate = 0 if prompt_ids is None else len(prompt_ids)
        replica = self._select(prompt_ids)
        self._request_queues[replica].put(("add", request_id, prompt, sampling_params))
        try:
            while True:
                chunk = await stream.get()
                if chunk is None:
                    return
                yield chunk
                if chunk.is_finished:
                    return
        finally:
            with self._lock:
                self._streams.pop(request_id, None)
            if not stream.finished and not self._closed:
                # A dead replica's queue is not ours to notice: the put is
                # best-effort the same way the parent's shutdown puts are.
                with contextlib.suppress(ValueError, OSError):
                    self._request_queues[replica].put(("abort", request_id))
            # Runs on every exit, error path included: a load-aware balancer
            # that only ever heard ``select`` would count this request forever.
            self._balancer.release(replica, estimated_tokens=estimate)

    async def generate_text(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> StreamedOutput:
        """Await a whole completion, discarding the intermediate chunks."""
        last: StreamedOutput | None = None
        async for chunk in self.generate(prompt, sampling_params, request_id):
            last = chunk
        if last is None:
            raise RuntimeError(f"request {request_id} produced no output")
        return last

    # -------------------------------------------------------------- the pump #
    def _pump_results(self) -> None:
        """Drain the result queue onto streams. The only reader of that queue.

        Waits as long as the replicas stay alive — generation has no deadline —
        and relies on :meth:`DataParallelEngine._await_message` to turn a dead
        replica into an error rather than a hang, which is why the pump does
        not reimplement that logic. On such a failure every stream still open
        hears the error: from a coroutine's side, a coordinator whose workers
        are gone is indistinguishable from one that never answers.
        """
        while True:
            try:
                message = self._await_message()
            except RuntimeError as exc:
                with self._lock:
                    streams = list(self._streams.values())
                    self._streams.clear()
                for stream in streams:
                    stream.finished = True
                    stream.push(RuntimeError(f"data-parallel engine failed: {exc}"))
                return
            if message[0] == _PUMP_STOP:
                return
            self._deliver(message)

    def _deliver(self, message: tuple) -> None:
        """Move one replica message onto its stream, if anyone still holds it."""
        kind, request_id = message[0], message[1]
        stream = self._streams.get(request_id)
        if stream is None:
            # The consumer went away; the abort command it queued on its way
            # out will reclaim the replica's slot, so there is nothing to do.
            return
        if kind == "delta":
            _, _, delta, text, prompt_tokens, completion_tokens = message
            stream.push(
                StreamedOutput(
                    request_id=request_id,
                    delta=delta,
                    text=text,
                    finish_reason=None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
        elif kind == "finished":
            _, _, reason, text, prompt_tokens, completion_tokens = message
            stream.finished = True
            stream.push(
                StreamedOutput(
                    request_id=request_id,
                    delta="",
                    text=text,
                    finish_reason=reason,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
        elif kind == "failed":
            stream.finished = True
            stream.push(RuntimeError(f"replica failed on request {request_id}:\n{message[2]}"))
