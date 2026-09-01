"""Continuous batching: a step-driven engine where requests join and leave freely.

Each ``step()`` asks the :class:`~lite_llama.engine.scheduler.Scheduler` for
a plan, runs it through the executor, harvests sampled tokens and updates
request state — so chunked prefills and running decodes share one pass.

Usage:
    engine.add_request(prompt, params)
    finished = engine.step()
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, NamedTuple

import torch

from ..distributed.parallel_state import get_tp_rank, get_tp_world_size
from ..executor.executor import (
    Executor,
    MultiprocExecutor,
    UniProcExecutor,
    launch_tensor_parallel,
)
from ..executor.worker import ModelInput, PassKind, PassLogprobs
from ..models.config import read_model_type
from ..models.registry import ModelRegistry
from ..observe import EngineMetrics, Tracer
from .detokenizer import IncrementalDetokenizer
from .outputs import CompletionOutput, RequestOutput
from .sampler import PositionLogprobs, SamplingParams
from .scheduler import (
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS,
    Request,
    Scheduler,
    SchedulerConfig,
)
from .stop_criteria import POLL_INTERVAL, detect_repetition

if TYPE_CHECKING:
    from .llm_engine import LLMEngine


class _Work(NamedTuple):
    """A plan plus the requests whose tokens it will produce, in that order.

    The plan names slots, not requests, so the step keeps request objects
    alongside it. ``requests`` is parallel to ``plan.sampled``;
    ``chunk_requests`` is parallel to ``plan.slots`` (chunk passes only).
    """

    plan: ModelInput
    requests: list[Request]
    chunk_requests: tuple[Request, ...] = ()


def _chunk_work(kind: PassKind, chunks: list[tuple[Request, int]]) -> _Work:
    """Plan one prompt-chunk pass; the two routes differ only in ``kind``.

    Chunk ``i`` writes cache rows ``[num_computed_tokens - chunk,
    num_computed_tokens)`` of its slot — the scheduler already advanced the
    counter. A chunk resuming on a prefix-cache hit also carries the copies.
    """
    slots, starts, lens, tokens = [], [], [], []
    sampled, requests = [], []
    prompt_logprobs, prompt_targets = [], []
    copies: list[tuple[int, int, int, int]] = []
    for row, (request, chunk) in enumerate(chunks):
        start = request.num_computed_tokens - chunk
        end = request.num_computed_tokens
        slots.append(request.slot)
        starts.append(start)
        lens.append(end)
        tokens.extend(request.prompt_token_ids[start:end])
        prompt_logprobs.append(request.params.prompt_logprobs)
        # Row j is scored against the token at start+j+1. A partial chunk's
        # last row targets the *next* chunk's first token; a final chunk's
        # last row is sampled and has no target. Both tails pad with 0.
        targets = request.prompt_token_ids[start + 1 : end + 1]
        prompt_targets.extend(targets + [0] * (chunk - len(targets)))
        copies += [
            (src_slot, request.slot, start_token, length)
            for src_slot, start_token, length in request.prefix_copies
        ]
        if request.num_computed_tokens == request.prompt_len:
            # Only a finished prompt has a next token to sample; the pass
            # mixes both, so sampled rows are a subset named by row index.
            sampled.append(row)
            requests.append(request)

    wants_prompt = any(k is not None for k in prompt_logprobs)
    return _Work(
        ModelInput(
            kind=kind,
            slots=tuple(slots),
            seq_starts=tuple(starts),
            seq_lens=tuple(lens),
            tokens=tuple(tokens),
            sampling=tuple(request.params for request in requests),
            sampled=tuple(sampled),
            # A first token has no repetition-penalty history yet.
            gen_counts=(0,) * len(requests),
            prefix_copies=tuple(copies),
            prompt_logprobs=tuple(prompt_logprobs) if wants_prompt else (),
            prompt_targets=tuple(prompt_targets) if wants_prompt else (),
        ),
        requests,
        tuple(request for request, _ in chunks),
    )


def _prefill_work(group: list[Request], chunk_lens: list[int]) -> list[_Work]:
    """Split the step's prompt chunks by the kernel each may legally use.

    A *first* chunk (``num_computed_tokens == chunk``) runs as a padded grid
    through the prefill kernel — nothing of the prompt is cached yet. A
    *resumed* chunk cannot: the prefill kernel never reads the cache, so its
    tokens would silently drop the prefix. Those extend instead, one row per
    token, each attending over its slot's whole cached history.
    """
    pairs = list(zip(group, chunk_lens, strict=True))
    routes = (
        (PassKind.PREFILL, [pair for pair in pairs if pair[0].num_computed_tokens == pair[1]]),
        (PassKind.EXTEND, [pair for pair in pairs if pair[0].num_computed_tokens > pair[1]]),
    )
    return [_chunk_work(kind, chunks) for kind, chunks in routes if chunks]


def _decode_work(running: list[Request]) -> _Work:
    """Plan one decode token for every fully prefilled request.

    The input token is the last one each request generated — already back on
    the host from the previous step's synchronisation.
    """
    # The request already counts the token it is about to feed.
    seq_lens = tuple(request.seq_len for request in running)
    return _Work(
        ModelInput(
            kind=PassKind.DECODE,
            slots=tuple(request.slot for request in running),
            # One token per row, landing at the row its cache length points at.
            seq_starts=tuple(length - 1 for length in seq_lens),
            seq_lens=seq_lens,
            tokens=tuple(request.output_token_ids[-1] for request in running),
            sampling=tuple(request.params for request in running),
            sampled=tuple(range(len(running))),
            gen_counts=tuple(len(request.output_token_ids) for request in running),
        ),
        running,
    )


class ContinuousBatchingEngine:
    """Serves independently arriving requests as one continuously reshaped batch.

    Drive it by calling :meth:`step` in a loop: each call runs the scheduler's
    plan (prompt chunks, then a decode token for everything running) and returns
    the requests that produced a token. One host-device synchronisation per
    step is deliberate — it reads sampled tokens back to detokenise and to
    decide who stops, so a finished request leaves the batch on the next step.

    Args:
        engine: A built :class:`~lite_llama.engine.llm_engine.LLMEngine`; takes
            over its KV cache and must be the only user of it.
        config: Admission limits. Defaults derive ``max_seq_len`` from the engine.
        executor: Where passes run. Defaults to a
            :class:`~lite_llama.executor.executor.UniProcExecutor` (single GPU);
            injecting a fake is how a test drives the step loop without a model.

    Raises:
        NotImplementedError: The checkpoint is multimodal — vision prefill
            needs per-request processor outputs the batched grid has no place for.
    """

    def __init__(
        self,
        engine: LLMEngine,
        config: SchedulerConfig | None = None,
        executor: Executor | None = None,
    ) -> None:
        if engine.model_runner.spec.is_multimodal:
            raise NotImplementedError(
                "continuous batching supports text-only checkpoints; "
                "use LLM.generate() for vision-language models"
            )

        self.engine = engine
        self.device = engine.device
        self.tokenizer = engine.tokenizer
        self.stop_token_ids = engine.stop_token_ids

        config = config or SchedulerConfig(max_seq_len=engine.max_seq_len)
        if config.max_seq_len > engine.max_seq_len:
            raise ValueError(
                f"scheduler max_seq_len {config.max_seq_len} exceeds the engine's "
                f"{engine.max_seq_len}"
            )
        self.config = config

        self._executor: Executor = executor or UniProcExecutor(
            engine, config.max_num_seqs, config.max_seq_len
        )
        # The executor owns the cache, so it decides how many requests can be in
        # flight; the scheduler hands out exactly those slots.
        self.scheduler = Scheduler(config, self._executor.num_slots)

        self._detokenizers: dict[str, IncrementalDetokenizer] = {}
        self._request_ids = itertools.count()
        self._step_count = 0

        # Observability: metrics and tracing are cheap no-ops when disabled,
        # so the hot loop never branches on them.
        self.metrics = EngineMetrics.from_env()
        self.tracer = Tracer.from_env()
        self._spans: dict[str, object] = {}

    # ------------------------------------------------------------------ build #
    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        tokenizer: str | None = None,
        max_seq_len: int = 2048,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool = True,
        quantization: str | None = None,
        tensor_parallel_size: int = 1,
        kv_cache_dtype: str = "auto",
        enable_prefix_cache: bool = False,
    ) -> ContinuousBatchingEngine:
        """Load a checkpoint and wrap it in a continuous-batching engine.

        Args:
            model: HuggingFace checkpoint directory.
            tokenizer: Tokenizer location; defaults to ``model``.
            max_seq_len: Context window, and the per-slot cache size.
            max_num_seqs: Concurrency ceiling.
            max_num_batched_tokens: Padded token budget for one prefill group.
            max_gpu_num_blocks: Manual KV-cache size in tokens; profiled when ``None``.
            device: Torch device string.
            use_cuda_graph: Capture decode graphs. Continuous batching pads
                odd batch sizes onto the captured grid, so most steps stay on
                the graph path. Ignored above ``tensor_parallel_size`` 1, where
                captured collectives would be unsafe.
            quantization: Runtime weight quantisation, forwarded to the engine —
                orthogonal to batching.
            tensor_parallel_size: GPUs this replica's weights are split over.
                Above 1, ranks 1.. spawn as followers and every step's plan is
                broadcast to them; this process stays rank 0. If this process
                already sits in a TP group (the CLI, a DP controller), that group
                is reused and the value only has to agree with it.
            kv_cache_dtype: KV-cache element type, forwarded to the engine
                (``"auto"`` for fp16, or an fp8 spelling to halve the cache).
            enable_prefix_cache: Reuse the K/V of prompt prefixes already
                resident in the cache. Off by default: it only pays when prompts
                share a prefix, and otherwise costs a hash per block. See
                :mod:`lite_llama.engine.prefix_cache`.

        Raises:
            NotImplementedError: The checkpoint is multimodal.
            ValueError: ``tensor_parallel_size`` contradicts a group this process
                is already a member of.
        """
        # Keep CPU-only planning and fake-executor tests importable without
        # Triton. A real model is the only path that needs the GPU engine.
        from .llm_engine import LLMEngine

        spec = ModelRegistry.resolve(read_model_type(model))
        if spec.is_multimodal:
            raise NotImplementedError(
                "continuous batching supports text-only checkpoints; "
                "use LLM.generate() for vision-language models"
            )

        engine_kwargs = {
            "checkpoints_dir": model,
            "tokenizer_path": tokenizer,
            "max_seq_len": max_seq_len,
            "max_gpu_num_blocks": max_gpu_num_blocks,
            # A captured graph would replay a sharded layer's collectives, so
            # tensor parallelism decodes eager.
            "use_cuda_graph": use_cuda_graph and tensor_parallel_size == 1,
            "quantization": quantization,
            "kv_cache_dtype": kv_cache_dtype,
        }

        # Followers must exist before this rank builds its engine: sharded
        # layers read their width from the process group.
        followers: tuple[BaseProcess, ...] = ()
        joined = get_tp_world_size()
        if joined > 1 and joined != tensor_parallel_size:
            raise ValueError(
                f"this process is already rank {get_tp_rank()} of a {joined}-way "
                f"tensor-parallel group, but tensor_parallel_size={tensor_parallel_size}"
            )
        if joined == 1 and tensor_parallel_size > 1:
            followers = launch_tensor_parallel(tensor_parallel_size, engine_kwargs, max_num_seqs)

        engine = LLMEngine(
            device=device,
            tensor_parallel_size=tensor_parallel_size,
            **engine_kwargs,
        )
        config = SchedulerConfig(
            max_seq_len=engine.max_seq_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_cache=enable_prefix_cache,
        )
        executor: Executor | None = None
        if get_tp_world_size() > 1:
            executor = MultiprocExecutor(engine, config.max_num_seqs, config.max_seq_len, followers)
        return cls(engine, config, executor)

    # ------------------------------------------------------------- public API #
    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> Request:
        """Queue a request and return the handle that tracks it.

        The handle is updated in place by :meth:`step`, so a caller can read
        ``delta``, ``text`` and ``finish_reason`` as generation proceeds.

        Args:
            prompt: Prompt text, already chat-templated if the model wants that.
            sampling_params: Per-request knobs; defaults to :class:`SamplingParams`.
            request_id: Caller-supplied id; generated when omitted.
            prompt_token_ids: Pre-tokenised prompt, to skip re-encoding.
        """
        if request_id is None:
            request_id = f"req-{next(self._request_ids)}"
            while request_id in self._detokenizers:
                request_id = f"req-{next(self._request_ids)}"
        request = Request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=(
                prompt_token_ids
                if prompt_token_ids is not None
                else self.tokenizer.encode(prompt, add_special_tokens=True)
            ),
            params=sampling_params or SamplingParams(),
        )
        self.scheduler.add_request(request)
        self._detokenizers[request.request_id] = IncrementalDetokenizer(self.tokenizer, 1)
        self._spans[request.request_id] = self.tracer.start_span(
            "request", request_id=request.request_id, prompt_tokens=request.prompt_len
        )
        return request

    def abort(self, request_id: str) -> Request | None:
        """Cancel a request; its slot is free for the next step."""
        request = self.scheduler.abort(request_id)
        if request is not None:
            self.metrics.finished.inc(finish_reason="abort")
            self.tracer.end_span(self._spans.pop(request.request_id, None), finish_reason="abort")
            self._retire(request)
        return request

    def has_unfinished_requests(self) -> bool:
        """Whether anything is queued or in flight."""
        return self.scheduler.has_unfinished_requests()

    @torch.inference_mode()
    def step(self) -> list[Request]:
        """Run one engine step and return the requests it advanced.

        Returns:
            The requests that produced a token this step, in pass order. A
            request that stopped on a stop token is included with an empty
            ``delta`` — the async front end learns a request ended only from
            this list, so leaving it out would strand its stream.
        """
        scheduled = self.scheduler.schedule()
        if scheduled.is_empty:
            return []

        self._step_count += 1
        # Freshly admitted requests owe a queue-time observation (num_computed
        # equals their first chunk); resumed chunks and preemption re-admissions
        # are skipped or re-counted by the same test.
        for request, chunk in zip(scheduled.prefill, scheduled.prefill_chunk_lens, strict=True):
            if request.num_computed_tokens == chunk:
                self.metrics.observe_queue_time(request)
        work: list[_Work] = []
        if scheduled.prefill:
            work += _prefill_work(scheduled.prefill, scheduled.prefill_chunk_lens)
        if scheduled.decode:
            work.append(_decode_work(scheduled.decode))

        # Execute every pass before reading any tokens back: a step's passes
        # are slot-disjoint, so one synchronisation per step suffices, and a
        # later pass's input prep rides the copy stream while an earlier pass's
        # forward is still on the GPU (the L1 overlap site).
        pending: list[tuple[_Work, torch.Tensor, PassLogprobs | None]] = []
        for work_item in work:
            tokens, logprobs = self._executor.execute(work_item.plan)
            pending.append((work_item, tokens, logprobs))

        emitted: list[tuple[Request, int, PositionLogprobs | None]] = []
        for work_item, tokens, logprobs in pending:
            # ``prompt`` is uniformly None for a decode pass.
            if logprobs is not None and any(logprobs.prompt):
                self._attribute_prompt_logprobs(work_item, logprobs.prompt)
            records = (
                logprobs.sampled
                if logprobs is not None and logprobs.sampled
                else (None,) * len(work_item.requests)
            )
            emitted += [
                (request, token, record)
                for (request, token), record in zip(
                    zip(work_item.requests, tokens.tolist(), strict=True), records, strict=True
                )
            ]
        # Counter properties, not len(running): those copy the lists.
        advanced = self._harvest(emitted)
        self.metrics.observe_load(self.scheduler.num_running, self.scheduler.num_waiting)
        return advanced

    def generate(
        self,
        prompts: Sequence[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Run a whole prompt set through the scheduler and return the completions.

        Offline convenience wrapper: submits every prompt at once and drives
        :meth:`step` to exhaustion. A set exceeding ``max_num_seqs`` is admitted
        in waves, and short answers free their slots early.

        Returns:
            One :class:`~lite_llama.engine.outputs.RequestOutput` per prompt, in
            submission order.
        """
        params = sampling_params or SamplingParams()
        requests = [self.add_request(prompt, params) for prompt in prompts]
        while self.has_unfinished_requests():
            self.step()
        return [
            RequestOutput(
                prompt=request.prompt,
                outputs=[
                    CompletionOutput(
                        0, request.text, request.finish_reason, logprobs=request.output_logprobs
                    )
                ],
                prompt_logprobs=request.prompt_logprobs,
            )
            for request in requests
        ]

    def shutdown(self) -> None:
        """Release the executor. The engine cannot serve any more steps after this."""
        self._executor.shutdown()

    def timeline_summary(self) -> str:
        """Stream region table of the steps run so far; empty unless tracing is on."""
        return self._executor.timeline_summary()

    # ---------------------------------------------------------------- harvest #
    def _attribute_prompt_logprobs(
        self, work: _Work, prompt: tuple[tuple[PositionLogprobs, ...] | None, ...]
    ) -> None:
        """Place a chunk pass's prompt records on their requests, by position.

        Entry ``j`` of sequence ``i`` covers position ``seq_starts[i] + j + 1``.
        Position 0 and prefix-cache hits stay ``None``; the list is allocated
        at full prompt length on first contact, so chunks may land in any order.
        """
        for request, start, records in zip(
            work.chunk_requests, work.plan.seq_starts, prompt, strict=True
        ):
            if records is None:
                continue
            if request.prompt_logprobs is None:
                request.prompt_logprobs = [None] * request.prompt_len
            for j, record in enumerate(records):
                request.prompt_logprobs[start + j + 1] = record

    def _harvest(
        self, emitted: list[tuple[Request, int, PositionLogprobs | None]]
    ) -> list[Request]:
        """Read the step's tokens back, detokenise them, and retire whoever stopped.

        The only host-device synchronisation in the loop. It makes stop handling
        exact — a stop token retires the request on the next step, and the freed
        slot goes straight to a queued request.
        """
        now = time.monotonic()
        check_repeat = self._step_count % POLL_INTERVAL == 0
        advanced: list[Request] = []

        for request, token_id, record in emitted:
            if request.first_token_time is None:
                request.first_token_time = now
            request.delta = ""
            request.delta_logprobs = None

            if token_id in self.stop_token_ids:
                # The stop token is model punctuation, not output; the request
                # still belongs in this step's return — its stream has to hear
                # the finish reason (see step()).
                self._finish(request, "eos")
                advanced.append(request)
                continue

            request.output_token_ids.append(token_id)
            if record is not None:
                # Parallel to output_token_ids: one record per accepted token.
                if request.output_logprobs is None:
                    request.output_logprobs = []
                request.output_logprobs.append(record)
                request.delta_logprobs = record
            request.delta = self._detokenizers[request.request_id].append(0, token_id)
            request.text += request.delta
            advanced.append(request)

            if not request.has_room or request.seq_len >= self.config.max_seq_len:
                self._finish(request, "length")
            elif check_repeat and request.params.stop_on_repeat and detect_repetition(request.text):
                self._finish(request, "repeat")

        return advanced

    def _finish(self, request: Request, reason: str) -> None:
        self.scheduler.finish(request, reason)
        self.metrics.observe_finish(request)
        self.tracer.end_span(
            self._spans.pop(request.request_id, None),
            finish_reason=reason,
            output_tokens=len(request.output_token_ids),
        )
        self._retire(request)

    def _retire(self, request: Request) -> None:
        """Drop the per-request state the engine owns; the caller keeps the handle."""
        self._detokenizers.pop(request.request_id, None)
