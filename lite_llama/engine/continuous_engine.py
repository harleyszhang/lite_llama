"""Continuous batching: a step-driven engine where requests join and leave mid-flight.

:class:`~lite_llama.engine.llm_engine.LLMEngine` fixes its batch when
``generate()`` is called. Every sequence starts on the same step and the batch
keeps running at full width until the *longest* one finishes, so a batch of eight
where seven stop at 20 tokens and one runs to 500 spends most of its time
computing 480 steps of padding. It also cannot accept a request that arrives one
millisecond after the call started.

:class:`ContinuousBatchingEngine` replaces the fixed batch with a per-step
decision, mirroring vLLM v1's split of the engine loop into "schedule → execute →
harvest". The middle stage is deliberately thin: a step turns the scheduler's
plan into :class:`~lite_llama.executor.worker.ModelInput` values — pure data, no
tensors — and hands each to an :class:`~lite_llama.executor.executor.Executor`,
which returns the tokens it sampled. A step produces up to three of them:

* a **prefill** grid for chunks whose prompt is not in the cache yet;
* an **extend** pass for chunks resuming on top of a cached prefix;
* a **decode** pass for every fully prefilled request.

All three can occur in the same step — chunked prefill interleaves with decode
instead of stalling it. Planning rather than executing is what keeps this file
free of device state: nothing here has to be invalidated when a request joins or
leaves, because a plan names the slots it means. It is also what lets tensor
parallelism drop in behind the same call, a plan being small enough to broadcast
and complete enough for every rank to derive the identical layout from it.

One host-device synchronisation per step is deliberate: sampled tokens are read
back to detokenise and to decide who stops, which is what retires a finished
request on the very next step.

Usage:
    engine = ContinuousBatchingEngine.from_pretrained("my_weight/Qwen2.5-0.5B")
    engine.add_request("The capital of France is", SamplingParams(temperature=0.0))
    while engine.has_unfinished_requests():
        for request in engine.step():
            print(request.delta, end="")
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
from ..executor.worker import ModelInput, PassKind
from ..models.config import read_model_type
from ..models.registry import ModelRegistry
from .detokenizer import IncrementalDetokenizer
from .outputs import CompletionOutput, RequestOutput
from .sampler import SamplingParams
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
    """A plan, and the requests whose tokens it will produce, in that order.

    The plan is anonymous by design — it names slots, not requests — so the step
    keeps the request objects alongside it to attribute the sampled tokens.
    """

    plan: ModelInput
    requests: list[Request]


def _chunk_work(kind: PassKind, chunks: list[tuple[Request, int]]) -> _Work:
    """Plan one prompt-chunk pass; the two routes differ only in ``kind``.

    Chunk ``i`` writes cache rows ``[num_computed_tokens - chunk,
    num_computed_tokens)`` of its slot — the scheduler has already advanced the
    counter, so this reads the chunk's span off the request rather than tracking
    it separately. A chunk that resumes on a prefix-cache hit also carries the
    copies that put that prefix in its slot.
    """
    slots, starts, lens, tokens = [], [], [], []
    sampled, requests = [], []
    copies: list[tuple[int, int, int, int]] = []
    for row, (request, chunk) in enumerate(chunks):
        start = request.num_computed_tokens - chunk
        slots.append(request.slot)
        starts.append(start)
        lens.append(request.num_computed_tokens)
        tokens.extend(request.prompt_token_ids[start : request.num_computed_tokens])
        # The scheduler names the source slot and offset; the destination is this
        # request's own slot, which only the plan knows about.
        copies += [
            (src_slot, request.slot, start_token, length)
            for src_slot, start_token, length in request.prefix_copies
        ]
        if request.num_computed_tokens == request.prompt_len:
            # Only a chunk that finished its prompt has a next token to sample,
            # and a pass mixes the two: the admission budget happily takes a
            # short prompt (done in one chunk) beside a long one (chunk-capped).
            # So the sampled rows are a subset, named by row index.
            sampled.append(row)
            requests.append(request)

    return _Work(
        ModelInput(
            kind=kind,
            slots=tuple(slots),
            seq_starts=tuple(starts),
            seq_lens=tuple(lens),
            tokens=tuple(tokens),
            sampling=tuple(request.params for request in requests),
            sampled=tuple(sampled),
            # A first token has no generated history yet, so the repetition
            # penalty is a no-op here whatever the request configured.
            gen_counts=(0,) * len(requests),
            prefix_copies=tuple(copies),
        ),
        requests,
    )


def _prefill_work(group: list[Request], chunk_lens: list[int]) -> list[_Work]:
    """Split the step's prompt chunks by the kernel each may legally use.

    A chunk routes by whether its slot already holds K/V from an earlier chunk:

    * a *first* chunk (``num_computed_tokens == chunk``) runs as a padded grid
      through the prefill kernel — pure self-attention over the grid, the cheap
      path, correct because nothing of the prompt is cached yet;
    * a *resumed* chunk cannot take it: the prefill kernel never reads the cache,
      so its tokens would attend only within the chunk and silently drop the
      prefix. Those tokens extend instead, one row per token, each attending over
      its slot's whole cached history.
    """
    pairs = list(zip(group, chunk_lens, strict=True))
    routes = (
        (PassKind.PREFILL, [pair for pair in pairs if pair[0].num_computed_tokens == pair[1]]),
        (PassKind.EXTEND, [pair for pair in pairs if pair[0].num_computed_tokens > pair[1]]),
    )
    return [_chunk_work(kind, chunks) for kind, chunks in routes if chunks]


def _decode_work(running: list[Request]) -> _Work:
    """Plan one decode token for every fully prefilled request.

    The input token is the last one each request generated, taken from the host:
    every step ends by synchronising to detokenise that token, so it is already
    sitting in ``output_token_ids``, and shipping it costs one upload of a few
    hundred bytes instead of a gather out of the generated-token grid.
    """
    # Cache length once this step's token is written: the request already counts
    # the token it is about to feed in ``output_token_ids``.
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

    Drive it by calling :meth:`step` in a loop. Each call runs the scheduler's
    plan for the step — prompt chunks for newly admitted or resumed requests, then
    a decode token for everything running — and returns the requests that
    produced a token, so a caller can stream text without knowing anything about
    batching.

    One synchronisation per step is unavoidable and deliberate: the sampled
    tokens are read back to detokenise them and to decide who stopped. That is
    what buys exact stop handling, and it is also what pays for itself, because a
    request that stops is out of the batch on the next step rather than padding
    it to the length cap.

    Args:
        engine: A built :class:`~lite_llama.engine.llm_engine.LLMEngine`; this
            object takes over its KV cache and must be the only user of it.
        config: Admission limits. Defaults derive ``max_seq_len`` from the engine.
        executor: Where passes run. Defaults to a
            :class:`~lite_llama.executor.executor.UniProcExecutor` over ``engine``,
            which is the single-GPU case; :meth:`from_pretrained` substitutes a
            tensor-parallel one. Injecting it is also how a test drives the step
            loop without a model.

    Raises:
        NotImplementedError: The checkpoint is multimodal. Vision prefill needs
            per-request processor outputs, which the batched prefill grid here
            has no place for.
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
            use_cuda_graph: Capture decode graphs. Worth keeping on: continuous
                batching pads odd batch sizes onto the captured grid, so most
                steps stay on the graph path. Ignored above
                ``tensor_parallel_size`` 1, where a sharded layer's collectives
                would be captured inside the graph.
            quantization: Runtime weight quantisation, forwarded to the engine.
                Orthogonal to batching -- it changes the linear layers, not the
                KV cache or the schedule.
            tensor_parallel_size: GPUs this replica's weights are split over.
                Above 1, ranks 1.. are spawned as follower processes and every
                step's plan is broadcast to them; this process stays rank 0 and
                keeps the scheduler. When the caller has *already* placed this
                process in a TP group (the CLI, a DP controller), no process is
                spawned and the existing group is used -- the value then only has
                to agree with it.
            kv_cache_dtype: KV-cache element type, forwarded to the engine
                (``"auto"`` for fp16, or an fp8 spelling to halve the cache).
            enable_prefix_cache: Reuse the K/V of prompt prefixes already resident
                in the cache instead of re-prefilling them. Off by default because
                it only pays when prompts share a prefix -- a system prompt, a
                few-shot preamble, a chat history -- and otherwise costs a hash per
                block. See :mod:`lite_llama.engine.prefix_cache`.

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
            # A captured graph would replay the collectives a sharded layer
            # issues, which is not safe, so tensor parallelism decodes eager --
            # the same trade-off :class:`~lite_llama.engine.llm.LLM` makes.
            "use_cuda_graph": use_cuda_graph and tensor_parallel_size == 1,
            "quantization": quantization,
            "kv_cache_dtype": kv_cache_dtype,
        }

        # Followers must exist *before* this rank builds its engine: sharded
        # layers read their width from the process group, and sizing the KV cache
        # is itself a collective over it.
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

        The returned :class:`~lite_llama.engine.scheduler.Request` is updated in
        place by :meth:`step`, so a caller can hold on to it and read ``delta``,
        ``text`` and ``finish_reason`` as generation proceeds.

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
        return request

    def abort(self, request_id: str) -> Request | None:
        """Cancel a request; its slot is free for the next step."""
        request = self.scheduler.abort(request_id)
        if request is not None:
            self._retire(request)
        return request

    def has_unfinished_requests(self) -> bool:
        """Whether anything is queued or in flight."""
        return self.scheduler.has_unfinished_requests()

    @torch.inference_mode()
    def step(self) -> list[Request]:
        """Run one engine step and return the requests it advanced.

        The shape is fixed — schedule, plan, execute, harvest — and the middle two
        stages are the only ones that know about batching at all.

        Returns:
            The requests that produced a token this step, in pass order. Each
            carries this step's text in ``delta``; those that stopped also carry
            a ``finish_reason``. A request that stopped on a stop token is
            included with an empty ``delta`` — the async front end learns a
            request ended only from what this list hands back, so leaving it
            out would strand its stream waiting on a final chunk that never
            comes.
        """
        scheduled = self.scheduler.schedule()
        if scheduled.is_empty:
            return []

        self._step_count += 1
        work: list[_Work] = []
        if scheduled.prefill:
            work += _prefill_work(scheduled.prefill, scheduled.prefill_chunk_lens)
        if scheduled.decode:
            work.append(_decode_work(scheduled.decode))

        emitted: list[tuple[Request, int]] = []
        for plan, requests in work:
            tokens = self._executor.execute(plan)
            emitted += zip(requests, tokens.tolist(), strict=True)
        return self._harvest(emitted)

    def generate(
        self,
        prompts: Sequence[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Run a whole prompt set through the scheduler and return the completions.

        Offline convenience wrapper: it submits every prompt at once and drives
        :meth:`step` to exhaustion. The prompts still flow through the scheduler,
        so a set that exceeds ``max_num_seqs`` is admitted in waves rather than
        rejected, and short answers free their slots early.

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
                outputs=[CompletionOutput(0, request.text, request.finish_reason)],
            )
            for request in requests
        ]

    def shutdown(self) -> None:
        """Release the executor. The engine cannot serve any more steps after this."""
        self._executor.shutdown()

    # ---------------------------------------------------------------- harvest #
    def _harvest(self, emitted: list[tuple[Request, int]]) -> list[Request]:
        """Read the step's tokens back, detokenise them, and retire whoever stopped.

        The only host-device synchronisation in the loop. It is what makes stop
        handling exact — a sequence that emits a stop token is out of the batch
        on the next step, not at the next poll interval — and exactness is worth
        more here than it is in the one-shot path, because the freed slot goes
        straight to a queued request.
        """
        now = time.monotonic()
        check_repeat = self._step_count % POLL_INTERVAL == 0
        advanced: list[Request] = []

        for request, token_id in emitted:
            if request.first_token_time is None:
                request.first_token_time = now
            request.delta = ""

            if token_id in self.stop_token_ids:
                # The stop token itself is model punctuation, not output; the
                # one-shot path drops it too. The request still belongs in
                # this step's return (see step()): its stream has to hear the
                # finish reason, or an async consumer waits on a final chunk
                # that never comes.
                self._finish(request, "eos")
                advanced.append(request)
                continue

            request.output_token_ids.append(token_id)
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
        self._retire(request)

    def _retire(self, request: Request) -> None:
        """Drop the per-request state the engine owns; the caller keeps the handle."""
        self._detokenizers.pop(request.request_id, None)
