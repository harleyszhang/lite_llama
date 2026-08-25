"""Continuous batching: a step-driven engine where requests join and leave mid-flight.

:class:`~lite_llama.engine.llm_engine.LLMEngine` fixes its batch when
``generate()`` is called. Every sequence starts on the same step and the batch
keeps running at full width until the *longest* one finishes, so a batch of eight
where seven stop at 20 tokens and one runs to 500 spends most of its time
computing 480 steps of padding. It also cannot accept a request that arrives one
millisecond after the call started.

:class:`ContinuousBatchingEngine` replaces the fixed batch with a per-step
decision built from two collaborating strategies, mirroring vLLM v1's split of
the engine loop into "schedule → execute passes → harvest":

* :class:`_PrefillPass` runs prompt chunks — a padded grid through the prefill
  kernel for a request's first chunk, one decode-style row per token for a
  resumed chunk whose prefix is already cached — and samples the first token of
  whichever prefills completed;
* :class:`_DecodePass` runs one decode token for every fully-prefilled request,
  owning the device-side batch state that makes consecutive steps sync-free.

Both passes can run in the same step — chunked prefill interleaves with decode
instead of stalling it — and :meth:`step` is the template method that sequences
them and harvests the emitted tokens. One host-device synchronisation per step is
deliberate: sampled tokens are read back to detokenise and to decide who stops,
which is what retires a finished request on the very next step.

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

import torch

from ..distributed.parallel_state import broadcast_tp, get_tp_world_size
from ..models.config import read_model_type
from ..models.registry import ModelRegistry
from .detokenizer import IncrementalDetokenizer
from .llm_engine import LLMEngine
from .outputs import CompletionOutput, RequestOutput
from .sampler import BatchedSamplingParams, GeneratedSpan, SamplingParams
from .scheduler import Request, Scheduler, SchedulerConfig
from .stop_criteria import POLL_INTERVAL, detect_repetition


def _sync_tp(tokens: torch.Tensor) -> torch.Tensor:
    """Broadcast rank 0's sampled ids so every TP rank decodes identically.

    Non-greedy sampling draws from a per-rank RNG, so without this the ranks
    would diverge one token in and corrupt every later step.
    """
    if get_tp_world_size() > 1:
        return broadcast_tp(tokens)
    return tokens


class _BatchTensors:
    """Device-side per-row state for whichever requests are decoding.

    Everything here is indexed by position in the current decode batch, so it
    is only valid while the running set is unchanged; :meth:`matches` reports
    that, and :meth:`advance` moves a still-valid batch one token forward
    without touching the host. The generated-token grid is the reason this
    exists at all: the repetition penalty needs every token a sequence has
    produced, and rebuilding that as a padded host tensor each step would cost
    more Python than the decode step costs GPU.

    Args:
        slots: Cache slot per running request.
        gen_counts: Tokens already generated per running request.
        params: Sampling configuration per running request.
        gen_grid: Shared ``[num_slots, max_seq_len]`` grid of generated tokens.
        device: Torch device string.
    """

    def __init__(
        self,
        slots: Sequence[int],
        gen_counts: Sequence[int],
        params: Sequence[SamplingParams],
        gen_grid: torch.Tensor,
        device: str,
    ) -> None:
        self.host_slots = list(slots)
        self._gen_grid = gen_grid
        self.slots = torch.tensor(self.host_slots, dtype=torch.long, device=device)
        # Column each row's *next* token goes to; also the count of tokens it has
        # already produced, which is what bounds the repetition-penalty window.
        self.gen_pos = torch.tensor(list(gen_counts), dtype=torch.long, device=device)
        self.sampling = BatchedSamplingParams.build(params, device)

    def matches(self, slots: Sequence[int]) -> bool:
        """Whether this state still describes ``slots``, in the same order."""
        return self.host_slots == list(slots)

    def advance(self) -> None:
        """Move every row one generated token forward, on-device."""
        self.gen_pos += 1

    def last_tokens(self) -> torch.Tensor:
        """``[batch]`` most recently generated token per row — the next decode input."""
        return self._gen_grid[self.slots, self.gen_pos - 1]

    def record(self, next_token: torch.Tensor) -> None:
        """Write this step's sampled tokens into the generated grid."""
        self._gen_grid[self.slots, self.gen_pos] = next_token

    def generated_span(self, columns: torch.Tensor, width: int) -> GeneratedSpan:
        """Padded view of every generated token, for the repetition penalty.

        Args:
            columns: Cached ``arange(max_seq_len)`` used to slice the grid.
            width: Longest generated sequence in the batch; the grid is far
                wider, and gathering all of it would copy megabytes per step.
        """
        cols = columns[:width]
        token_ids = self._gen_grid[self.slots.unsqueeze(1), cols.unsqueeze(0)]
        return GeneratedSpan(token_ids, cols.unsqueeze(0) < self.gen_pos.unsqueeze(1))


class _StepPass:
    """Strategy: one model pass over the work the scheduler handed a step.

    A pass converts ``(requests, chunk lens)`` into ``(request, token)`` pairs —
    its only contract with :meth:`ContinuousBatchingEngine.step`, which owns
    harvesting. Subclasses hold whatever per-pass device state they need; the
    engine keeps no batch state of its own.

    Args:
        engine: Owning engine, for the executor, sampler and shared tensors.
    """

    def __init__(self, engine: ContinuousBatchingEngine) -> None:
        self._engine = engine
        # Collaborators resolved once; the engine's tokenizer and runner are
        # immutable after construction.
        self._runner = engine.engine.model_runner
        self._sampler = engine.engine.sampler
        self._device = engine.device
        self._pad_id = engine.engine.pad_id


class _PrefillPass(_StepPass):
    """Runs prompt chunks and samples the first token of whichever complete.

    A chunk routes by whether its request's slot already holds K/V from an
    earlier chunk:

    * a *first* chunk (``num_computed_tokens == chunk``) runs as a padded grid
      through the prefill kernel — pure self-attention over the grid, the cheap
      path, correct because nothing of the prompt is cached yet;
    * a *resumed* chunk cannot take it: the prefill kernel never reads the
      cache, so its tokens would attend only within the chunk and silently drop
      the prefix. Those tokens instead expand into one decode-style row each
      (:meth:`SlotBatch.begin_extend`), each query attending over its slot's
      whole cached history — exact causal extend semantics at one row per token.

    Both routes write K/V at each token's own cache row; only requests whose
    chunk completes the prompt get a sampled token this step.
    """

    def run(self, group: list[Request], chunk_lens: list[int]) -> list[tuple[Request, int]]:
        pairs = list(zip(group, chunk_lens, strict=True))
        first = [(r, c) for r, c in pairs if r.num_computed_tokens == c]
        resumed = [(r, c) for r, c in pairs if r.num_computed_tokens > c]
        emitted: list[tuple[Request, int]] = []
        if first:
            emitted += self._run_grid(first)
        if resumed:
            emitted += self._run_extend(resumed)
        return emitted

    def _run_grid(self, chunks: list[tuple[Request, int]]) -> list[tuple[Request, int]]:
        """First chunks as one padded grid through the prefill kernel."""
        engine = self._engine
        slots = [r.slot for r, _ in chunks]
        starts = [r.num_computed_tokens - c for r, c in chunks]
        ends = [r.num_computed_tokens for r, _ in chunks]
        width = max(c for _, c in chunks)

        rows = [
            request.prompt_token_ids[start:end] + [self._pad_id] * (width - (end - start))
            for (request, _), start, end in zip(chunks, starts, ends, strict=True)
        ]
        input_ids = torch.tensor(rows, dtype=torch.long, device=self._device)
        # Padded columns run past a row's real position, but attention never
        # reads past that row's b_seq_len, so the junk positions are inert.
        positions = torch.tensor(starts, dtype=torch.long, device=self._device).unsqueeze(1)
        positions = positions + torch.arange(width, device=self._device).unsqueeze(0)

        engine.slot_batch.begin_prefill(slots, starts, ends)
        # Each sequence's next-token logits sit at its own last real position
        # rather than at the end of the padded row; the model gathers exactly
        # those rows before the lm_head GEMM.
        last = torch.tensor(ends, dtype=torch.long, device=self._device) - 1
        logits = self._runner.forward(input_ids, positions, None, logits_positions=last)

        # Only a chunk that completed its prompt has a next token to sample, and
        # a grid can mix the two: the admission budget happily takes a short
        # prompt (done in one chunk) alongside a long one (chunk-capped). So the
        # completed requests must be paired with *their own* grid rows rather
        # than with the leading rows of the batch.
        done = [
            (row, r) for row, (r, _) in enumerate(chunks) if r.num_computed_tokens == r.prompt_len
        ]
        if not done:
            return []
        rows = torch.tensor([row for row, _ in done], dtype=torch.long, device=self._device)
        return self._sample_first([request for _, request in done], logits[rows])

    def _run_extend(self, chunks: list[tuple[Request, int]]) -> list[tuple[Request, int]]:
        """Resumed chunks as one decode-style row per token."""
        engine = self._engine
        slots = [r.slot for r, _ in chunks]
        starts = [r.num_computed_tokens - c for r, c in chunks]
        ends = [r.num_computed_tokens for r, _ in chunks]

        padded = engine.slot_batch.begin_extend(slots, starts, ends)

        # Flatten chunk tokens in batch order; remember the row each completed
        # chunk's last token lands on — its logits are that request's first
        # sample. Filler rows pad the input to the metadata's row count; their
        # tokens and logits are discarded.
        tokens: list[int] = []
        last_row: list[int] = []
        completed: list[Request] = []
        for request, chunk in chunks:
            start = request.num_computed_tokens - chunk
            tokens.extend(request.prompt_token_ids[start : start + chunk])
            if request.num_computed_tokens == request.prompt_len:
                completed.append(request)
                last_row.append(len(tokens) - 1)
        tokens.extend([self._pad_id] * (padded - len(tokens)))

        input_ids = torch.tensor(tokens, dtype=torch.long, device=self._device).view(padded, 1)
        # begin_extend set b_seq_len to each row's absolute position plus one,
        # so the token's position is exactly that minus one.
        positions = (engine.slot_batch.seq_lens - 1).view(-1, 1)

        logits = self._runner.forward(input_ids, positions, None)
        if not completed:
            return []
        # Decode-style logits run [rows, 1, vocab]; a completed chunk's first
        # token comes from the last row of its stretch.
        rows = torch.tensor(last_row, dtype=torch.long, device=self._device)
        return self._sample_first(completed, logits[:, -1, :][rows])

    def _sample_first(
        self, requests: list[Request], logits: torch.Tensor
    ) -> list[tuple[Request, int]]:
        """Sample first tokens for ``requests`` and land them in the generated grid.

        A first sample has no repetition window yet, so the penalty is a no-op
        regardless of configuration; no generated span is gathered. Tokens land
        in column 0 of the generated grid, where the decode pass's rebuilt state
        expects to find them.
        """
        if not requests:
            return []
        sampling = BatchedSamplingParams.build([r.params for r in requests], self._device)
        tokens = _sync_tp(self._sampler.sample_batched(logits, sampling).reshape(-1))
        slots = torch.tensor([r.slot for r in requests], dtype=torch.long, device=self._device)
        self._engine.gen_grid[slots, 0] = tokens
        return list(zip(requests, tokens.tolist(), strict=True))


class _DecodePass(_StepPass):
    """Runs one decode token for every fully-prefilled request.

    Owns the :class:`_BatchTensors` that makes consecutive steps sync-free:
    while the running set holds steady the pass just advances positions
    on-device, and any membership change (finish, abort, admission) invalidates
    it through :meth:`invalidate` so the next run rebuilds from the host.
    """

    def __init__(self, engine: ContinuousBatchingEngine) -> None:
        super().__init__(engine)
        self._batch: _BatchTensors | None = None

    def invalidate(self) -> None:
        """Forget the cached batch; the next step rebuilds its metadata.

        Called whenever the running set changes out from under a step —
        finishes, aborts, admissions all do — because row indices are only
        meaningful while membership and order are stable.
        """
        self._batch = None

    def run(self, running: list[Request]) -> list[tuple[Request, int]]:
        engine = self._engine
        slots = [request.slot for request in running]
        # Cache length once this step's token is written: the request already
        # counts the token it is about to feed in ``output_token_ids``.
        seq_lens = [request.seq_len for request in running]

        batch = self._batch
        if batch is not None and batch.matches(slots):
            batch.advance()
        else:
            batch = _BatchTensors(
                slots,
                [len(r.output_token_ids) for r in running],
                [r.params for r in running],
                engine.gen_grid,
                self._device,
            )
            self._batch = batch

        padded = engine.slot_batch.begin_decode(slots, seq_lens)
        input_ids = self._decode_inputs(batch.last_tokens(), len(running), padded)
        # Position of the token being fed is its cache row, i.e. length minus one.
        positions = engine.slot_batch.seq_lens.view(-1, 1) - 1

        logits = self._runner.forward(input_ids, positions, None)
        logits = logits[: len(running), -1, :]

        generated = None
        if batch.sampling.any_penalty:
            width = max(len(r.output_token_ids) for r in running)
            generated = batch.generated_span(engine.columns, width)

        tokens = _sync_tp(
            self._sampler.sample_batched(logits, batch.sampling, generated).reshape(-1)
        )
        batch.record(tokens)
        return list(zip(running, tokens.tolist(), strict=True))

    def _decode_inputs(self, last_tokens: torch.Tensor, size: int, padded: int) -> torch.Tensor:
        """Shape this step's input tokens to the size the model will be called with."""
        if padded == size:
            return last_tokens.view(size, 1)
        # Filler rows exist only to reach a captured graph batch size; whatever
        # token id they carry is thrown away with their logits.
        staging = self._engine._decode_input
        staging[:size, 0] = last_tokens
        return staging[:padded]


class ContinuousBatchingEngine:
    """Serves independently arriving requests as one continuously reshaped batch.

    Drive it by calling :meth:`step` in a loop. Each call runs the scheduler's
    plan for the step — a prefill pass over newly admitted or resumed chunks,
    then a decode pass over everything running — and returns the requests that
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

    Raises:
        NotImplementedError: The checkpoint is multimodal. Vision prefill needs
            per-request processor outputs, which the batched prefill grid here
            has no place for.
    """

    def __init__(self, engine: LLMEngine, config: SchedulerConfig | None = None) -> None:
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

        self._slot_batch = engine.model_runner.enable_slot_kv_cache()
        # Slot ids stay below max_num_seqs, which keeps the generated-token grid
        # proportional to the concurrency the caller actually asked for rather
        # than to however many slots happen to fit in the cache.
        num_slots = min(self._slot_batch.num_slots, config.max_num_seqs)
        self.scheduler = Scheduler(config, num_slots)

        self.gen_grid = torch.zeros(
            (num_slots, config.max_seq_len), dtype=torch.long, device=self.device
        )
        self.columns = torch.arange(config.max_seq_len, device=self.device)
        # Decode inputs for a graph-padded batch: written on-device every step,
        # so filler rows keep whatever token id was there last. They are discarded.
        self._decode_input = torch.zeros(
            (self._slot_batch.num_slots + 1, 1), dtype=torch.long, device=self.device
        )

        self._prefill_pass = _PrefillPass(self)
        self._decode_pass = _DecodePass(self)
        self._detokenizers: dict[str, IncrementalDetokenizer] = {}
        self._request_ids = itertools.count()
        self._step_count = 0

    @property
    def slot_batch(self):
        """The fixed-slot KV view both passes drive."""
        return self._slot_batch

    # ------------------------------------------------------------------ build #
    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        tokenizer: str | None = None,
        max_seq_len: int = 2048,
        max_num_seqs: int = 32,
        max_num_batched_tokens: int = 8192,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool = True,
        quantization: str | None = None,
        tensor_parallel_size: int = 1,
        kv_cache_dtype: str = "auto",
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
                steps stay on the graph path.
            quantization: Runtime weight quantisation, forwarded to the engine.
                Orthogonal to batching -- it changes the linear layers, not the
                KV cache or the schedule.
            tensor_parallel_size: Must be 1; see below.
            kv_cache_dtype: KV-cache element type, forwarded to the engine
                (``"auto"`` for fp16, or an fp8 spelling to halve the cache).

        Raises:
            NotImplementedError: The checkpoint is multimodal, or tensor
                parallelism was requested.
        """
        if tensor_parallel_size != 1:
            # TP is driven by the multi-process worker path in lite_llama.cli,
            # which owns one engine per rank. Combining that with this engine's
            # single worker thread and per-request slot pool is a real design
            # question, not a parameter to forward -- so it is refused rather
            # than accepted and silently ignored.
            raise NotImplementedError(
                "continuous batching does not support tensor_parallel_size > 1 yet; "
                "use `lite-llama chat` for tensor-parallel inference"
            )

        spec = ModelRegistry.resolve(read_model_type(model))
        if spec.is_multimodal:
            raise NotImplementedError(
                "continuous batching supports text-only checkpoints; "
                "use LLM.generate() for vision-language models"
            )

        engine = LLMEngine(
            checkpoints_dir=model,
            tokenizer_path=tokenizer,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            device=device,
            use_cuda_graph=use_cuda_graph,
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
        )
        return cls(
            engine,
            SchedulerConfig(
                max_seq_len=engine.max_seq_len,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
            ),
        )

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
        request = Request(
            request_id=request_id or f"req-{next(self._request_ids)}",
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

        The template is fixed — schedule, execute the prefill pass, execute the
        decode pass, harvest — and each stage is a method, so behaviour varies
        only through the passes themselves.

        Returns:
            The requests that produced a token this step, in pass order. Each
            carries this step's text in ``delta``; those that stopped also carry
            a ``finish_reason``.
        """
        scheduled = self.scheduler.schedule()
        if scheduled.is_empty:
            return []

        self._step_count += 1
        emitted: list[tuple[Request, int]] = []
        if scheduled.prefill:
            emitted += self._prefill_pass.run(scheduled.prefill, scheduled.prefill_chunk_lens)
        if scheduled.decode:
            emitted += self._decode_pass.run(scheduled.decode)
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
                # one-shot path drops it too.
                self._finish(request, "eos")
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
        # The running set just changed, so the next step must rebuild its
        # device-side metadata instead of incrementing the cached batch's.
        self._decode_pass.invalidate()
        self._slot_batch.reset()
