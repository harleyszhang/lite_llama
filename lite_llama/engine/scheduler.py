"""Request scheduling with chunked prefill and prefix caching.

The scheduler decides per step: which requests prefill (in chunks), which decode,
and which cache slot each holds. Chunked prefill splits long prompts so decode
steps interleave with prefill chunks — preventing head-of-line blocking where a
4K-token prompt stalls all running decode requests for hundreds of ms.

Scheduling is *committing* (mirrors vLLM v1): :meth:`Scheduler.schedule` advances
each prefilling request's ``num_computed_tokens`` the moment it schedules a chunk
of it, so the engine never reports execution progress back to the scheduler. The
scheduler may therefore assume a returned chunk ran; a caller who crashed mid-step
re-derives everything from the request objects, which are the single source of
truth for prefill state.

Key features:
    - Chunked prefill: long prompts split into max_chunk_size token chunks;
      several partial prefills may share one step within the token budget.
    - Preemption: requests can be evicted when KV pressure hits the watermark.
    - Prefix caching (hash-based): shared prompt prefixes reuse cached KV.

Usage:
    sched = Scheduler(SchedulerConfig(max_num_seqs=16, max_chunk_size=512), num_slots=64)
    sched.add_request(Request(...))
    output = sched.schedule()  # chunk progress already advanced on return
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from .prefix_cache import PrefixCache
from .sampler import SamplingParams


class RequestStatus(StrEnum):
    """Where a request sits in its lifecycle.

    ``WAITING`` requests hold no cache slot; ``RUNNING`` ones own exactly one and
    are part of every step until they finish.
    """

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    """One generation request, from arrival to completion.

    Carries both the inputs and everything the engine accumulates about it, so a
    caller holding a :class:`Request` can report progress without consulting the
    engine. ``delta`` is deliberately per-step scratch: the engine overwrites it
    each step and streaming callers drain it, while ``text`` keeps the whole
    completion for callers that only want the final answer.

    Attributes:
        request_id: Caller-visible identifier, unique among live requests.
        prompt: The prompt as submitted, kept for echoing back in the response.
        prompt_token_ids: Tokenised prompt.
        params: Per-request sampling configuration.
        max_new_tokens: Generation cap, resolved against the context window at
            admission so the scheduler never has to consult the engine.
        num_computed_tokens: Prompt tokens whose KV is already in the cache —
            the cached prefix plus every chunk scheduled so far. The next chunk
            of this request starts at exactly this offset.
        arrival_time: ``time.monotonic()`` when the request entered the queue.
        status: Lifecycle position.
        slot: Cache slot while running, ``None`` otherwise.
        output_token_ids: Tokens generated so far.
        text: Detokenised completion so far.
        delta: Text produced by the most recent step only.
        finish_reason: ``"eos"``, ``"length"``, ``"repeat"`` or ``"abort"``.
        first_token_time: When the first token became visible (for TTFT).
        finish_time: When the request finished.
    """

    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    params: SamplingParams
    max_new_tokens: int = 0
    num_computed_tokens: int = 0
    arrival_time: float = field(default_factory=time.monotonic)
    status: RequestStatus = RequestStatus.WAITING
    slot: int | None = None
    output_token_ids: list[int] = field(default_factory=list)
    text: str = ""
    delta: str = ""
    finish_reason: str | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    #: Leading prompt tokens served from the prefix cache; their KV is reused,
    #: so prefill starts from this offset instead of from 0.
    num_cached_tokens: int = 0

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def seq_len(self) -> int:
        """Tokens currently in this request's KV cache."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status is RequestStatus.FINISHED

    @property
    def has_room(self) -> bool:
        """Whether the generation cap still allows another token."""
        return len(self.output_token_ids) < self.max_new_tokens

    @property
    def prefill_done(self) -> bool:
        """Whether the whole prompt is in the cache, i.e. decode may proceed."""
        return self.num_computed_tokens >= self.prompt_len


@dataclass(frozen=True)
class SchedulerConfig:
    """Limits the admission policy enforces.

    Attributes:
        max_seq_len: Context window; also the per-slot cache capacity.
        max_num_seqs: Ceiling on concurrently running requests.
        max_num_batched_tokens: Ceiling on the padded token count of one
            prefill group. Measured on *chunks*, not whole prompts: with
            chunked prefill a step's grid is as wide as its longest chunk.
        max_chunk_size: Maximum tokens per prefill chunk. When a prompt is
            longer than this, it is split into chunks and interleaved with
            decode steps (chunked prefill). Set to 0 to disable chunking.
        enable_prefix_cache: Whether to use hash-based prefix caching.
        enable_preemption: When True, ``max_num_seqs`` is honoured as a desired
            concurrency even beyond the slot count: an oversubscribed batch
            time-shares slots by preempting (recompute) the youngest running
            request to admit an older waiting one. When False (default) the
            batch is capped at the slot count and nothing is ever evicted.
    """

    max_seq_len: int = 2048
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 8192
    max_chunk_size: int = 512
    enable_prefix_cache: bool = False
    enable_preemption: bool = False


@dataclass(frozen=True)
class SchedulerOutput:
    """What one engine step should run.

    Prefill and decode coexist in one step: the engine executes the prefill
    pass, then the decode pass, so chunked prefill never stalls decode.

    Attributes:
        prefill: Requests receiving prefill work this step, in batch order.
        decode: Requests receiving one decode token this step.
        prefill_chunk_lens: Tokens to process per prefill request. A request
            whose chunk completes its prompt produces its first sampled token
            this step; a partial chunk produces none.
        preempted: Requests evicted this step (for logging/metrics).
    """

    prefill: list[Request] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)
    prefill_chunk_lens: list[int] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill and not self.decode


class _NullPrefixCache:
    """Null Object standing in for a disabled :class:`PrefixCache`.

    Lets the admission path run one branch-free version of itself whether or
    not prefix caching is on; the no-op answers are exactly what the
    ``if cache is None`` branches used to compute.
    """

    def query(self, token_ids: list[int]) -> int:
        return 0

    def register(self, token_ids: list[int]) -> None:
        pass

    def release(self, token_ids: list[int]) -> None:
        pass

    @property
    def hit_rate(self) -> float:
        return 0.0


def _discard(requests: list[Request], request: Request) -> None:
    """Remove ``request`` from ``requests`` by identity, preserving order.

    :class:`Request` is a value-equal dataclass, so ``list.remove`` could drop a
    *different* request whose fields happen to match; identity is the only safe
    key once two live requests can carry equal prompts.
    """
    for index, candidate in enumerate(requests):
        if candidate is request:
            del requests[index]
            return


class Scheduler:
    """FCFS admission with chunked prefill and preemption support.

    The scheduling step is a three-stage template (mirrors vLLM v1's
    ``schedule``): resume in-flight partial prefills, admit queued requests
    into whatever budget and slots remain, then collect the decode batch.
    Every stage advances ``num_computed_tokens`` on the request itself as it
    commits work, which is why no external ``advance`` protocol exists.

    Args:
        config: Admission and chunking limits.
        num_slots: Cache slots available.
    """

    def __init__(self, config: SchedulerConfig, num_slots: int) -> None:
        if num_slots < 1:
            raise ValueError(f"need at least one cache slot, got {num_slots}")
        self.config = config
        self.num_slots = num_slots
        # Preemption lets the running set exceed the slot count (slots are
        # time-shared via recompute); otherwise concurrency is slot-capped.
        self.max_num_seqs = (
            config.max_num_seqs
            if config.enable_preemption
            else min(config.max_num_seqs, num_slots)
        )

        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []
        self._free_slots: list[int] = list(reversed(range(num_slots)))
        # Null Object when disabled, so the hot path pays no branches for the
        # feature being off.
        self._prefix_cache: PrefixCache | _NullPrefixCache = (
            PrefixCache(block_size=16) if config.enable_prefix_cache else _NullPrefixCache()
        )
        self.num_preemptions: int = 0

    # ------------------------------------------------------------------ queue #
    def add_request(self, request: Request) -> None:
        """Queue a request, resolving its generation cap against the context window.

        Raises:
            ValueError: The prompt is empty, or leaves no room to generate.
        """
        limit = self.config.max_seq_len
        if request.prompt_len == 0:
            raise ValueError(f"request {request.request_id} has an empty prompt")
        if request.prompt_len >= limit:
            raise ValueError(
                f"request {request.request_id} prompt length {request.prompt_len} "
                f"leaves no room under max_seq_len {limit}"
            )

        room = limit - request.prompt_len
        requested = request.params.max_gen_len
        request.max_new_tokens = min(requested, room) if requested else room
        request.status = RequestStatus.WAITING
        self._waiting.append(request)

    def abort(self, request_id: str) -> Request | None:
        """Drop a request wherever it is; returns it, or ``None`` if unknown.

        A running request releases its slot immediately, so an abandoned HTTP
        connection frees capacity on the next step rather than at its length cap.
        """
        for request in self._waiting:
            if request.request_id == request_id:
                self._waiting.remove(request)
                request.status = RequestStatus.FINISHED
                request.finish_reason = "abort"
                return request
        for request in self._running:
            if request.request_id == request_id:
                self.finish(request, "abort")
                return request
        return None

    # -------------------------------------------------------------- scheduling #
    def schedule(self) -> SchedulerOutput:
        """Decide this step's work: chunked prefill + decode batch.

        Prefill and decode coexist: partial prefill chunks for some requests
        and decode tokens for the rest run in the same step, so a long prompt
        stretches over several steps without freezing anyone's decode. Chunk
        progress is committed here — the returned output describes work already
        accounted for in each request's ``num_computed_tokens``.
        """
        prefill: list[Request] = []
        chunk_lens: list[int] = []

        # Stage 1: resume requests whose prefill is still in flight. Their
        # slot and KV are held, so they come before new admissions — FCFS by
        # arrival, and re-admitting a new request first would starve them.
        for request in self._running:
            if request.prefill_done:
                continue
            chunk = self._chunk_of(request)
            prefill.append(request)
            chunk_lens.append(chunk)
            request.num_computed_tokens += chunk

        # Stage 2: admit queued requests into the remaining budget and slots.
        preempted = self._admit(prefill, chunk_lens)

        # Stage 3: the decode batch is every running request that finished
        # prefill before this step. Requests whose prefill completes *this*
        # step produce their first token in the prefill pass and join decode
        # on the next one; partial prefills produce nothing yet.
        in_prefill = {id(request) for request in prefill}
        decode = [
            request
            for request in self._running
            if request.prefill_done and id(request) not in in_prefill
        ]

        return SchedulerOutput(prefill=prefill, decode=decode,
                               prefill_chunk_lens=chunk_lens, preempted=preempted)

    def _admit(self, prefill: list[Request], chunk_lens: list[int]) -> list[Request]:
        """Admit waiting requests, committing their first chunk (or whole prompt).

        A newly admitted request's first chunk is its uncached remainder capped
        at ``max_chunk_size``; short prompts therefore finish prefill in this
        very step, while long ones re-enter through Stage 1 on later steps.

        Returns:
            Requests preempted to make room, for reporting.
        """
        preempted: list[Request] = []
        longest = max(chunk_lens, default=0)

        while self._waiting:
            capacity = self.max_num_seqs - len(self._running)
            if capacity <= 0:
                break
            candidate = self._waiting[0]
            if not self._free_slots:
                preempted_victim = self._maybe_preempt(candidate, prefill)
                if preempted_victim is None:
                    break
                preempted.append(preempted_victim)

            # The grid cost of adding this request is its chunk, not its whole
            # prompt: with chunking, a 4K prompt occupies one chunk column per
            # step, which is exactly what the budget should price.
            chunk = self._first_chunk_of(candidate)
            padded = max(longest, chunk) * (len(prefill) + 1)
            if prefill and padded > self.config.max_num_batched_tokens:
                break

            self._waiting.popleft()
            candidate.slot = self._free_slots.pop()
            candidate.status = RequestStatus.RUNNING
            self._running.append(candidate)

            # Prefix cache: reuse KV of any leading blocks already cached, then
            # register this prompt so later requests with the same prefix hit
            # it. Never skip the whole prompt: at least one token must be
            # prefilled to produce the first logits, exactly as vLLM keeps the
            # last block uncached.
            cached = min(self._prefix_cache.query(candidate.prompt_token_ids),
                         candidate.prompt_len - 1)
            candidate.num_cached_tokens = cached
            self._prefix_cache.register(candidate.prompt_token_ids)
            # num_computed_tokens tracks KV *actually resident in the slot*:
            # the hash cache is bookkeeping until block copy lands, so prefill
            # still starts at 0 and cached stays advisory (future KV-copy path
            # will fast-forward computed past the copied prefix).
            candidate.num_computed_tokens = 0

            prefill.append(candidate)
            chunk_lens.append(chunk)
            candidate.num_computed_tokens += chunk
            longest = max(longest, chunk)

        return preempted

    def _chunk_of(self, request: Request) -> int:
        """Next chunk for a request whose prefill is already under way."""
        remaining = request.prompt_len - request.num_computed_tokens
        size = self.config.max_chunk_size
        return remaining if size <= 0 else min(remaining, size)

    def _first_chunk_of(self, request: Request) -> int:
        """First chunk for a fresh admission: the whole prompt, chunk-capped.

        Prefix-cache hits do not shrink this chunk: the cached KV is not yet
        copied into the slot (see :meth:`_admit`), so every token still has to
        run through prefill.
        """
        size = self.config.max_chunk_size
        return request.prompt_len if size <= 0 else min(request.prompt_len, size)

    def _maybe_preempt(self, newcomer: Request, prefill: list[Request]) -> Request | None:
        """Free a slot for ``newcomer`` by evicting a running request, if allowed.

        The victim is the youngest running request eligible for preemption;
        ``None`` means nobody may be evicted and admission stops. Fixing the
        intended newcomer before any preemption (as the caller does by peeking
        the queue head) keeps a re-queued victim from jumping ahead of it.
        """
        if not self.config.enable_preemption:
            return None
        victim = self._pick_preemption_victim(prefill)
        if victim is None:
            return None
        self._preempt(victim)
        return victim

    def _pick_preemption_victim(self, prefill: list[Request]) -> Request | None:
        """Youngest running request eligible for preemption, or None.

        Eligible = past prefill (evicting a partial prefill would discard real
        KV work), not part of this step's prefill group, and past the progress
        quantum (>=1 output token). The quantum stops a just-recomputed
        request from being evicted again before it makes progress, so the
        recompute cycle cannot livelock.
        """
        scheduled = {id(request) for request in prefill}
        for request in reversed(self._running):
            if not request.prefill_done or id(request) in scheduled:
                continue
            if len(request.output_token_ids) >= 1:
                return request
        return None

    def _preempt(self, request: Request) -> None:
        """Evict a running request back to the waiting queue (recompute strategy).

        The KV built so far is dropped: on re-admission the prompt is prefilled
        again from its cached-prefix offset. Generated tokens are cleared so the
        request restarts decoding from where recompute leaves off (its already
        emitted tokens are preserved by the engine's output buffer, not here).
        """
        if request.slot is not None:
            self._free_slots.append(request.slot)
            request.slot = None
        request.status = RequestStatus.WAITING
        request.num_computed_tokens = 0
        request.num_cached_tokens = 0
        request.output_token_ids.clear()
        self._discard_running(request)
        self._waiting.appendleft(request)  # re-queue at front, keeps FCFS age
        self.num_preemptions += 1

    def finish(self, request: Request, reason: str) -> None:
        """Retire a running request and return its slot to the pool."""
        if request.status is RequestStatus.FINISHED:
            return
        request.status = RequestStatus.FINISHED
        request.finish_reason = reason
        request.finish_time = time.monotonic()
        if request.slot is not None:
            self._free_slots.append(request.slot)
            request.slot = None
        self._discard_running(request)
        # Drop this request's hold on its cached prefix blocks. A shared prefix
        # survives as long as any other live request still references it.
        self._prefix_cache.release(request.prompt_token_ids)

    def _discard_running(self, request: Request) -> None:
        """Remove ``request`` from the running list by identity."""
        _discard(self._running, request)

    @property
    def prefix_cache_hit_rate(self) -> float:
        """Fraction of queried prompt tokens served from the prefix cache."""
        return self._prefix_cache.hit_rate

    # ------------------------------------------------------------------ status #
    @property
    def running(self) -> list[Request]:
        """Requests in the decode batch, in admission order."""
        return list(self._running)

    @property
    def waiting(self) -> list[Request]:
        """Queued requests, in arrival order."""
        return list(self._waiting)

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def has_unfinished_requests(self) -> bool:
        """Whether anything is queued or in flight."""
        return bool(self._waiting or self._running)
