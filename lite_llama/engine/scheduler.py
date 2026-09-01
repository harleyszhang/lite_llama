"""Request scheduling with chunked prefill and prefix caching.

Per step the :class:`Scheduler` decides which requests prefill (in chunks),
which decode, and which slot each holds, so a long prompt cannot stall
running decodes. Scheduling commits: once planned, a chunk is assumed to
have run, and the request objects are the single source of truth.

Usage:
    scheduler = Scheduler(SchedulerConfig(...), num_slots)
    scheduler.add_request(request); plan = scheduler.schedule()
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum

from .prefix_cache import PREFIX_CACHE_BLOCK_SIZE, PrefixCache, PrefixMatch
from .sampler import PositionLogprobs, SamplingParams


class RequestStatus(StrEnum):
    """Where a request sits in its lifecycle.

    ``WAITING`` requests hold no cache slot; ``RUNNING`` ones own exactly one and
    are part of every step until they finish.
    """

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(slots=True)
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
        scheduled_time: When the request last left the queue for a slot — the
            queue wait is ``scheduled_time - arrival_time`` (re-admission after
            a preemption restarts it, since the request was waiting again).
        status: Lifecycle position.
        slot: Cache slot while running, ``None`` otherwise.
        output_token_ids: Tokens generated so far.
        text: Detokenised completion so far.
        delta: Text produced by the most recent step only.
        finish_reason: ``"eos"``, ``"length"``, ``"repeat"`` or ``"abort"``.
        first_token_time: When the first token became visible (for TTFT).
        finish_time: When the request finished.
        num_cached_tokens: Leading prompt tokens this request reused from the
            prefix cache instead of prefilling: their K/V was copied into this
            slot before the first chunk ran, so ``num_computed_tokens`` starts
            here rather than at 0.
        prefix_copies: Per-step scratch, set only on the step this request is
            admitted: ``(src_slot, start_token, num_tokens)`` runs of prefix K/V
            to copy into this request's slot before its first chunk runs. Empty
            on a miss, and on every later chunk.
        prompt_logprobs: Per-position records for the prompt, ``prompt_len``
            long once prefill completes; position 0 and prefix-cache hits stay
            ``None`` (their predictor never ran). Built chunk by chunk; ``None``
            until the first chunk of a request that asked arrives, ``None``
            forever when it did not.
        output_logprobs: Per-token records for the generated span, parallel to
            ``output_token_ids``; ``None`` when the request did not ask.
        delta_logprobs: Per-step scratch like ``delta``: the record of the token
            this step produced, drained by the streaming layer. ``None`` on
            steps with no new token, and for requests that did not ask.
    """

    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    params: SamplingParams
    max_new_tokens: int = 0
    num_computed_tokens: int = 0
    arrival_time: float = field(default_factory=time.monotonic)
    scheduled_time: float | None = None
    status: RequestStatus = RequestStatus.WAITING
    slot: int | None = None
    output_token_ids: list[int] = field(default_factory=list)
    text: str = ""
    delta: str = ""
    finish_reason: str | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    num_cached_tokens: int = 0
    prefix_copies: tuple[tuple[int, int, int], ...] = ()
    prompt_logprobs: list[PositionLogprobs | None] | None = None
    output_logprobs: list[PositionLogprobs] | None = None
    delta_logprobs: PositionLogprobs | None = None

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


DEFAULT_MAX_NUM_SEQS = 32
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192


@dataclass(frozen=True, slots=True)
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
        prefix_cache_blocks: Resident-block ceiling for the prefix cache.
            ``None`` derives one from the cache geometry (see
            :meth:`Scheduler._default_prefix_capacity`); an explicit value is for
            deployments that want the bookkeeping bounded tighter than that.
        enable_preemption: When True, ``max_num_seqs`` is honoured as a desired
            concurrency even beyond the slot count: an oversubscribed batch
            time-shares slots by preempting (recompute) the youngest running
            request to admit an older waiting one. When False (default) the
            batch is capped at the slot count and nothing is ever evicted.
    """

    max_seq_len: int = 2048
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS
    max_chunk_size: int = 512
    enable_prefix_cache: bool = False
    prefix_cache_blocks: int | None = None
    enable_preemption: bool = False

    def __post_init__(self) -> None:
        if self.max_seq_len < 2:
            raise ValueError(f"max_seq_len must be >= 2, got {self.max_seq_len}")
        if self.max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be >= 1, got {self.max_num_seqs}")
        if self.max_num_batched_tokens < 1:
            raise ValueError(
                f"max_num_batched_tokens must be >= 1, got {self.max_num_batched_tokens}"
            )
        if self.max_chunk_size < 0:
            raise ValueError(f"max_chunk_size must be >= 0, got {self.max_chunk_size}")
        if self.prefix_cache_blocks is not None and self.prefix_cache_blocks < 1:
            raise ValueError(
                f"prefix_cache_blocks must be >= 1 or None, got {self.prefix_cache_blocks}"
            )


@dataclass(frozen=True, slots=True)
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

    def admit(self, token_ids: list[int]) -> PrefixMatch:
        return PrefixMatch()

    def invalidate_slot(self, slot: int) -> None:
        pass

    def assign_owner(self, token_ids: list[int], slot: int, upto_tokens: int) -> None:
        pass

    def release(self, token_ids: list[int]) -> None:
        pass

    @property
    def hit_rate(self) -> float:
        return 0.0


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
            config.max_num_seqs if config.enable_preemption else min(config.max_num_seqs, num_slots)
        )

        # Ordered map = FCFS iteration plus constant-time cancellation. Serving
        # queues can be much deeper than the running batch, so a deque scan on
        # every disconnected client becomes measurable under overload.
        self._waiting: OrderedDict[str, Request] = OrderedDict()
        self._running: list[Request] = []

        self._requests: dict[str, Request] = {}
        self._free_slots: list[int] = list(reversed(range(num_slots)))

        self._prefix_cache: PrefixCache | _NullPrefixCache = (
            PrefixCache(
                block_size=PREFIX_CACHE_BLOCK_SIZE,
                capacity=config.prefix_cache_blocks or self._default_prefix_capacity(num_slots),
            )
            if config.enable_prefix_cache
            else _NullPrefixCache()
        )

        self._pending_owners: list[tuple[Request, int, int]] = []
        self.num_preemptions: int = 0

    def _default_prefix_capacity(self, num_slots: int) -> int:
        """Resident-block ceiling to use when the deployment does not name one.

        An unbounded prefix cache is a leak rather than a cache: a block whose
        ``ref_cnt`` falls to zero stays resident forever by design (that is what
        keeps a shared system prompt warm), so a server that never restarts
        accumulates one entry per distinct block it has ever hashed. Bounding it
        at the number of blocks the KV cache itself could hold keeps the
        bookkeeping proportional to the hardware, and costs no hit rate: a prefix
        too large to ever be resident is one this cache could not serve anyway.
        """
        return max(1, num_slots * self.config.max_seq_len // PREFIX_CACHE_BLOCK_SIZE)

    # ------------------------------------------------------------------ queue #
    def add_request(self, request: Request) -> None:
        """Queue a request, resolving its generation cap against the context window.

        Raises:
            ValueError: The prompt is empty, or leaves no room to generate.
        """
        if request.request_id in self._requests:
            raise ValueError(f"request id {request.request_id!r} is already active")
        if request.status is not RequestStatus.WAITING or request.slot is not None:
            raise ValueError("only a fresh waiting request can be submitted")

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
        self._requests[request.request_id] = request
        self._waiting[request.request_id] = request

    def abort(self, request_id: str) -> Request | None:
        """Drop a request wherever it is; returns it, or ``None`` if unknown.

        A running request releases its slot immediately, so an abandoned HTTP
        connection frees capacity on the next step rather than at its length cap.
        """
        request = self._requests.get(request_id)
        if request is None:
            return None
        if request.status is RequestStatus.WAITING:
            self._waiting.pop(request_id, None)
            request.status = RequestStatus.FINISHED
            request.finish_reason = "abort"
            request.finish_time = time.monotonic()
            self._requests.pop(request_id, None)
        else:
            self.finish(request, "abort")
        return request

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

        # Stage 0: last step's prefills have executed by now, so the prefixes
        # they computed are really in their slots and may be copied from.
        self._promote_pending_owners()

        # Stage 1: resume requests whose prefill is still in flight. Their
        # slot and KV are held, so they come before new admissions — FCFS by
        # arrival, and re-admitting a new request first would starve them.
        for request in self._running:
            if request.prefill_done:
                continue
            # Copies belong to the admission step only; a resumed chunk's prefix
            # is already in its slot.
            request.prefix_copies = ()
            chunk = self._chunk_of(request)
            prefill.append(request)
            chunk_lens.append(chunk)
            request.num_computed_tokens += chunk
            self._track_owner(request)

        # Stage 2: admit queued requests into the remaining budget and slots.
        preempted = self._admit(prefill, chunk_lens)

        # Stage 3: decode every request that finished prefill before this
        # step. A prompt completed *this* step is sampled in its prefill pass
        # and joins decode on the next one; partial prefills never pass the
        # ``prefill_done`` gate.
        just_prefilled = {id(request) for request in prefill if request.prefill_done}
        decode = [
            request
            for request in self._running
            if request.prefill_done and id(request) not in just_prefilled
        ]

        return SchedulerOutput(
            prefill=prefill, decode=decode, prefill_chunk_lens=chunk_lens, preempted=preempted
        )

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
            if len(self._running) >= self.max_num_seqs:
                break
            candidate = next(iter(self._waiting.values()))

            chunk = self._chunk_of(candidate, computed=0)
            padded = max(longest, chunk) * (len(prefill) + 1)
            if prefill and padded > self.config.max_num_batched_tokens:
                break

            if not self._free_slots:
                preempted_victim = self._maybe_preempt(prefill)
                if preempted_victim is None:
                    break
                preempted.append(preempted_victim)

            # A preempted victim may just have been inserted at the head, so
            # remove the original candidate by id rather than popping the head.
            self._waiting.pop(candidate.request_id)
            candidate.slot = self._free_slots.pop()
            candidate.status = RequestStatus.RUNNING
            candidate.scheduled_time = time.monotonic()
            self._running.append(candidate)

            # Prefix cache: reuse the K/V of leading blocks that are both cached
            # and still resident in some slot, then register this prompt so later
            # requests with the same prefix hit it. Never skip the whole prompt:
            # at least one token must run to produce the first logits, exactly as
            # vLLM keeps the last block uncached.
            match = self._prefix_cache.admit(candidate.prompt_token_ids)
            cached = min(match.copyable_tokens, candidate.prompt_len - 1)
            candidate.num_cached_tokens = cached
            candidate.prefix_copies = tuple(
                run for run in match.segments if run[0] != candidate.slot
            )
            candidate.num_computed_tokens = cached

            self._prefix_cache.invalidate_slot(candidate.slot)

            chunk = self._chunk_of(candidate)

            prefill.append(candidate)
            chunk_lens.append(chunk)
            candidate.num_computed_tokens += chunk
            self._track_owner(candidate)
            longest = max(longest, chunk)

            if preempted:
                break

        return preempted

    def _chunk_of(self, request: Request, computed: int | None = None) -> int:
        """Next chunk to run, chunk- and budget-capped.

        ``computed=0`` prices a fresh admission from progress zero — an upper
        bound, since how much the prefix cache will cover is only known once
        the request holds a slot. Under-pricing would admit a group that
        overflows ``max_num_batched_tokens``.
        """
        progress = request.num_computed_tokens if computed is None else computed
        remaining = request.prompt_len - progress
        size = self.config.max_chunk_size
        if size <= 0:
            return remaining
        # vLLM's chunked prefill consumes at most the iteration token budget.
        # Without this cap, one long request violates the advertised ceiling.
        return min(remaining, size, self.config.max_num_batched_tokens)

    def _promote_pending_owners(self) -> None:
        """Hand prefix ownership to the slots that now really hold the K/V."""
        for request, slot, upto in self._pending_owners:
            if request.status is RequestStatus.RUNNING and request.slot == slot:
                self._claim_prefix(request, slot, upto)
        self._pending_owners.clear()

    def _track_owner(self, request: Request) -> None:
        """Queue an ownership claim for the chunk this step just planned.

        Per chunk, not per completed prompt: a shareable prefix is usually
        long enough to be split, and waiting for the whole prompt would leave
        its blocks unowned exactly while the requests that would reuse them
        arrive.
        """
        if request.slot is not None:
            self._pending_owners.append((request, request.slot, request.num_computed_tokens))

    def _claim_prefix(self, request: Request, slot: int, upto: int) -> None:
        """Record ``slot`` as the live copy of ``request``'s first ``upto`` tokens."""
        self._prefix_cache.assign_owner(request.prompt_token_ids, slot, upto)

    def _settle_pending_owner(self, request: Request) -> None:
        """Cash in a retiring request's claim instead of discarding it.

        Retirement runs after the step executed, so the K/V really is in the
        slot — the request that first pays for a shared prompt typically
        finishes before the ones that would inherit it arrive.
        """
        kept: list[tuple[Request, int, int]] = []
        for candidate, slot, upto in self._pending_owners:
            if candidate is request:
                self._claim_prefix(request, slot, upto)
            else:
                kept.append((candidate, slot, upto))
        self._pending_owners = kept

    def _drop_pending_owner(self, request: Request) -> None:
        """Cancel a queued ownership claim, by identity.

        Mandatory for preemption: it moves generated tokens into
        ``prompt_token_ids``, so a replayed claim would map the *new* prompt's
        hashes onto rows holding the old prompt's K/V.
        """
        self._pending_owners = [entry for entry in self._pending_owners if entry[0] is not request]

    def _maybe_preempt(self, prefill: list[Request]) -> Request | None:
        """Evict the youngest eligible running request, or ``None``.

        Eligible = past prefill, not in this step's prefill group, and past
        the progress quantum (>=1 output token) — without the quantum a
        just-recomputed request could be evicted again before making progress.
        """
        if not self.config.enable_preemption:
            return None
        scheduled = {id(request) for request in prefill}
        for request in reversed(self._running):
            if not request.prefill_done or id(request) in scheduled:
                continue
            if request.output_token_ids:
                self._preempt(request)
                return request
        return None

    def _preempt(self, request: Request) -> None:
        """Evict a running request back to the waiting queue (recompute strategy).

        The KV built so far is dropped. Generated tokens move into the prompt
        — vLLM v1's recompute semantics — so re-admission replays them and
        decoding continues the text the caller already saw. The generation
        cap shrinks by the moved tokens.
        """
        if request.slot is not None:
            self._free_slots.append(request.slot)
            request.slot = None
        request.status = RequestStatus.WAITING
        request.num_computed_tokens = 0
        request.num_cached_tokens = 0
        request.prefix_copies = ()
        # The prompt changes underneath the logprob records (generated tokens
        # move into it), so both spans restart empty for re-admission.
        request.prompt_logprobs = None
        request.output_logprobs = None
        request.delta_logprobs = None
        self._drop_pending_owner(request)
        # Release the prefix-block references; skipping this would inflate
        # ref_cnt once per preempt cycle, and referenced blocks never evict.
        self._prefix_cache.release(request.prompt_token_ids)
        moved = len(request.output_token_ids)
        request.prompt_token_ids.extend(request.output_token_ids)
        request.output_token_ids.clear()
        request.max_new_tokens -= moved
        self._discard_running(request)
        self._waiting[request.request_id] = request
        self._waiting.move_to_end(request.request_id, last=False)
        self.num_preemptions += 1

    def finish(self, request: Request, reason: str) -> None:
        """Retire a request and release whatever resources it holds."""
        if request.status is RequestStatus.FINISHED:
            return
        was_running = request.status is RequestStatus.RUNNING
        request.status = RequestStatus.FINISHED
        request.finish_reason = reason
        request.finish_time = time.monotonic()
        if request.slot is not None:
            self._free_slots.append(request.slot)
            request.slot = None
        if was_running:
            self._discard_running(request)
            self._settle_pending_owner(request)
            # A shared prefix survives as long as another live request references it.
            self._prefix_cache.release(request.prompt_token_ids)
        else:
            # A queued request registered no blocks; releasing would decrement
            # a different request's references when the prompts match.
            self._waiting.pop(request.request_id, None)
        if self._requests.get(request.request_id) is request:
            self._requests.pop(request.request_id)

    def _discard_running(self, request: Request) -> None:
        """Remove ``request`` from the running list by identity.

        :class:`Request` is value-equal, so ``list.remove`` could drop a
        *different* request with matching fields.
        """
        for index, candidate in enumerate(self._running):
            if candidate is request:
                del self._running[index]
                return

    @property
    def prefix_cache_hit_rate(self) -> float:
        """Fraction of queried prompt tokens served from the prefix cache."""
        return self._prefix_cache.hit_rate

    @property
    def running(self) -> list[Request]:
        """Requests in the decode batch, in admission order."""
        return list(self._running)

    @property
    def waiting(self) -> list[Request]:
        """Queued requests, in arrival order."""
        return list(self._waiting.values())

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
