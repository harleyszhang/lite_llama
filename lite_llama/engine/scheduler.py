"""Request scheduling with chunked prefill and prefix caching.

The scheduler decides per step: which requests prefill (in chunks), which decode,
and which cache slot each holds. Chunked prefill splits long prompts so decode
steps interleave with prefill chunks — preventing head-of-line blocking where a
4K-token prompt stalls all running decode requests for hundreds of ms.

Key features (v0.7):
    - Chunked prefill: long prompts split into max_chunk_size token chunks.
    - Preemption: requests can be evicted when KV pressure hits the watermark.
    - Prefix caching (hash-based): shared prompt prefixes reuse cached KV.

Usage:
    sched = Scheduler(SchedulerConfig(max_num_seqs=16, max_chunk_size=512), num_slots=64)
    sched.add_request(Request(...))
    output = sched.schedule()  # may return a chunked prefill + decode batch
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .prefix_cache import PrefixCache
from .sampler import SamplingParams


class RequestStatus(str, Enum):
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


@dataclass(frozen=True)
class SchedulerConfig:
    """Limits the admission policy enforces.

    Attributes:
        max_seq_len: Context window; also the per-slot cache capacity.
        max_num_seqs: Ceiling on concurrently running requests.
        max_num_batched_tokens: Ceiling on the padded token count of one
            prefill group.
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

    With chunked prefill both lists can be populated simultaneously:
    prefill chunks run alongside the decode batch in the same step.
    """

    prefill: list[Request] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)
    # For chunked prefill: how many tokens to process per prefilling request.
    prefill_chunk_lens: list[int] = field(default_factory=list)
    # Requests that were preempted this step (for logging/metrics).
    preempted: list[Request] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill and not self.decode


class Scheduler:
    """FCFS admission with chunked prefill and preemption support.

    v0.7 features:
      - Chunked prefill: long prompts split into chunks of max_chunk_size tokens.
        Decode steps interleave with prefill chunks, so running requests are not
        blocked by a single long prompt.
      - Preemption: when no free slots remain, the most recently admitted request
        is evicted (recompute strategy) to make room for new admissions.
      - Prefix cache readiness: Request.prefix_hash is computed for future use.

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

        # Chunked prefill state: requests mid-prefill (partial prompt processed)
        self._chunking: list[Request] = []
        # Tracks how many prompt tokens have been processed for each chunking request
        self._chunk_progress: dict[str, int] = {}

        # Prefix cache: reuse KV of shared prompt prefixes (vLLM-style block hashing).
        # ``None`` when disabled so the hot path pays nothing.
        self._prefix_cache: PrefixCache | None = (
            PrefixCache(block_size=16) if config.enable_prefix_cache else None
        )

        # Preemption counter for metrics
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

        With chunked prefill enabled, a long prompt is split into chunks. Each
        step processes at most max_chunk_size tokens of prefill, interleaved with
        the full decode batch. This prevents head-of-line blocking.
        """
        # Step 1: Try to admit new requests (may preempt under oversubscription)
        prefill_group, preempted = self._schedule_prefill()

        # Step 2: Compute chunk lengths for prefilling requests
        chunk_lens = []
        for req in prefill_group:
            remaining = req.prompt_len - self._chunk_progress.get(req.request_id, 0)
            chunk = min(remaining, self.config.max_chunk_size) if self.config.max_chunk_size > 0 else remaining
            chunk_lens.append(chunk)

        # Step 3: The decode batch is always the full set of running requests
        # that have completed their prefill.
        decode_batch = [r for r in self._running if r not in prefill_group and r not in self._chunking]

        return SchedulerOutput(
            prefill=prefill_group,
            decode=decode_batch,
            prefill_chunk_lens=chunk_lens,
            preempted=preempted,
        )

    def advance_chunks(self, prefill_group: list[Request], chunk_lens: list[int]) -> None:
        """Call after a step to advance chunk progress for prefilling requests.

        Requests whose prefill is complete move to the running (decode) state.
        """
        for req, chunk_len in zip(prefill_group, chunk_lens):
            progress = self._chunk_progress.get(req.request_id, 0) + chunk_len
            self._chunk_progress[req.request_id] = progress
            if progress >= req.prompt_len:
                # Prefill complete — move to decode batch
                self._chunk_progress.pop(req.request_id, None)
                if req in self._chunking:
                    self._chunking.remove(req)

    def _schedule_prefill(self) -> tuple[list[Request], list[Request]]:
        """Admit queued requests, with chunked prefill and optional preemption.

        When ``max_chunk_size > 0``, large prompts are tracked in ``_chunking``
        and processed incrementally. Under oversubscription (preemption enabled),
        a full slot pool is freed by evicting the youngest running request that
        has already produced a token (recompute strategy); the just-recomputed
        request is protected until it decodes once, which bounds the cycle and
        prevents two requests from preempting each other forever.

        Returns:
            ``(prefill_group, preempted)`` for this step.
        """
        group: list[Request] = []
        preempted: list[Request] = []
        longest = 0
        capacity = self.max_num_seqs - len(self._running)

        # First: continue any in-progress chunked prefills
        for req in list(self._chunking):
            group.append(req)
            remaining = req.prompt_len - self._chunk_progress.get(req.request_id, 0)
            longest = max(longest, min(remaining, self.config.max_chunk_size or remaining))

        # Then: admit new waiting requests
        while self._waiting and len(group) < capacity + len(self._chunking):
            # Fix the request we intend to admit *before* any preemption, so the
            # victim (re-queued at the front for recompute) cannot jump ahead of
            # it and starve it.
            candidate = self._waiting[0]

            if not self._free_slots:
                # Oversubscription: free a slot by preempting the youngest
                # *decoding* victim that has produced >=1 token (the progress
                # quantum). A request still prefilling or freshly recomputed is
                # never chosen, so the cycle always makes forward progress.
                if not self.config.enable_preemption:
                    break
                victim = self._pick_preemption_victim(group)
                if victim is None:
                    break
                self._preempt(victim)
                preempted.append(victim)

            padded = max(longest, candidate.prompt_len) * (len(group) + 1)
            if group and padded > self.config.max_num_batched_tokens:
                break

            self._waiting.remove(candidate)
            candidate.slot = self._free_slots.pop()
            candidate.status = RequestStatus.RUNNING
            self._running.append(candidate)
            group.append(candidate)

            # Prefix cache: reuse KV of any leading blocks already cached, then
            # register this prompt so later requests with the same prefix hit it.
            cached = 0
            if self._prefix_cache is not None:
                cached = self._prefix_cache.query(candidate.prompt_token_ids)
                # Never skip the whole prompt: at least one token must be
                # prefilled to produce the first logits, exactly as vLLM keeps
                # the last block uncached.
                cached = min(cached, candidate.prompt_len - 1)
                candidate.num_cached_tokens = cached
                self._prefix_cache.register(candidate.prompt_token_ids)
            # Prefill starts past the cached prefix.
            self._chunk_progress[candidate.request_id] = cached

            # Track chunked prefill if the *uncached* remainder exceeds a chunk.
            chunk_size = self.config.max_chunk_size
            if chunk_size > 0 and (candidate.prompt_len - cached) > chunk_size:
                self._chunking.append(candidate)

            longest = max(longest, candidate.prompt_len - cached)

        return group, preempted

    def _pick_preemption_victim(self, group: list[Request]) -> Request | None:
        """Youngest running request eligible for preemption, or None.

        Eligible = still decoding (not mid-prefill), not part of this step's
        prefill group, and past the progress quantum (>=1 output token). The
        quantum is what stops a just-recomputed request from being evicted again
        before it makes progress, so the recompute cycle cannot livelock.
        """
        for request in reversed(self._running):
            if request in self._chunking or request in group:
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
        request.output_token_ids.clear()
        # Recompute restarts prefill from scratch (minus any cached prefix).
        self._chunk_progress.pop(request.request_id, None)
        request.num_cached_tokens = 0
        self._running = [r for r in self._running if r is not request]
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
        # Identity comparison: two distinct requests may carry the same id if a
        # caller reuses one, and removing the wrong object would leak a slot.
        self._running = [r for r in self._running if r is not request]
        self._chunk_progress.pop(request.request_id, None)
        # Drop this request's hold on its cached prefix blocks. A shared prefix
        # survives as long as any other live request still references it.
        if self._prefix_cache is not None:
            self._prefix_cache.release(request.prompt_token_ids)

    @property
    def prefix_cache_hit_rate(self) -> float:
        """Fraction of queried prompt tokens served from the prefix cache."""
        return self._prefix_cache.hit_rate if self._prefix_cache is not None else 0.0

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
