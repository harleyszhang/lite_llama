"""Request bookkeeping and the admission policy behind continuous batching.

The one-shot path fixes its batch at ``generate()`` time: every sequence starts
together, and the batch keeps running at full width until the *longest* one
finishes. Continuous batching replaces that with a per-step decision — which
requests prefill now, which decode now, and which cache slot each one holds —
so a finished request leaves immediately and a waiting one takes its place.

This module is the host-side half of that decision and holds no tensors, which
is what makes the policy testable without a GPU or a checkpoint. The device-side
half lives in :class:`~lite_llama.executor.slot_batch.SlotBatch`.

Usage:
    sched = Scheduler(SchedulerConfig(max_num_seqs=16, max_seq_len=2048), num_slots=64)
    sched.add_request(Request(...))
    output = sched.schedule()        # -> prefill group, or the decode batch
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

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
        max_seq_len: Context window; also the per-slot cache capacity, so a
            request admitted under this bound can never be evicted mid-flight.
        max_num_seqs: Ceiling on concurrently running requests. Beyond some
            width a decode step stops getting faster per token while latency
            keeps growing, so this is a latency knob as much as a memory one.
        max_num_batched_tokens: Ceiling on the *padded* token count of one
            prefill group, ``group_size * longest_prompt``. Prefill runs on a
            rectangular grid, so mixing a 1000-token prompt with seven 20-token
            ones costs 8000 token-slots of attention to do 1140 tokens of work;
            this budget is what stops a group from growing into that.
    """

    max_seq_len: int = 2048
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 8192


@dataclass(frozen=True)
class SchedulerOutput:
    """What one engine step should run.

    Exactly one of the two lists is populated. Prefill takes priority, which
    keeps time-to-first-token low and gets new requests into the decode batch at
    the next step; the alternative — appending prefill work to a decode step —
    needs a mixed-phase attention kernel this framework does not have.
    """

    prefill: list[Request] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill and not self.decode


class Scheduler:
    """FCFS admission with fixed cache slots and no preemption.

    Each running request owns one slot worth ``max_seq_len`` tokens of cache, and
    admission refuses a request whose prompt plus generation cap would not fit in
    one. Those two rules together mean a running request can never run out of
    cache, so there is no eviction, no swap-out and no recompute path here —
    unlike a paged scheduler, which trades that simplicity for denser memory use.

    Args:
        config: Admission limits.
        num_slots: Cache slots available, from
            :attr:`~lite_llama.executor.slot_batch.SlotBatch.num_slots`. The
            effective concurrency is ``min(num_slots, config.max_num_seqs)``.
    """

    def __init__(self, config: SchedulerConfig, num_slots: int) -> None:
        if num_slots < 1:
            raise ValueError(f"need at least one cache slot, got {num_slots}")
        self.config = config
        self.num_slots = num_slots
        self.max_num_seqs = min(config.max_num_seqs, num_slots)

        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []
        # Free slots as a stack: which one a request gets is immaterial, and
        # popping the end avoids the O(n) shift a queue would pay.
        self._free_slots: list[int] = list(reversed(range(num_slots)))

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
        """Decide this step's work: a prefill group if any fits, else the decode batch."""
        prefill = self._schedule_prefill()
        if prefill:
            return SchedulerOutput(prefill=prefill)
        return SchedulerOutput(decode=list(self._running))

    def _schedule_prefill(self) -> list[Request]:
        """Admit as many queued requests as slots and the token budget allow.

        Grows the group while the padded cost stays inside
        ``max_num_batched_tokens``. The first request is admitted even when it
        alone blows the budget: refusing it would leave it at the head of a FCFS
        queue forever, blocking everything behind it.
        """
        group: list[Request] = []
        longest = 0
        capacity = self.max_num_seqs - len(self._running)

        while self._waiting and len(group) < capacity and self._free_slots:
            candidate = self._waiting[0]
            padded = max(longest, candidate.prompt_len) * (len(group) + 1)
            if group and padded > self.config.max_num_batched_tokens:
                break

            self._waiting.popleft()
            candidate.slot = self._free_slots.pop()
            candidate.status = RequestStatus.RUNNING
            self._running.append(candidate)
            group.append(candidate)
            longest = max(longest, candidate.prompt_len)

        return group

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
