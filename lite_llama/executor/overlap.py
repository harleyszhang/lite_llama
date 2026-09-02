"""Cross-stream overlap: the L1 policy, its stream pool and its timeline.

:class:`OverlapPolicy` (read from ``LITE_LLAMA_OVERLAP``) decides whether
input uploads run on a copy stream; :class:`StreamPool` stages the async
uploads — and, in the opposite direction, reads results back into pinned
host memory — while :class:`Timeline` records regions as timeline evidence.

The module also carries the O3.2 two-batch-overlap *executor primitive*
(:class:`YieldOperation` + :func:`execute_overlapped`): the stage/yield
interleaving discipline sglang's ``batch_overlap`` runs its micro-batches on,
kept here with its own tests until the model's forward is expressed as
operation streams and can ride it.

Usage:
    policy = OverlapPolicy.from_env()
    timeline = Timeline.from_env("cuda"); print(timeline.summary())
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from ..utils.logger import get_logger

_log = get_logger(__name__)

#: Environment variable switching every overlap site at once (``0`` disables).
OVERLAP_ENV = "LITE_LLAMA_OVERLAP"

#: Environment variable turning on per-region CUDA-event recording.
TIMELINE_ENV = "LITE_LLAMA_OVERLAP_TIMELINE"


@dataclass(frozen=True)
class OverlapPolicy:
    """One flag answering for every cross-stream overlap site.

    ``enabled=False`` collapses every code path back to the inline, single-stream
    behaviour, which is also the answer when there is no CUDA device at all —
    the pool is simply never built.
    """

    enabled: bool = True

    @classmethod
    def from_env(cls) -> OverlapPolicy:
        """Read ``LITE_LLAMA_OVERLAP``; anything but ``0``/``false``/``off`` means on."""
        raw = os.environ.get(OVERLAP_ENV, "1").strip().lower()
        return cls(enabled=raw not in ("0", "false", "off"))


class StreamPool:
    """The copy stream and the pinned staging rings overlap traffic rides on.

    Staging buffers are pinned host memory reused through freelists. A buffer
    goes back into rotation only once the event recorded after its last copy
    reports completion; if nothing free fits, a fresh buffer is allocated — the
    ring grows to the high-water mark of passes in flight (two for the
    continuous engine) and stays there. A busy buffer is never force-reused:
    overwriting bytes a copy engine is still reading is the classic
    pinned-memory race.

    Two rings share one copy stream and one discipline, one per direction:

    * **Upload** (:meth:`upload_async`): host values go out on H2D, and the
      compute stream waits on the event before its kernels read them.
    * **Readback** (:meth:`readback_async`): device results come back on D2H,
      and the *host* waits on the event — one step later, when it wants the
      values — before reading the pinned buffer. The copy itself is ordered
      after the compute stream's queue at issue time, so it observes whatever
      kernels produced the tensor without any host-side sync. The caller that
      holds the returned view must finish reading it before the pool is asked
      for the next readback; the engine's launch/harvest split guarantees that
      by harvesting step N-1's buffer before launching step N's readback.

    Args:
        device: Device string (``"cuda"`` / ``"cuda:1"``) the streams belong to.
        policy: The shared switch; a disabled policy makes :meth:`upload_async`
            fall back to a plain blocking upload and :meth:`readback_async` to
            a plain blocking ``.cpu()``, both returning no event, so call sites
            read the same either way.
        timeline: Where the copy regions are recorded; disabled timelines make
            the recording a no-op.
    """

    def __init__(
        self, device: str, policy: OverlapPolicy, timeline: Timeline | None = None
    ) -> None:
        self._device = torch.device(device)
        self._policy = policy
        self._timeline = timeline if timeline is not None else Timeline()
        self._copy_stream: torch.cuda.Stream | None = None
        # (flat pinned buffer, event recording its copy's completion); the event
        # is None only for a buffer that has never carried a copy, hence free.
        # One deque per direction: a token upload and a token readback would
        # otherwise contend for the same buffers at crossed lifetimes.
        self._staging: deque[tuple[torch.Tensor, torch.cuda.Event | None]] = deque()
        self._spill: deque[tuple[torch.Tensor, torch.cuda.Event | None]] = deque()

    @property
    def copy_stream(self) -> torch.cuda.Stream:
        """The side stream copies are issued on, created on first use."""
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self._device)
        return self._copy_stream

    def upload_async(
        self, values: Any, *, dtype: torch.dtype, label: str = "upload"
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """Upload host values and return ``(device tensor, completion event)``.

        With the policy on, the values land in a pinned staging buffer and the
        H2D copy is issued on the copy stream — this call returns without
        waiting for it. With the policy off it is exactly the legacy inline
        upload and the event is ``None``, which :meth:`consume` treats as a
        no-op. Either way the returned tensor keeps the shape ``values`` had.

        The device tensor is allocated on the copy stream, so a recycled block
        can only come from a copy that already finished ahead of this one; the
        compute-stream side of the same race is covered by :meth:`consume`'s
        ``record_stream``. ``label`` names the timeline region the copy is
        recorded under.
        """
        host = torch.as_tensor(values, dtype=dtype)
        flat = host.reshape(-1)
        if not self._policy.enabled:
            return host.to(self._device), None

        staging = self._acquire(self._staging, dtype, flat.numel())
        staging[: flat.numel()].copy_(flat)
        with torch.cuda.stream(self.copy_stream), self._timeline.region(label, "copy"):
            device_tensor = staging[: flat.numel()].to(self._device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.copy_stream)
        device_tensor = device_tensor.view(host.shape)
        # The fresh event replaces the buffer's old completion marker: the
        # buffer is busy again until *this* copy lands.
        self._staging.append((staging, event))
        return device_tensor, event

    def readback_async(
        self, device_tensor: torch.Tensor, *, label: str = "readback"
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """Copy device results into pinned host memory; ``(host view, event)``.

        The D2H copy rides the copy stream, ordered after the compute stream's
        queue *at the moment of issue* — which is exactly the kernels that
        produced ``device_tensor`` when the caller asks right after a pass. The
        host does not wait: it reads the returned view later, after
        ``event.synchronize()``, by which point the copy has long since landed
        under the next pass's compute. With the policy off this is a plain
        blocking ``.cpu()`` and the event is ``None`` — a no-CUDA pool returns
        the tensor itself.

        The returned view aliases a ring buffer. Whoever holds it must be done
        reading before the pool issues a readback that recycles the buffer;
        the ring's event check only covers the copy engine, not the host's own
        reads. The engine's pipeline (harvest N-1 strictly before launch N's
        readback) is that guarantee.
        """
        if not self._policy.enabled:
            return device_tensor.cpu(), None
        flat = device_tensor.reshape(-1)
        pinned = self._acquire(self._spill, flat.dtype, flat.numel())
        # Grabbed before entering the copy-stream context: inside it, the
        # "current" stream is the copy stream, and waiting on ourselves would
        # order nothing.
        compute = torch.cuda.current_stream(self._device)
        with torch.cuda.stream(self.copy_stream), self._timeline.region(label, "copy"):
            self.copy_stream.wait_stream(compute)
            pinned[: flat.numel()].copy_(flat, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.copy_stream)
            # The source block is read on the copy stream while the caller
            # may already have dropped its last reference: without this, the
            # caching allocator can hand the block to the next allocation and
            # the copy reads whatever that wrote — the mirror image of the
            # upload path's record_stream in consume().
            flat.record_stream(self.copy_stream)
        self._spill.append((pinned, event))
        return pinned[: flat.numel()].view(device_tensor.shape), event

    def consume(self, event: torch.cuda.Event | None, *tensors: torch.Tensor | None) -> None:
        """Make the current stream wait for an :meth:`upload_async` event.

        The wait is stream-ordered, not host-side: the CPU keeps running and the
        kernels launched after this call simply cannot start before the copy
        lands. ``None`` (overlap off) is a no-op.

        The uploaded tensors should be passed along: they are
        ``record_stream``-ed on this stream, because their blocks belong to the
        copy stream's allocator pool — without the mark, a tensor freed after
        its pass could be recycled into the next upload while this stream's
        kernels are still reading it.
        """
        if event is None:
            return
        stream = torch.cuda.current_stream(self._device)
        stream.wait_event(event)
        for tensor in tensors:
            if tensor is not None:
                tensor.record_stream(stream)

    def _acquire(
        self, ring: deque[tuple[torch.Tensor, torch.cuda.Event | None]], dtype: torch.dtype, numel: int
    ) -> torch.Tensor:
        """Find a free buffer of ``ring`` holding ``numel`` elements of ``dtype``.

        "Free" means the event recorded after its last copy has completed.
        Buffers still busy stay in the ring; a fresh allocation is cheaper than
        a host-side sync on a copy that has almost certainly finished by the
        time the next pass asks. Buffers of the wrong dtype or too small are
        retired rather than kept around.
        """
        for _ in range(len(ring)):
            buffer, event = ring.popleft()
            if event is not None and not event.query():
                ring.append((buffer, event))  # still in flight
                continue
            if buffer.dtype == dtype and buffer.numel() >= numel:
                return buffer
            # Completed but never useful for this request: drop it.
        return torch.empty(numel, dtype=dtype, pin_memory=True)

    def pending(self) -> int:
        """How many copies are in flight right now, either direction (test hook)."""
        return sum(
            1
            for ring in (self._staging, self._spill)
            for _, event in ring
            if event is not None and not event.query()
        )


@dataclass
class RegionRecord:
    """One resolved timeline region: a named span of work on one stream.

    ``start_ms``/``end_ms`` are on the timeline's shared device clock, so a copy
    region and a compute region compare directly — overlap is literally
    ``a.start_ms < b.end_ms and b.start_ms < a.end_ms``.
    """

    name: str
    stream: str
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class Timeline:
    """CUDA-event recorder for overlap evidence; costs one flag check when off.

    Regions are recorded with events rather than host timestamps because the
    question a timeline answers — "did the copy region and the compute region
    actually overlap on the device?" — is a device-side fact. Host clocks can
    only say when work was *launched*. All events sit on one clock by measuring
    them against an epoch event recorded when the first region opens; event
    timestamps share the device's globaltimer, so cross-stream comparison is
    valid.

    Args:
        enabled: Typically from ``LITE_LLAMA_OVERLAP_TIMELINE``; when ``False``
            :meth:`region` is a near-free context manager.
        device: Device the events are recorded against.
    """

    enabled: bool = False
    device: str = "cuda"
    _epoch: torch.cuda.Event | None = field(default=None, repr=False)
    _pending: list[tuple[str, str, torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list, repr=False
    )

    @classmethod
    def from_env(cls, device: str = "cuda") -> Timeline:
        raw = os.environ.get(TIMELINE_ENV, "0").strip().lower()
        return cls(enabled=raw in ("1", "true", "on"), device=device)

    @contextmanager
    def region(self, name: str, stream: str = "compute") -> Iterator[None]:
        """Record a named region on the current stream; near-free when disabled."""
        if not self.enabled:
            yield
            return
        if self._epoch is None:
            self._epoch = torch.cuda.Event(enable_timing=True)
            self._epoch.record(torch.cuda.current_stream(self.device))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            end.record(torch.cuda.current_stream(self.device))
            self._pending.append((name, stream, start, end))

    def collect(self) -> list[RegionRecord]:
        """Synchronise and resolve every recorded region onto the shared clock."""
        if self._epoch is None:
            return []
        self._epoch.synchronize()
        records = []
        for name, stream, start, end in self._pending:
            end.synchronize()
            records.append(
                RegionRecord(
                    name,
                    stream,
                    self._epoch.elapsed_time(start),
                    self._epoch.elapsed_time(end),
                )
            )
        return records

    def summary(self) -> str:
        """Human-readable region table, for benchmark logs and test output."""
        return "\n".join(
            f"{r.stream:>8}  {r.name:<24} [{r.start_ms:9.3f}, {r.end_ms:9.3f}] ms"
            for r in self.collect()
        )


# --------------------------------------------------------------------------- #
# O3.2 two-batch-overlap executor primitive
# --------------------------------------------------------------------------- #
class YieldOperation:
    """Stage boundary in an operation stream: where the executor may switch.

    Ops between two yields form one stage — indivisible, run to completion
    before the other micro-batch resumes. Sglang's ``batch_overlap`` calls the
    same concept a yield operation; the yields are where micro-batch A's GEMMs
    overlap micro-batch B sitting inside a communication op.
    """


def _stages_of(
    ops: Sequence[Callable[[], Any] | YieldOperation],
) -> list[list[Callable[[], Any]]]:
    """Cut an operation stream into stages at yield boundaries."""
    stages: list[list[Callable[[], Any]]] = [[]]
    for op in ops:
        if isinstance(op, YieldOperation):
            stages.append([])
        else:
            stages[-1].append(op)
    return [stage for stage in stages if stage]


def execute_overlapped(
    ops_a: Sequence[Callable[[], Any] | YieldOperation],
    ops_b: Sequence[Callable[[], Any] | YieldOperation],
    *,
    delta_stages: int = 0,
) -> None:
    """Run two micro-batch operation streams interleaved, stage by stage (O3.2).

    The TBO pattern: while micro-batch A's ops run, micro-batch B sits at its
    last yield — ideally inside an op the copy engines or NICs handle without
    SMs — then they swap. ``delta_stages`` is how far the lead stream runs
    before the trailing one starts (and symmetrically at the tail); ``0``
    strictly alternates, A first.

    Only the *executor* half of O3.2 lives here, deliberately consumer-free:
    the interleaving discipline and its tests are fixed now so the day the
    model's forward is expressed as operation streams reuses them verbatim.

    Args:
        ops_a: Lead stream — zero-arg callables separated by
            :class:`YieldOperation` markers.
        ops_b: Trailing stream, same shape.
        delta_stages: Stages A runs ahead of B before alternation starts.

    Raises:
        ValueError: ``delta_stages`` is negative.
    """
    if delta_stages < 0:
        raise ValueError(f"delta_stages must be >= 0, got {delta_stages}")
    stages_a = _stages_of(ops_a)
    stages_b = _stages_of(ops_b)

    def _drain(stages: list[list[Callable[[], Any]]], index: int, count: int) -> int:
        for stage in stages[index : index + count]:
            for op in stage:
                op()
        return min(index + count, len(stages))

    # The lead runs ahead, then alternation (A's stage, then B's) while both
    # have stages left, then whichever stream still owes stages drains.
    ia = _drain(stages_a, 0, delta_stages)
    ib = 0
    while ia < len(stages_a) and ib < len(stages_b):
        ia = _drain(stages_a, ia, 1)
        ib = _drain(stages_b, ib, 1)
    _drain(stages_a, ia, len(stages_a))
    _drain(stages_b, ib, len(stages_b))
