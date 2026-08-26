"""Collective accounting: what tensor parallelism actually puts on the wire.

A tensor-parallel step is fast or slow for one reason above all others — how many
bytes it makes the ranks exchange — and that number is invisible in a profile of
kernel names. This module counts it: every collective in
:mod:`lite_llama.distributed.parallel_state` reports its op and its payload here, so
a caller can ask "how much traffic did that step cost, on which plane" and get an
answer in bytes rather than an intuition.

Core design. Recording is *windowed*, not global: nothing is counted until someone
opens a window with :meth:`CollectiveStats.collect`, which keeps the disabled path a
single falsy check and makes each measurement scoped to the code the caller cares
about. The open windows live in a :class:`~contextvars.ContextVar`, so a window
belongs to the thread and asyncio task that opened it — DP replicas step
concurrently, and a per-step measurement that also counted a sibling replica's
traffic would be worse than no measurement at all. Windows nest, and an event is
credited to every open one, so a per-step window inside a whole-run window needs no
bookkeeping from the caller.

Ops and planes are enums rather than strings. A mistyped op name would otherwise open
a tally of its own and leave the intended one reporting zero, which reads exactly
like the traffic being absent — the one answer this module exists to give. The two
planes are counted apart because they are traded against each other by design:
keeping logits sharded costs a couple of scalars of **control** traffic (plans and
sampled ids, gloo) to avoid a vocabulary-sized gather on the **data** plane
(activations and logits, NCCL).

Usage:
    with CollectiveStats.collect() as stats:
        engine.step()
    print(stats.report())                       # per-op calls and bytes, by plane
    stats.tally(Collective.ALL_GATHER).nbytes    # 0 — the sampler never gathers
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum


class Plane(StrEnum):
    """Which of the two wires a collective travels on.

    Budgeted apart because they cost differently and are spent against each other:
    control traffic is a few hundred pickled bytes over gloo that never touch device
    memory or queue behind the NCCL stream, data traffic is activation-sized tensors
    that do both.
    """

    DATA = "data"
    CONTROL = "control"


class Collective(StrEnum):
    """The collectives :mod:`lite_llama.distributed.parallel_state` can run.

    A closed set, which is the point: tallies are keyed by this enum, so an op that
    is not listed here cannot be recorded at all, and a misspelling fails at the call
    site instead of silently splitting one op's traffic into two rows.
    """

    ALL_REDUCE = "all_reduce"
    ALL_REDUCE_MAX = "all_reduce_max"
    ALL_REDUCE_MIN = "all_reduce_min"
    ALL_GATHER = "all_gather"
    BROADCAST = "broadcast"
    BROADCAST_OBJECT = "broadcast_object"

    @property
    def plane(self) -> Plane:
        """The wire this op uses.

        Answered by the op rather than passed in at each call site: the plane follows
        from what the collective carries, and asking the caller would let two uses of
        the same collective disagree about it.
        """
        return Plane.CONTROL if self is Collective.BROADCAST_OBJECT else Plane.DATA


@dataclass(frozen=True)
class Tally:
    """How much one collective was used inside a window."""

    calls: int = 0
    nbytes: int = 0

    @property
    def bytes_per_call(self) -> float:
        return self.nbytes / self.calls if self.calls else 0.0

    def plus(self, nbytes: int) -> Tally:
        return Tally(self.calls + 1, self.nbytes + nbytes)


class CollectiveStats:
    """Per-op call and byte counts for one recording window.

    Instances are not thread-safe, and deliberately so: a rank runs its collectives
    from one thread in step order, and a lock here would show up in the measurement.
    Concurrent measurements stay apart by not sharing a window at all — see
    :attr:`_open`.
    """

    #: Windows currently recording, innermost last. Empty is the normal state, and
    #: the only cost instrumentation then has is this tuple being falsy. A ContextVar
    #: rather than a plain module global so the set is per-thread and per-asyncio
    #: task: see the module docstring on concurrent DP replicas.
    _open: ContextVar[tuple[CollectiveStats, ...]] = ContextVar(
        "lite_llama_open_collective_windows", default=()
    )

    def __init__(self) -> None:
        self._tallies: dict[Collective, Tally] = {}

    # -- recording ---------------------------------------------------------- #
    @classmethod
    @contextmanager
    def collect(cls) -> Iterator[CollectiveStats]:
        """Record every collective run inside this block into a fresh window.

        Windows nest and an event lands in all of them, which is how a per-step tally
        and a whole-run tally are collected in one pass. Closing restores the exact
        previous set from the token, so the window goes away even if the block
        raises: a failed step cannot leave instrumentation switched on.
        """
        stats = cls()
        token = cls._open.set((*cls._open.get(), stats))
        try:
            yield stats
        finally:
            cls._open.reset(token)

    @classmethod
    def record(cls, op: Collective, nbytes: int) -> None:
        """Report one collective to every open window; a no-op when none is open.

        Called from the collectives themselves, *after* the early return for a world
        of one: a no-op collective moves no bytes, and counting it would measure the
        call site instead of the wire.
        """
        for stats in cls._open.get():
            stats._credit(op, nbytes)

    @classmethod
    def collecting(cls) -> bool:
        """Whether any window is open — what lets a caller skip work it only needs
        in order to measure (sizing a pickled plan, say) when nobody is counting."""
        return bool(cls._open.get())

    def _credit(self, op: Collective, nbytes: int) -> None:
        self._tallies[op] = self._tallies.get(op, Tally()).plus(nbytes)

    # -- reading ------------------------------------------------------------ #
    def tally(self, op: Collective) -> Tally:
        """Counts for ``op``; an all-zero :class:`Tally` if it never ran.

        Returning zeros rather than raising is what makes *absence* assertable:
        ``stats.tally(Collective.ALL_GATHER).nbytes == 0`` is the claim that the
        sampler keeps the logits sharded, and it should read like a claim about
        traffic.
        """
        return self._tallies.get(op, Tally())

    def tallies(self) -> dict[Collective, Tally]:
        """Every op that ran, ordered by traffic (heaviest first)."""
        return dict(sorted(self._tallies.items(), key=lambda kv: -kv[1].nbytes))

    @property
    def calls(self) -> int:
        return sum(tally.calls for tally in self._tallies.values())

    @property
    def nbytes(self) -> int:
        return sum(tally.nbytes for tally in self._tallies.values())

    def bytes_on(self, plane: Plane) -> int:
        """Bytes that travelled on ``plane``.

        Takes the enum, so there is no plane name to spell wrong and no hand-written
        validation to keep in step with it; :class:`Plane` still accepts a plain
        string for a caller that read one out of a config.
        """
        wanted = Plane(plane)
        return sum(tally.nbytes for op, tally in self._tallies.items() if op.plane is wanted)

    def report(self) -> str:
        """A one-screen table: what ran, how often, how much, on which plane."""
        if not self._tallies:
            return "no collectives (tp_world_size == 1, or nothing ran)"
        rows = [f"{'op':<18}{'plane':<9}{'calls':>7}{'bytes':>12}{'per call':>12}"]
        for op, tally in self.tallies().items():
            rows.append(
                f"{op:<18}{op.plane:<9}{tally.calls:>7}"
                f"{human_bytes(tally.nbytes):>12}{human_bytes(tally.bytes_per_call):>12}"
            )
        rows.append(
            f"{'total':<18}{'':<9}{self.calls:>7}{human_bytes(self.nbytes):>12}"
            f"   (data {human_bytes(self.bytes_on(Plane.DATA))},"
            f" control {human_bytes(self.bytes_on(Plane.CONTROL))})"
        )
        return "\n".join(rows)


def human_bytes(nbytes: float) -> str:
    """Format a byte count the way a bandwidth budget is read: 3 significant units."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    raise AssertionError("unreachable")
