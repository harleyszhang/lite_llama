"""Collective accounting: what tensor parallelism actually puts on the wire.

A tensor-parallel step is fast or slow for one reason above all others — how many
bytes it makes the ranks exchange — and that number is invisible in a profile of
kernel names. This module counts it: every collective in
:mod:`lite_llama.distributed.parallel_state` reports its op and its payload here, so
a caller can ask "how much traffic did that step cost, on which plane" and get an
answer in bytes rather than an intuition.

Core design. Recording is *windowed*, not global: nothing is counted until someone
opens a window with :func:`record_collectives`, which keeps the disabled path a
single ``if`` on a module global and makes each measurement scoped to the code the
caller cares about. Windows nest, and an event is credited to every open one, so a
per-step window inside a whole-run window needs no bookkeeping from the caller. The
ledger separates the **data plane** (activations and logits, NCCL) from the
**control plane** (plans and sampled ids, gloo), because those two are traded
against each other by design: keeping logits sharded costs a couple of scalars of
control traffic to avoid a vocabulary-sized gather.

Usage:
    with record_collectives() as ledger:
        engine.step()
    print(ledger.report())          # per-op calls and bytes, split by plane
    ledger.tally("all_gather").nbytes  # 0 — the sampler never gathers logits
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

#: Ops that carry Python objects over the gloo group instead of tensors over NCCL.
#: Named here rather than passed in at every call site: the plane is a property of
#: the collective, not of the moment it is used.
CONTROL_PLANE_OPS = frozenset({"broadcast_object"})

#: Windows currently recording, innermost last. Empty is the normal state, and the
#: only cost instrumentation then has is this tuple being falsy.
_OPEN: tuple[CollectiveLedger, ...] = ()


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


class CollectiveLedger:
    """Per-op call and byte counts for one recording window.

    Not thread-safe, and deliberately so: a rank runs its collectives from one
    thread in step order, and a lock here would show up in the measurement.
    """

    def __init__(self) -> None:
        self._tallies: dict[str, Tally] = {}

    # -- writing ----------------------------------------------------------- #
    def record(self, op: str, nbytes: int) -> None:
        """Credit one call of ``op`` carrying ``nbytes`` of payload."""
        self._tallies[op] = self._tallies.get(op, Tally()).plus(nbytes)

    # -- reading ----------------------------------------------------------- #
    def tally(self, op: str) -> Tally:
        """Counts for ``op``; an all-zero :class:`Tally` if it never ran.

        Returning zeros rather than raising is what makes *absence* assertable:
        ``ledger.tally("all_gather").nbytes == 0`` is the claim that the sampler
        keeps the logits sharded, and it should read like a claim about traffic.
        """
        return self._tallies.get(op, Tally())

    def tallies(self) -> dict[str, Tally]:
        """Every op that ran, ordered by traffic (heaviest first)."""
        return dict(sorted(self._tallies.items(), key=lambda kv: -kv[1].nbytes))

    @property
    def calls(self) -> int:
        return sum(tally.calls for tally in self._tallies.values())

    @property
    def nbytes(self) -> int:
        return sum(tally.nbytes for tally in self._tallies.values())

    def plane_bytes(self, plane: str) -> int:
        """Bytes on ``"data"`` or ``"control"``.

        Args:
            plane: ``"data"`` (tensors, NCCL) or ``"control"`` (objects, gloo).

        Raises:
            ValueError: If ``plane`` is neither.
        """
        if plane not in ("data", "control"):
            raise ValueError(f"plane must be 'data' or 'control', got {plane!r}")
        want_control = plane == "control"
        return sum(
            tally.nbytes
            for op, tally in self._tallies.items()
            if (op in CONTROL_PLANE_OPS) is want_control
        )

    def report(self) -> str:
        """A one-screen table: what ran, how often, how much, on which plane."""
        if not self._tallies:
            return "no collectives (tp_world_size == 1, or nothing ran)"
        rows = [f"{'op':<18}{'plane':<9}{'calls':>7}{'bytes':>12}{'per call':>12}"]
        for op, tally in self.tallies().items():
            plane = "control" if op in CONTROL_PLANE_OPS else "data"
            rows.append(
                f"{op:<18}{plane:<9}{tally.calls:>7}"
                f"{human_bytes(tally.nbytes):>12}{human_bytes(tally.bytes_per_call):>12}"
            )
        rows.append(
            f"{'total':<18}{'':<9}{self.calls:>7}{human_bytes(self.nbytes):>12}"
            f"   (data {human_bytes(self.plane_bytes('data'))},"
            f" control {human_bytes(self.plane_bytes('control'))})"
        )
        return "\n".join(rows)


def record_collective(op: str, nbytes: int) -> None:
    """Report one collective to every open window; a no-op when none is open.

    Called from the collectives themselves, *after* the early return for a world of
    one: a no-op collective moves no bytes, and a ledger that counted it would be
    measuring the call site instead of the wire.
    """
    for ledger in _OPEN:
        ledger.record(op, nbytes)


@contextmanager
def record_collectives() -> Iterator[CollectiveLedger]:
    """Record every collective run inside this block into a fresh ledger.

    Windows nest and an event lands in all of them, which is how a per-step ledger
    and a whole-run ledger are collected in one pass. The window closes even if the
    block raises, so a failed step cannot leave instrumentation switched on.
    """
    global _OPEN
    ledger = CollectiveLedger()
    _OPEN = (*_OPEN, ledger)
    try:
        yield ledger
    finally:
        _OPEN = tuple(open_ledger for open_ledger in _OPEN if open_ledger is not ledger)


def is_recording() -> bool:
    """Whether any window is open — what lets a caller skip work it only needs to
    measure (sizing a pickled plan, say) when nobody is counting."""
    return bool(_OPEN)


def human_bytes(nbytes: float) -> str:
    """Format a byte count the way a bandwidth budget is read: 3 significant units."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    raise AssertionError("unreachable")
