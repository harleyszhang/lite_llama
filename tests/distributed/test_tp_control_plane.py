"""The control plane: what a driver rank tells its tensor-parallel followers.

A tensor-parallel step is decided once and run everywhere, so the decision has to
travel intact. If it does not — a slot renumbered, a plan skipped, a stop signal
one rank missed — the symptom is not a wrong answer but a hang, because the ranks
stop calling the same collectives. That makes these the tests a wrong-looking
number cannot substitute for.

They run on a real ``gloo`` process grid and no GPU at all, which is the point: the
control plane is pickled bytes over a CPU group, so it can be verified on any
machine, and it is verified against the *same* functions the executor calls rather
than a re-implementation of them. The follower loop appears here in the shape
:func:`~lite_llama.executor.executor.run_follower` uses it — receive until
``None`` — minus the model.

Usage:
    pytest tests/distributed/test_tp_control_plane.py
"""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from lite_llama.distributed.parallel_state import (
    broadcast_object_tp,
    get_dp_rank,
    get_tp_rank,
)
from lite_llama.engine.sampler import SamplingParams
from lite_llama.executor.executor import ensure_followers_alive
from lite_llama.executor.worker import ModelInput, PassKind
from tests.distributed.tp_harness import run_on_tp_ranks


def a_plan(slot: int = 0) -> ModelInput:
    """A plan whose every field is distinct, so a mangled one cannot pass."""
    return ModelInput(
        kind=PassKind.DECODE,
        slots=(slot, slot + 1),
        seq_starts=(11, 22),
        seq_lens=(12, 23),
        tokens=(31, 32),
        sampling=(SamplingParams(temperature=0.7, top_p=0.9), SamplingParams(temperature=0.0)),
        sampled=(0, 1),
        gen_counts=(3, 4),
    )


def _publish_one_plan(rank: int) -> ModelInput:
    """Rank 0 publishes; every rank reports the plan it holds afterwards."""
    return broadcast_object_tp(a_plan() if get_tp_rank() == 0 else None)


def _drain_a_stream(rank: int) -> int:
    """Rank 0 sends three plans then the stop signal; followers count what arrives.

    The follower branch is ``run_follower``'s loop with the model taken out, so a
    change to the stop protocol breaks this test rather than deadlocking a GPU run.
    """
    if get_tp_rank() == 0:
        for slot in range(3):
            broadcast_object_tp(a_plan(slot=slot))
        broadcast_object_tp(None)
        return 3
    seen = 0
    while broadcast_object_tp() is not None:
        seen += 1
    return seen


def _publish_per_replica(rank: int) -> tuple[int, ...]:
    """Each replica's rank 0 publishes a plan naming its own slots."""
    plan = a_plan(slot=10 * get_dp_rank()) if get_tp_rank() == 0 else None
    return broadcast_object_tp(plan).slots


class TestPlanBroadcast:
    """A plan must arrive field-for-field, on every rank, over the CPU group."""

    def test_every_rank_receives_the_drivers_plan(self):
        received = run_on_tp_ranks(_publish_one_plan, tp_size=2, backend="gloo")

        assert received[1] == received[0] == a_plan()

    def test_sampling_parameters_survive_the_trip(self):
        """The per-request knobs travel as objects, not as floats on a wire.

        The mirror-process scheme this replaces packed them into five doubles and
        unpacked them by index on the other side — two encodings to keep in step,
        and a divergence there is what made tensor-parallel chat deadlock.
        """
        received = run_on_tp_ranks(_publish_one_plan, tp_size=2, backend="gloo")

        assert received[1].sampling == (
            SamplingParams(temperature=0.7, top_p=0.9),
            SamplingParams(temperature=0.0),
        )

    def test_a_follower_runs_until_the_stop_signal(self):
        assert run_on_tp_ranks(_drain_a_stream, tp_size=2, backend="gloo") == [3, 3]

    def test_a_replicas_plan_stays_inside_it(self):
        """Under DP x TP the broadcast is per replica, not world-wide.

        Ranks are laid out ``dp_rank * tp_size + tp_rank``, so replica 1's driver
        is global rank 2 — a broadcast rooted at global rank 0 would silently feed
        replica 1 the wrong batch.
        """
        slots = run_on_tp_ranks(_publish_per_replica, tp_size=2, dp_size=2, backend="gloo")

        assert slots == [(0, 1), (0, 1), (10, 11), (10, 11)]

    def test_a_world_of_one_needs_no_group(self):
        """Single-GPU code calls the same function; it must not touch torch.distributed."""
        plan = a_plan()

        assert broadcast_object_tp(plan) is plan


def _exit_now() -> None:
    """Body of a follower that is already gone by the time the driver looks."""


def _stay_alive() -> None:
    """Body of a follower that is still around; the test terminates it."""
    time.sleep(30)


class TestFollowerLiveness:
    """A dead follower must be reported, not waited for."""

    def test_a_live_follower_passes(self):
        process = mp.get_context("spawn").Process(target=_stay_alive, daemon=True)
        process.start()
        try:
            ensure_followers_alive([process])
        finally:
            process.terminate()
            process.join(timeout=10)

    def test_a_dead_follower_is_named_by_rank(self):
        process = mp.get_context("spawn").Process(target=_exit_now, daemon=True)
        process.start()
        process.join(timeout=30)

        with pytest.raises(RuntimeError, match="rank 1"):
            ensure_followers_alive([process])
