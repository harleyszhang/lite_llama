"""Assert what the collective stats claim: the bytes, not the calls.

Recording starts only inside a window, bytes accumulate per op, ops
that never ran report zero, nesting sees its own span, and a raising
block still closes its window — plus report ordering.

Usage:
    pytest tests/tools/test_collective_stats.py
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from lite_llama.distributed import parallel_state as ps
from lite_llama.engine.sampler import global_argmax, local_vocab_offset, sharded_top_p
from lite_llama.tools.observability import (
    Collective,
    CollectiveStats,
    Plane,
    Tally,
    human_bytes,
)
from tests.distributed.tp_harness import run_on_tp_ranks

BATCH = 8

#: Deliberately small so ``min(k, vocab / tp)`` is ``k`` for every vocabulary below;
#: the candidate pool is then set by ``k``, which is the property under test.
CANDIDATES = 32


# --------------------------------------------------------------------------- #
# The recording window
# --------------------------------------------------------------------------- #
def test_nothing_is_recorded_until_a_window_is_open():
    """Instrumentation that is on by default is instrumentation nobody trusts to be
    free; the disabled path must not even reach a stats."""
    assert not CollectiveStats.collecting()
    CollectiveStats.record(Collective.ALL_REDUCE, 4096)  # falls on the floor

    with CollectiveStats.collect() as stats:
        assert CollectiveStats.collecting()
    assert stats.nbytes == 0
    assert not CollectiveStats.collecting()


def test_calls_and_bytes_accumulate_per_op():
    with CollectiveStats.collect() as stats:
        CollectiveStats.record(Collective.ALL_REDUCE, 1024)
        CollectiveStats.record(Collective.ALL_REDUCE, 512)
        CollectiveStats.record(Collective.BROADCAST, 8)

    assert stats.tally(Collective.ALL_REDUCE) == Tally(calls=2, nbytes=1536)
    assert stats.tally(Collective.BROADCAST) == Tally(calls=1, nbytes=8)
    assert (stats.calls, stats.nbytes) == (3, 1544)


def test_an_op_that_never_ran_reports_zero_rather_than_raising():
    """``tally(...).nbytes == 0`` is how *absence* is asserted, so it has to be a
    number: the claim "the sampler never gathers logits" should read as traffic."""
    with CollectiveStats.collect() as stats:
        CollectiveStats.record(Collective.ALL_REDUCE, 16)

    assert stats.tally(Collective.ALL_GATHER) == Tally(calls=0, nbytes=0)


def test_the_heaviest_op_is_reported_first():
    """Ordering by traffic, not by name or by call order: the first row of a report
    should be the one worth optimising."""
    with CollectiveStats.collect() as stats:
        CollectiveStats.record(Collective.BROADCAST_OBJECT, 300)
        CollectiveStats.record(Collective.ALL_REDUCE, 40_000)
        CollectiveStats.record(Collective.BROADCAST, 4)

    assert list(stats.tallies()) == ["all_reduce", "broadcast_object", "broadcast"]


def test_nested_windows_each_see_their_own_span():
    """A per-step window inside a whole-run window, collected in one pass — which is
    how the visualisation gets both without the caller subtracting anything."""
    with CollectiveStats.collect() as run:
        CollectiveStats.record(Collective.ALL_REDUCE, 100)
        with CollectiveStats.collect() as step:
            CollectiveStats.record(Collective.ALL_REDUCE, 200)
        CollectiveStats.record(Collective.ALL_REDUCE, 400)

    assert step.nbytes == 200
    assert run.nbytes == 700


def test_a_window_closes_even_when_its_block_raises():
    """A failed step must not leave recording switched on for the rest of the process."""
    with pytest.raises(RuntimeError), CollectiveStats.collect():
        raise RuntimeError("step blew up")

    assert not CollectiveStats.collecting()


def test_planes_are_a_property_of_the_op():
    """Which plane a collective uses is decided by the collective, not by its caller,
    so the split cannot drift between call sites."""
    with CollectiveStats.collect() as stats:
        CollectiveStats.record(Collective.ALL_REDUCE, 2048)  # tensors, NCCL
        CollectiveStats.record(Collective.BROADCAST_OBJECT, 256)  # pickled plan, gloo

    assert stats.bytes_on(Plane.DATA) == 2048
    assert stats.bytes_on(Plane.CONTROL) == 256
    with pytest.raises(ValueError, match=r"not a valid Plane"):
        stats.bytes_on("nccl")


def test_concurrent_tasks_do_not_share_a_window():
    """DP replicas step concurrently, and each one's window must count only its own.

    The windows live in a ContextVar, so a task opening one cannot be seen by its
    siblings. Held as a module global instead, both windows would be open at once and
    every replica would be billed the whole grid's traffic — a plausible number, and
    wrong by exactly the DP degree.
    """

    async def step(nbytes: int) -> int:
        with CollectiveStats.collect() as stats:
            await asyncio.sleep(0)  # hand over while this window is open
            CollectiveStats.record(Collective.ALL_REDUCE, nbytes)
            await asyncio.sleep(0)
        return stats.nbytes

    async def concurrently() -> list[int]:
        return list(await asyncio.gather(step(100), step(2000)))

    assert asyncio.run(concurrently()) == [100, 2000]


def test_a_report_names_every_op_and_totals_both_planes():
    with CollectiveStats.collect() as stats:
        CollectiveStats.record(Collective.ALL_REDUCE, 1_048_576)
        CollectiveStats.record(Collective.BROADCAST_OBJECT, 512)

    report = stats.report()
    assert "all_reduce" in report and "broadcast_object" in report
    assert "1.0 MB" in report and "512 B" in report
    assert "data 1.0 MB" in report and "control 512 B" in report


def test_an_empty_window_says_so_instead_of_printing_a_header():
    assert "no collectives" in CollectiveStats().report()


@pytest.mark.parametrize(
    ("nbytes", "text"), [(0, "0 B"), (1023, "1023 B"), (1024, "1.0 KB"), (1 << 21, "2.0 MB")]
)
def test_bytes_are_formatted_the_way_a_bandwidth_budget_is_read(nbytes, text):
    assert human_bytes(nbytes) == text


# --------------------------------------------------------------------------- #
# A world of one
# --------------------------------------------------------------------------- #
def test_a_world_of_one_records_nothing_because_it_moves_nothing():
    """Single-GPU code calls the same collectives, which return early. Counting those
    would measure call sites; the count is about the wire."""
    with CollectiveStats.collect() as stats:
        ps.all_reduce(torch.ones(1024))
        ps.broadcast(torch.ones(1024))
        ps.all_gather(torch.ones(1024))
        ps.broadcast_object({"plan": [1, 2, 3]})

    assert stats.calls == 0


# --------------------------------------------------------------------------- #
# Real collectives, over gloo
# --------------------------------------------------------------------------- #
def _all_reduce_payload(rank: int) -> tuple[int, int]:
    with CollectiveStats.collect() as stats:
        ps.all_reduce(torch.ones(1024, dtype=torch.float32))
    tally = stats.tally(Collective.ALL_REDUCE)
    return tally.calls, tally.nbytes


def test_an_all_reduce_is_billed_its_tensor_on_every_rank():
    """Every rank contributes the whole tensor, so every rank's tally reads the same:
    a reduce is not a scatter, and a per-rank division here would understate it."""
    assert run_on_tp_ranks(_all_reduce_payload, tp_size=2, backend="gloo") == [
        (1, 4096),
        (1, 4096),
    ]


def _control_plane_payload(rank: int) -> tuple[int, int, int]:
    plan = {"slots": list(range(64)), "step": 7}
    with CollectiveStats.collect() as stats:
        ps.broadcast_object(plan if rank == 0 else None)
    return (
        stats.tally(Collective.BROADCAST_OBJECT).calls,
        stats.bytes_on(Plane.CONTROL),
        stats.bytes_on(Plane.DATA),
    )


def test_a_broadcast_plan_is_billed_to_the_control_plane_on_both_sides():
    """The follower is charged the same bytes as the driver, measured after delivery —
    the plan crossed the wire once, and both ends should be able to say how big it was
    without one of them reporting zero."""
    (driver, follower) = run_on_tp_ranks(_control_plane_payload, tp_size=2, backend="gloo")

    assert driver[0] == follower[0] == 1
    assert driver[1] == follower[1] > 0  # same plan, same size, both ranks
    assert driver[2] == follower[2] == 0  # nothing on the data plane


# --------------------------------------------------------------------------- #
# What vocabulary-parallel sampling buys
# --------------------------------------------------------------------------- #
def _sampling_traffic(rank: int, vocab: int, greedy: bool) -> tuple[int, int, int]:
    """Sample one step from this rank's slice; report (bytes, gather bytes, vocab)."""
    local_width = vocab // ps.get_tp_world_size()
    generator = torch.Generator().manual_seed(11)
    local_logits = torch.randn(BATCH, local_width, generator=generator)
    offset = local_vocab_offset(local_width) or 0

    with CollectiveStats.collect() as stats:
        if greedy:
            global_argmax(local_logits, offset)
        else:
            sharded_top_p(local_logits, 0.8, 0.95, offset, k=CANDIDATES)
    return stats.nbytes, stats.tally(Collective.ALL_GATHER).nbytes, vocab


def _greedy_small(rank: int) -> tuple[int, int, int]:
    return _sampling_traffic(rank, vocab=4096, greedy=True)


def _greedy_large(rank: int) -> tuple[int, int, int]:
    return _sampling_traffic(rank, vocab=32768, greedy=True)


def _nucleus_small(rank: int) -> tuple[int, int, int]:
    return _sampling_traffic(rank, vocab=4096, greedy=False)


def _nucleus_large(rank: int) -> tuple[int, int, int]:
    return _sampling_traffic(rank, vocab=32768, greedy=False)


@pytest.mark.parametrize(
    ("small", "large", "how"),
    [(_greedy_small, _greedy_large, "greedy"), (_nucleus_small, _nucleus_large, "nucleus")],
)
def test_sampling_traffic_does_not_grow_with_the_vocabulary(small, large, how):
    """The whole point of keeping logits sharded, as a byte count.

    An implementation that all-gathered the logits would pass every correctness test
    in this directory — same tokens, same log-probs — and would move eight times as
    many bytes here. Only a byte count can tell the two apart.
    """
    narrow = run_on_tp_ranks(small, tp_size=2, backend="gloo")
    wide = run_on_tp_ranks(large, tp_size=2, backend="gloo")

    for (narrow_bytes, _, narrow_vocab), (wide_bytes, _, wide_vocab) in zip(
        narrow, wide, strict=True
    ):
        assert narrow_bytes == wide_bytes, (
            f"{how} sampling moved {narrow_bytes} B over a {narrow_vocab}-token vocabulary "
            f"and {wide_bytes} B over {wide_vocab}: the traffic is following the vocabulary, "
            f"so logits are being gathered somewhere"
        )


def test_sampling_moves_orders_of_magnitude_less_than_the_logits_it_samples_from():
    """States the saving as a ratio, so the test fails if a gather creeps back in."""
    (measured, _gathered, vocab), _ = run_on_tp_ranks(_nucleus_large, tp_size=2, backend="gloo")
    gathering_a_shard = BATCH * (vocab // 2) * 4  # fp32 logits, one rank's slice

    assert measured * 100 < gathering_a_shard, (
        f"nucleus sampling moved {human_bytes(measured)} where gathering one rank's "
        f"logits would move {human_bytes(gathering_a_shard)}; the point of the "
        f"decentralised log_softmax is that the first number stays negligible"
    )
