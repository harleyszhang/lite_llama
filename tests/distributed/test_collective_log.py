"""Assert what the collective ledger claims: the bytes, not the calls.

The ledger exists to make one design decision measurable — vocabulary-parallel
sampling exchanges a couple of scalars per row instead of gathering logits — so these
tests are about *payloads*. They come in two halves. The bookkeeping half runs on
plain CPU with no process group at all, because windowing, nesting and plane
attribution are ordinary logic and deserve millisecond tests. The other half runs the
real collectives over a two-rank **gloo** grid: bytes are what the wire sees, and only
a real ``dist`` call can say what that was.

The sharp assertion is the last one: sampling traffic must not change when the
vocabulary grows eightfold. That is the difference between this sampler and one that
all-gathers logits, and it is invisible in output text — a gathering sampler produces
exactly the same tokens, just slower per step as the vocabulary grows.

Usage:
    pytest tests/distributed/test_collective_log.py      # no GPU needed
"""

from __future__ import annotations

import pytest
import torch

from lite_llama.distributed import parallel_state as ps
from lite_llama.distributed.collective_log import (
    CollectiveLedger,
    Tally,
    human_bytes,
    is_recording,
    record_collective,
    record_collectives,
)
from lite_llama.engine.sampler import global_argmax, local_vocab_offset, sharded_top_p
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
    free; the disabled path must not even reach a ledger."""
    assert not is_recording()
    record_collective("all_reduce", 4096)  # falls on the floor

    with record_collectives() as ledger:
        assert is_recording()
    assert ledger.nbytes == 0
    assert not is_recording()


def test_calls_and_bytes_accumulate_per_op():
    with record_collectives() as ledger:
        record_collective("all_reduce", 1024)
        record_collective("all_reduce", 512)
        record_collective("broadcast", 8)

    assert ledger.tally("all_reduce") == Tally(calls=2, nbytes=1536)
    assert ledger.tally("broadcast") == Tally(calls=1, nbytes=8)
    assert (ledger.calls, ledger.nbytes) == (3, 1544)


def test_an_op_that_never_ran_reports_zero_rather_than_raising():
    """``tally(...).nbytes == 0`` is how *absence* is asserted, so it has to be a
    number: the claim "the sampler never gathers logits" should read as traffic."""
    with record_collectives() as ledger:
        record_collective("all_reduce", 16)

    assert ledger.tally("all_gather") == Tally(calls=0, nbytes=0)


def test_the_heaviest_op_is_reported_first():
    """Ordering by traffic, not by name or by call order: the first row of a report
    should be the one worth optimising."""
    with record_collectives() as ledger:
        record_collective("broadcast_object", 300)
        record_collective("all_reduce", 40_000)
        record_collective("broadcast", 4)

    assert list(ledger.tallies()) == ["all_reduce", "broadcast_object", "broadcast"]


def test_nested_windows_each_see_their_own_span():
    """A per-step ledger inside a whole-run ledger, collected in one pass — which is
    how the visualisation gets both without the caller subtracting anything."""
    with record_collectives() as run:
        record_collective("all_reduce", 100)
        with record_collectives() as step:
            record_collective("all_reduce", 200)
        record_collective("all_reduce", 400)

    assert step.nbytes == 200
    assert run.nbytes == 700


def test_a_window_closes_even_when_its_block_raises():
    """A failed step must not leave recording switched on for the rest of the process."""
    with pytest.raises(RuntimeError), record_collectives():
        raise RuntimeError("step blew up")

    assert not is_recording()


def test_planes_are_a_property_of_the_op():
    """Which plane a collective uses is decided by the collective, not by its caller,
    so the split cannot drift between call sites."""
    with record_collectives() as ledger:
        record_collective("all_reduce", 2048)  # tensors, NCCL
        record_collective("broadcast_object", 256)  # pickled plan, gloo

    assert ledger.plane_bytes("data") == 2048
    assert ledger.plane_bytes("control") == 256
    with pytest.raises(ValueError, match=r"data.*control"):
        ledger.plane_bytes("nccl")


def test_a_report_names_every_op_and_totals_both_planes():
    with record_collectives() as ledger:
        record_collective("all_reduce", 1_048_576)
        record_collective("broadcast_object", 512)

    report = ledger.report()
    assert "all_reduce" in report and "broadcast_object" in report
    assert "1.0 MB" in report and "512 B" in report
    assert "data 1.0 MB" in report and "control 512 B" in report


def test_an_empty_ledger_says_so_instead_of_printing_a_header():
    assert "no collectives" in CollectiveLedger().report()


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
    would measure call sites; the ledger is about the wire."""
    with record_collectives() as ledger:
        ps.all_reduce_tp(torch.ones(1024))
        ps.broadcast_tp(torch.ones(1024))
        ps.all_gather_tp(torch.ones(1024))
        ps.broadcast_object_tp({"plan": [1, 2, 3]})

    assert ledger.calls == 0


# --------------------------------------------------------------------------- #
# Real collectives, over gloo
# --------------------------------------------------------------------------- #
def _all_reduce_payload(rank: int) -> tuple[int, int]:
    with record_collectives() as ledger:
        ps.all_reduce_tp(torch.ones(1024, dtype=torch.float32))
    tally = ledger.tally("all_reduce")
    return tally.calls, tally.nbytes


def test_an_all_reduce_is_billed_its_tensor_on_every_rank():
    """Every rank contributes the whole tensor, so every rank's ledger reads the same:
    a reduce is not a scatter, and a per-rank division here would understate it."""
    assert run_on_tp_ranks(_all_reduce_payload, tp_size=2, backend="gloo") == [
        (1, 4096),
        (1, 4096),
    ]


def _control_plane_payload(rank: int) -> tuple[int, int, int]:
    plan = {"slots": list(range(64)), "step": 7}
    with record_collectives() as ledger:
        ps.broadcast_object_tp(plan if rank == 0 else None)
    return (
        ledger.tally("broadcast_object").calls,
        ledger.plane_bytes("control"),
        ledger.plane_bytes("data"),
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

    with record_collectives() as ledger:
        if greedy:
            global_argmax(local_logits, offset)
        else:
            sharded_top_p(local_logits, 0.8, 0.95, offset, k=CANDIDATES)
    return ledger.nbytes, ledger.tally("all_gather").nbytes, vocab


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
    many bytes here. Only the ledger can tell the two apart.
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
