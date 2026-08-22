"""Tests for the DP x TP rank grid in :mod:`lite_llama.distributed.parallel_state`.

The grid is global mutable state that every layer reads at construction time, so
these tests pin down two things:

* **the coordinate arithmetic** — a global rank maps to exactly one
  ``(dp_rank, tp_rank)`` pair, and the layout keeps a replica's TP ranks contiguous.
  Getting this wrong does not crash; it silently gives two replicas the same shard of
  the weights, which is why it is asserted rather than eyeballed.
* **the world-of-one default** — every accessor and collective must be a no-op when
  neither kind of parallelism is on, because that is the path single-GPU inference
  takes and it must never branch.

Multi-rank grids are exercised through :func:`grid_coordinates`, never through
``init_parallel``: with ``tp_size > 1`` that function blocks in an NCCL rendezvous
waiting for ranks a single-process test will never start. ``init_parallel`` itself is
therefore only called with ``tp_size=1``, where it creates no process group.
"""

from __future__ import annotations

import pytest

from lite_llama.distributed import parallel_state as ps


@pytest.fixture(autouse=True)
def _reset_grid():
    """Restore the world of one after each test.

    The grid is module-level state and model layers read it while being built, so a
    test that left ``dp_size=4`` behind would change how unrelated tests place ranks.
    """
    yield
    ps.destroy_parallel()


def test_defaults_to_a_world_of_one():
    assert ps.get_tp_rank() == 0
    assert ps.get_tp_world_size() == 1
    assert ps.get_dp_rank() == 0
    assert ps.get_dp_world_size() == 1
    assert ps.get_world_size() == 1


@pytest.mark.parametrize(
    ("global_rank", "tp_size", "dp_size", "dp_rank", "tp_rank"),
    [
        # Pure DP: every rank is its own replica.
        (0, 1, 4, 0, 0),
        (3, 1, 4, 3, 0),
        # Pure TP: one replica, ranks are TP ranks.
        (0, 4, 1, 0, 0),
        (3, 4, 1, 0, 3),
        # 2x2 grid: replica 1 owns global ranks 2 and 3.
        (1, 2, 2, 0, 1),
        (2, 2, 2, 1, 0),
        (3, 2, 2, 1, 1),
    ],
)
def test_grid_coordinates(global_rank, tp_size, dp_size, dp_rank, tp_rank):
    """``global_rank = dp_rank * tp_size + tp_rank``, decomposed the other way."""
    assert ps.grid_coordinates(global_rank, tp_size, dp_size) == (dp_rank, tp_rank)


def test_every_global_rank_maps_to_a_distinct_cell():
    """No two ranks of a 2x3 grid may share a ``(dp_rank, tp_rank)`` cell."""
    tp_size, dp_size = 3, 2
    cells = [
        ps.grid_coordinates(global_rank, tp_size, dp_size)
        for global_rank in range(tp_size * dp_size)
    ]

    assert len(set(cells)) == tp_size * dp_size


def test_a_replicas_tp_ranks_are_contiguous():
    """Replica ``d`` must own exactly the global ranks ``[d*tp, (d+1)*tp)``.

    The TP process group is built from that range, so a layout where a replica's ranks
    were strided would all-reduce across replicas instead of within one.
    """
    tp_size, dp_size = 2, 3
    for dp_rank in range(dp_size):
        owned = [
            global_rank
            for global_rank in range(tp_size * dp_size)
            if ps.grid_coordinates(global_rank, tp_size, dp_size)[0] == dp_rank
        ]
        assert owned == list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))


def test_rank_outside_the_grid_is_rejected():
    with pytest.raises(ValueError, match="outside a 2x2 grid"):
        ps.grid_coordinates(4, tp_size=2, dp_size=2)


@pytest.mark.parametrize(("tp_size", "dp_size"), [(0, 1), (1, 0), (-1, 2)])
def test_non_positive_sizes_are_rejected(tp_size, dp_size):
    with pytest.raises(ValueError, match="must be >= 1"):
        ps.grid_coordinates(0, tp_size=tp_size, dp_size=dp_size)


def test_init_parallel_records_pure_dp_without_a_process_group():
    """``tp_size=1`` must place the rank and return, touching no collective library."""
    import torch.distributed as dist

    ps.init_parallel(global_rank=1, tp_size=1, dp_size=2)

    assert (ps.get_dp_rank(), ps.get_tp_rank()) == (1, 0)
    assert ps.get_dp_world_size() == 2
    assert ps.get_world_size() == 2
    assert not dist.is_initialized()


def test_init_parallel_validates_before_mutating_state():
    """A bad rank must leave the previous grid intact, not half-applied."""
    ps.init_parallel(global_rank=1, tp_size=1, dp_size=2)

    with pytest.raises(ValueError):
        ps.init_parallel(global_rank=9, tp_size=1, dp_size=2)

    assert ps.get_dp_rank() == 1
    assert ps.get_dp_world_size() == 2


def test_init_tensor_parallel_is_the_dp_1_case():
    """The TP entry point must place the process in a single-replica grid."""
    ps.init_tensor_parallel(rank=0, world_size=1)

    assert ps.get_tp_world_size() == 1
    assert ps.get_dp_world_size() == 1


def test_destroy_parallel_restores_the_world_of_one():
    ps.init_parallel(global_rank=2, tp_size=1, dp_size=4)
    assert ps.get_dp_rank() == 2

    ps.destroy_parallel()

    assert ps.get_dp_rank() == 0
    assert ps.get_dp_world_size() == 1


def test_collectives_are_no_ops_without_tensor_parallelism():
    """Both collectives must return their input untouched in a world of one.

    They are called unconditionally by ``RowParallelLinear`` and the KV-cache
    profiler, so on a single GPU they have to be free *and* transparent.
    """
    import torch

    ps.init_parallel(global_rank=1, tp_size=1, dp_size=2)
    tensor = torch.ones(4)

    assert ps.all_reduce_tp(tensor) is tensor
    assert ps.all_reduce_min(7) == 7


def test_divide_names_the_dimension_that_does_not_fit():
    assert ps.divide(8, 4, "attention heads") == 2
    with pytest.raises(ValueError, match="attention heads 6 does not divide across 4"):
        ps.divide(6, 4, "attention heads")
