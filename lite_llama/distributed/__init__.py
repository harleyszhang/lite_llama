"""Parallelism state: the DP x TP rank grid and the collectives TP needs.

What those collectives cost is counted by
:mod:`lite_llama.tools.observability.collective_stats`, which is where the reporting
lives — this package runs the wire, it does not account for it.
"""

from .parallel_state import (
    all_reduce_min,
    all_reduce_tp,
    broadcast_object_tp,
    destroy_parallel,
    destroy_tensor_parallel,
    divide,
    get_dp_rank,
    get_dp_world_size,
    get_tp_rank,
    get_tp_world_size,
    get_world_size,
    grid_coordinates,
    init_parallel,
    init_tensor_parallel,
)

__all__ = [
    "all_reduce_min",
    "all_reduce_tp",
    "broadcast_object_tp",
    "destroy_parallel",
    "destroy_tensor_parallel",
    "divide",
    "get_dp_rank",
    "get_dp_world_size",
    "get_tp_rank",
    "get_tp_world_size",
    "get_world_size",
    "grid_coordinates",
    "init_parallel",
    "init_tensor_parallel",
]
