"""Parallelism state: the DP x TP rank grid and the collectives TP needs.

Re-exports :mod:`lite_llama.distributed.parallel_state` so that rank queries
(``get_tp_rank``) and group setup (``init_parallel``) come from one import.

Usage:
    from lite_llama.distributed import get_tp_rank, init_parallel
"""

from .parallel_state import (
    all_gather,
    all_reduce,
    all_reduce_min,
    all_to_all,
    broadcast,
    broadcast_object,
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
    recv,
    reduce_scatter,
    send,
)

__all__ = [
    "all_gather",
    "all_reduce",
    "all_reduce_min",
    "all_to_all",
    "broadcast",
    "broadcast_object",
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
    "recv",
    "reduce_scatter",
    "send",
]
