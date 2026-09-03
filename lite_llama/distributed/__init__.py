"""Parallelism state: the DP x TP rank grid and the collectives TP needs.

Re-exports :mod:`lite_llama.distributed.parallel_state` so that rank queries
(``get_tensor_model_parallel_rank``) and group setup (``init_parallel``) come from
one import. The collective names follow vLLM's
``vllm.distributed.parallel_state`` spelling (``tensor_model_parallel_all_reduce``
and friends) so both codebases read the same at the call site.

Usage:
    from lite_llama.distributed import get_tensor_model_parallel_rank, init_parallel
"""

from .parallel_state import (
    all_to_all,
    destroy_parallel,
    destroy_tensor_parallel,
    divide,
    expert_parallel_enabled,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_ep_group,
    get_ep_rank,
    get_ep_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_world_size,
    grid_coordinates,
    init_parallel,
    init_tensor_parallel,
    recv,
    reduce_scatter,
    send,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce_max,
    tensor_model_parallel_all_reduce_min,
    tensor_model_parallel_broadcast,
    tensor_model_parallel_broadcast_object_list,
    tensor_model_parallel_ranks_agree,
    warmup_collectives,
)

__all__ = [
    "all_to_all",
    "destroy_parallel",
    "destroy_tensor_parallel",
    "divide",
    "expert_parallel_enabled",
    "get_data_parallel_rank",
    "get_data_parallel_world_size",
    "get_ep_group",
    "get_ep_rank",
    "get_ep_world_size",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
    "get_world_size",
    "grid_coordinates",
    "init_parallel",
    "init_tensor_parallel",
    "recv",
    "reduce_scatter",
    "send",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_all_reduce",
    "tensor_model_parallel_all_reduce_max",
    "tensor_model_parallel_all_reduce_min",
    "tensor_model_parallel_broadcast",
    "tensor_model_parallel_broadcast_object_list",
    "tensor_model_parallel_ranks_agree",
    "warmup_collectives",
]
