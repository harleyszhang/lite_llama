"""Tensor-parallel process group: the one piece of global state TP needs.

Every rank builds the *same* model class with the *same* inputs; the only
difference is that each holds a slice of every weight matrix, so the layers that
split their output need no communication and the layers that split their input
finish with an all-reduce (:func:`all_reduce_tp`). Keeping the group in one module
means the model layers can ask "how many ranks, which one am I" without any of the
plumbing being threaded through their constructors, exactly as vLLM's
``parallel_state`` does.

The default state is a world of one, where :func:`all_reduce_tp` is a no-op and
every layer is full width — so single-GPU code paths never branch on TP.

Usage:
    init_tensor_parallel(rank=0, world_size=2, master_port=29500)
    y = all_reduce_tp(partial_y)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from ..utils.logger import get_logger

_log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Module-level state
# --------------------------------------------------------------------------- #
_TP_RANK: int = 0
_TP_WORLD_SIZE: int = 1
_TP_GROUP: dist.ProcessGroup | None = None


def init_tensor_parallel(
    rank: int = 0,
    world_size: int = 1,
    master_port: int = 29500,
) -> None:
    """Initialise the tensor-parallel process group.

    When ``world_size == 1`` this is a no-op: every layer stays full width and
    all collectives become no-ops, so single-GPU inference never branches.

    Args:
        rank: This process's rank within the TP group.
        world_size: Number of TP ranks.
        master_port: TCP port for the rendezvous (rank 0 listens).
    """
    global _TP_RANK, _TP_WORLD_SIZE, _TP_GROUP
    _TP_RANK = rank
    _TP_WORLD_SIZE = world_size
    if world_size <= 1:
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
    _TP_GROUP = dist.new_group(list(range(world_size)), backend="nccl")
    _log.info("TP initialised: rank %d / %d", rank, world_size)


def destroy_tensor_parallel() -> None:
    """Tear down the TP process group."""
    global _TP_RANK, _TP_WORLD_SIZE, _TP_GROUP
    if _TP_GROUP is not None:
        dist.destroy_process_group()
    _TP_RANK = 0
    _TP_WORLD_SIZE = 1
    _TP_GROUP = None


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #
def get_tp_rank() -> int:
    """This process's rank within the TP group (0 when TP is disabled)."""
    return _TP_RANK


def get_tp_world_size() -> int:
    """Number of TP ranks (1 when TP is disabled)."""
    return _TP_WORLD_SIZE


def divide(a: int, b: int, what: str = "") -> int:
    """``a // b`` with a clear error when it does not divide evenly.

    Tensor-parallel sharding divides dimensions across ranks; a non-divisible
    width means the requested TP size is too large for this model.
    """
    if a % b != 0:
        raise ValueError(
            f"{what or 'value'} {a} does not divide across {b} tensor-parallel ranks"
        )
    return a // b


# --------------------------------------------------------------------------- #
# Collectives
# --------------------------------------------------------------------------- #
def all_reduce_tp(tensor: torch.Tensor) -> torch.Tensor:
    """Sum ``tensor`` across all TP ranks. No-op when ``world_size == 1``."""
    if _TP_WORLD_SIZE <= 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=_TP_GROUP)
    return tensor


def all_reduce_min(value: int) -> int:
    """Smallest ``value`` across the group.

    Used to agree on a KV-cache size: profiling runs per rank and two cards with
    different amounts of free memory would otherwise size their caches
    differently, which desynchronises every subsequent allocation index.
    """
    if _TP_WORLD_SIZE <= 1:
        return value
    device = torch.device(f"cuda:{_TP_RANK}") if torch.cuda.is_available() else None
    tensor = torch.tensor([value], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=_TP_GROUP)
    return int(tensor.item())
