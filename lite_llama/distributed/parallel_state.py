"""Parallel process groups: the rank grid that TP and DP agree on.

Ranks form a ``dp_size x tp_size`` grid laid out so that one replica's TP ranks are
contiguous — ``global_rank = dp_rank * tp_size + tp_rank`` — which is what lets the two
kinds of parallelism stay unaware of each other. **TP** splits every weight matrix
across the ranks of one replica, so layers that split their input finish with an
all-reduce (:func:`all_reduce_tp`) over *that replica's* group only. **DP** replicates
the whole model per group and shards the *requests* instead, so it needs no collective
in the forward at all; ``torch.distributed`` is therefore only initialised when
``tp_size > 1``, and pure DP runs on plain multiprocessing queues
(:mod:`lite_llama.engine.data_parallel`).

Keeping the grid in one module means a layer can ask "how many ranks, which one am I"
without the plumbing being threaded through its constructor, exactly as vLLM's
``parallel_state`` does. The default state is a world of one, where every collective is
a no-op and every layer is full width — so single-GPU code paths never branch.

Usage:
    init_parallel(global_rank=3, tp_size=2, dp_size=2)   # replica 1, tp rank 1
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
_DP_RANK: int = 0
_DP_WORLD_SIZE: int = 1


def grid_coordinates(global_rank: int, tp_size: int, dp_size: int) -> tuple[int, int]:
    """Decompose a global rank into ``(dp_rank, tp_rank)``.

    The inverse of the layout ``global_rank = dp_rank * tp_size + tp_rank``. Kept
    separate from :func:`init_parallel` because it is the whole of the grid's logic and
    none of its side effects: a caller — or a test — can ask where rank 5 of a 3x2 grid
    sits without a CUDA device or an NCCL rendezvous.

    Args:
        global_rank: This process's rank in ``[0, tp_size * dp_size)``.
        tp_size: Ranks per replica.
        dp_size: Number of replicas.

    Returns:
        ``(dp_rank, tp_rank)`` — which replica, and which rank inside it.

    Raises:
        ValueError: If the sizes are not positive, or ``global_rank`` falls outside
            the grid.
    """
    if tp_size < 1 or dp_size < 1:
        raise ValueError(f"tp_size and dp_size must be >= 1, got {tp_size} and {dp_size}")
    world_size = tp_size * dp_size
    if not 0 <= global_rank < world_size:
        raise ValueError(
            f"global_rank {global_rank} is outside a {dp_size}x{tp_size} grid of {world_size} ranks"
        )
    return global_rank // tp_size, global_rank % tp_size


def init_parallel(
    global_rank: int = 0,
    tp_size: int = 1,
    dp_size: int = 1,
    master_port: int = 29500,
    backend: str | None = None,
) -> None:
    """Place this process in the ``dp_size x tp_size`` rank grid.

    The coordinates come from :func:`grid_coordinates`, so a caller only has to hand
    out consecutive global ranks. A process group is created only when ``tp_size > 1``
    — DP replicas share no tensors, so for pure DP there is nothing to rendezvous about
    and NCCL is never touched.

    Note that with ``tp_size > 1`` this call *blocks* until all ``tp_size * dp_size``
    ranks have joined: that is what ``init_process_group`` means.

    Args:
        global_rank: This process's rank in ``[0, tp_size * dp_size)``.
        tp_size: Ranks per replica (weights split across them).
        dp_size: Number of replicas (requests split across them).
        master_port: TCP port for the TP rendezvous (rank 0 listens).
        backend: ``torch.distributed`` backend; ``None`` picks ``nccl`` on a machine
            with GPUs and ``gloo`` without, which is what lets the sharded layers and
            the vocabulary-parallel sampler be verified on CPU.

    Raises:
        ValueError: If the sizes are not positive, or ``global_rank`` falls outside
            the grid.
    """
    global _TP_RANK, _TP_WORLD_SIZE, _TP_GROUP, _DP_RANK, _DP_WORLD_SIZE

    _DP_RANK, _TP_RANK = grid_coordinates(global_rank, tp_size, dp_size)
    _DP_WORLD_SIZE = dp_size
    _TP_WORLD_SIZE = tp_size
    if tp_size <= 1:
        return

    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    world_size = tp_size * dp_size
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=global_rank, world_size=world_size)
    # Every rank must create every group, in the same order, even the ones it is not
    # a member of: ``new_group`` is itself a collective over the whole world.
    for replica in range(dp_size):
        members = list(range(replica * tp_size, (replica + 1) * tp_size))
        group = dist.new_group(members, backend=backend)
        if replica == _DP_RANK:
            _TP_GROUP = group
    _log.info(
        "parallel state: global rank %d/%d (dp %d/%d, tp %d/%d)",
        global_rank,
        world_size,
        _DP_RANK,
        dp_size,
        _TP_RANK,
        tp_size,
    )


def init_tensor_parallel(
    rank: int = 0,
    world_size: int = 1,
    master_port: int = 29500,
    backend: str | None = None,
) -> None:
    """Initialise a TP-only world: one replica whose ranks are ``[0, world_size)``.

    Kept as the TP entry point (the CLI, benchmarks and tests all call it); it is
    :func:`init_parallel` with ``dp_size=1``.

    Args:
        rank: This process's rank within the TP group.
        world_size: Number of TP ranks.
        master_port: TCP port for the rendezvous (rank 0 listens).
        backend: See :func:`init_parallel`.
    """
    init_parallel(
        global_rank=rank,
        tp_size=world_size,
        dp_size=1,
        master_port=master_port,
        backend=backend,
    )


def destroy_parallel() -> None:
    """Tear down the process group and reset the grid to a world of one."""
    global _TP_RANK, _TP_WORLD_SIZE, _TP_GROUP, _DP_RANK, _DP_WORLD_SIZE
    if _TP_GROUP is not None:
        dist.destroy_process_group()
    _TP_RANK = 0
    _TP_WORLD_SIZE = 1
    _TP_GROUP = None
    _DP_RANK = 0
    _DP_WORLD_SIZE = 1


def destroy_tensor_parallel() -> None:
    """Alias of :func:`destroy_parallel`, kept for the existing TP call sites."""
    destroy_parallel()


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #
def get_tp_rank() -> int:
    """This process's rank within its replica's TP group (0 when TP is disabled)."""
    return _TP_RANK


def get_tp_world_size() -> int:
    """Number of TP ranks per replica (1 when TP is disabled)."""
    return _TP_WORLD_SIZE


def get_dp_rank() -> int:
    """Which replica this process belongs to (0 when DP is disabled)."""
    return _DP_RANK


def get_dp_world_size() -> int:
    """Number of model replicas (1 when DP is disabled)."""
    return _DP_WORLD_SIZE


def get_world_size() -> int:
    """Total ranks across the grid, ``dp_size * tp_size``."""
    return _DP_WORLD_SIZE * _TP_WORLD_SIZE


def divide(a: int, b: int, what: str = "") -> int:
    """``a // b`` with a clear error when it does not divide evenly.

    Tensor-parallel sharding divides dimensions across ranks; a non-divisible
    width means the requested TP size is too large for this model.
    """
    if a % b != 0:
        raise ValueError(f"{what or 'value'} {a} does not divide across {b} tensor-parallel ranks")
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


def all_reduce_max_tp(tensor: torch.Tensor) -> torch.Tensor:
    """Element-wise maximum of ``tensor`` across this replica's TP ranks, in place.

    The other half of a vocabulary-parallel ``log_softmax``: each rank holds a slice of
    the vocabulary, so the row maximum that keeps ``exp`` from overflowing has to be the
    maximum over *all* slices. One float per row crosses the wire, not one per token id.

    No-op when ``tp_world_size == 1``.
    """
    if _TP_WORLD_SIZE <= 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=_TP_GROUP)
    return tensor


def all_gather_tp(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Concatenate every TP rank's ``tensor`` along ``dim``, rank order.

    Used for the small candidate tensors of vocabulary-parallel sampling — the ``k``
    best ``(logit, id)`` pairs per rank — where the transfer is ``O(k * tp)`` and
    independent of the vocabulary size. Every rank must pass the same shape.

    Returns ``tensor`` itself when ``tp_world_size == 1``.
    """
    if _TP_WORLD_SIZE <= 1:
        return tensor
    tensor = tensor.contiguous()
    parts = [torch.empty_like(tensor) for _ in range(_TP_WORLD_SIZE)]
    dist.all_gather(parts, tensor, group=_TP_GROUP)
    return torch.cat(parts, dim=dim)


def broadcast_tp(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast ``tensor`` from the TP-local ``src`` rank to all TP ranks.

    Used after sampling: rank 0 draws the next token and broadcasts it so every
    rank feeds the same input_ids on the next decode step. Without this, each
    rank would run ``torch.multinomial`` with an independent RNG state and
    diverge on the first non-greedy sample.

    No-op when ``tp_world_size == 1``.
    """
    if _TP_WORLD_SIZE <= 1:
        return tensor
    # ``src`` is the *global* rank of the broadcast root within the TP group.
    global_src = _DP_RANK * _TP_WORLD_SIZE + src
    dist.broadcast(tensor, src=global_src, group=_TP_GROUP)
    return tensor


def all_reduce_min(value: int) -> int:
    """Smallest ``value`` across this replica's TP group.

    Used to agree on a KV-cache size: profiling runs per rank and two cards with
    different amounts of free memory would otherwise size their caches
    differently, which desynchronises every subsequent allocation index. DP
    replicas are deliberately excluded — they allocate independently, so one
    replica on a busier card is free to hold a smaller cache.
    """
    if _TP_WORLD_SIZE <= 1:
        return value
    # The tensor must land on *this process's* device, which is not ``_TP_RANK``:
    # under DP x TP the process owns device ``dp_rank * tp_size + tp_rank``, and under
    # ``CUDA_VISIBLE_DEVICES`` the ordinal is remapped again. Asking torch which device
    # the process already selected is the only spelling that is right in every case;
    # the old ``cuda:{_TP_RANK}`` made replica 1 all-reduce from replica 0's card.
    on_gpu = torch.cuda.is_available()
    device = torch.device("cuda", torch.cuda.current_device()) if on_gpu else None
    tensor = torch.tensor([value], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=_TP_GROUP)
    return int(tensor.item())
