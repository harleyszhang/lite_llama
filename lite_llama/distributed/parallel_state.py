"""The DP x TP rank grid: one global rank space both parallel axes agree on.

``init_parallel`` splits the world into contiguous TP groups — one per DP
replica — and builds the TP process group; the ``get_*`` accessors then
answer rank queries from module state anywhere in the codebase.

Usage:
    init_parallel(global_rank, tp_size, dp_size, master_port)
    assert get_tp_rank() < get_tp_world_size()
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import torch
import torch.distributed as dist

from ..tools.observability import Collective, CollectiveStats
from ..utils.logger import get_logger

_log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Module-level state
# --------------------------------------------------------------------------- #
_TP_RANK: int = 0
_TP_WORLD_SIZE: int = 1
_TP_GROUP: dist.ProcessGroup | None = None
_TP_CPU_GROUP: dist.ProcessGroup | None = None
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
    global _TP_RANK, _TP_WORLD_SIZE, _TP_GROUP, _TP_CPU_GROUP, _DP_RANK, _DP_WORLD_SIZE

    _DP_RANK, _TP_RANK = grid_coordinates(global_rank, tp_size, dp_size)
    _DP_WORLD_SIZE = dp_size
    _TP_WORLD_SIZE = tp_size
    if tp_size <= 1:
        return

    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    world_size = tp_size * dp_size
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    # Decode replays a CUDA graph while prefill stays eager, so captured and
    # non-captured collectives share one communicator for the life of the process.
    # NCCL only supports that mix when this is 1 — with it off, a graph-captured
    # all-reduce and an eager one can end up using the same internal buffers and
    # the result is corrupt activations or a hang, not an error. 1 is NCCL's own
    # default; the line is here so the requirement is stated where the group is
    # built, and ``setdefault`` leaves a deliberate override alone.
    os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=global_rank, world_size=world_size)
    # Every rank must create every group, in the same order, even the ones it is not
    # a member of: ``new_group`` is itself a collective over the whole world.
    for replica in range(dp_size):
        members = list(range(replica * tp_size, (replica + 1) * tp_size))
        group = dist.new_group(members, backend=backend)
        # A second, CPU-backed group over the same ranks carries the *control*
        # plane: :func:`broadcast_object` ships pickled plans, and nccl can
        # only move device memory, so it would have to stage every plan through
        # the GPU. gloo sends the bytes straight from host memory.
        cpu_group = group if backend == "gloo" else dist.new_group(members, backend="gloo")
        if replica == _DP_RANK:
            _TP_GROUP = group
            _TP_CPU_GROUP = cpu_group
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
    global _TP_RANK, _TP_WORLD_SIZE, _TP_GROUP, _TP_CPU_GROUP, _DP_RANK, _DP_WORLD_SIZE
    if _TP_GROUP is not None:
        dist.destroy_process_group()
    _TP_RANK = 0
    _TP_WORLD_SIZE = 1
    _TP_GROUP = None
    _TP_CPU_GROUP = None
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
#
# Primitive names carry no domain suffix: the communication domain is expressed
# by ``group``, not by the name (the vLLM/SGLang convention, and what
# ``torch.distributed``'s own signatures do). ``group=None`` resolves to the
# module-state TP group — today the only persistent group, and therefore what
# every existing call site means; once EP/CP groups exist their callers pass the
# group and these names stay put. Each op reports to :class:`CollectiveStats`
# *after* the world-of-one early return: a no-op collective moves no bytes, so
# counting it would measure call sites rather than traffic.
# --------------------------------------------------------------------------- #
def _payload(tensor: torch.Tensor) -> int:
    """Bytes one rank contributes to a collective over ``tensor``."""
    return tensor.numel() * tensor.element_size()


def _resolve(group: dist.ProcessGroup | None) -> dist.ProcessGroup | None:
    """``None`` means the module-state TP group; an explicit group is used as given."""
    return _TP_GROUP if group is None else group


def _world_of_one(group: dist.ProcessGroup | None) -> bool:
    """Whether a collective over ``group`` has no peer to talk to."""
    return _TP_WORLD_SIZE <= 1 if group is None else dist.get_world_size(group) <= 1


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    *,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Reduce ``tensor`` across the group's ranks, in place. No-op for a world of one.

    The MAX spelling is the other half of a vocabulary-parallel ``log_softmax``:
    each rank holds a slice of the vocabulary, so the row maximum that keeps
    ``exp`` from overflowing has to be the maximum over *all* slices. One float
    per row crosses the wire, not one per token id.
    """
    if _world_of_one(group):
        return tensor
    dist.all_reduce(tensor, op=op, group=_resolve(group))
    CollectiveStats.record(
        Collective.ALL_REDUCE_MAX if op is dist.ReduceOp.MAX else Collective.ALL_REDUCE,
        _payload(tensor),
    )
    return tensor


def all_gather(
    tensor: torch.Tensor, dim: int = -1, *, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """Concatenate every rank's ``tensor`` along ``dim``, rank order.

    Used for the small candidate tensors of vocabulary-parallel sampling — the ``k``
    best ``(logit, id)`` pairs per rank — where the transfer is ``O(k * world_size)``
    and independent of the vocabulary size. Every rank must pass the same shape.
    """
    if _world_of_one(group):
        return tensor
    group = _resolve(group)
    tensor = tensor.contiguous()
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, tensor, group=group)
    CollectiveStats.record(Collective.ALL_GATHER, _payload(tensor))
    return torch.cat(parts, dim=dim)


def reduce_scatter(
    tensor: torch.Tensor, dim: int = -1, *, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """Sum ``tensor`` across ranks, then keep only this rank's shard along ``dim``.

    An all-reduce is a reduce-scatter followed by an all-gather; when the caller
    only wants its shard (sequence parallelism, EP output combining), the gather
    half of that traffic is pure waste, which is what this primitive skips.

    Raises:
        ValueError: If ``dim`` does not divide across the group's ranks.
    """
    if _world_of_one(group):
        return tensor
    group = _resolve(group)
    world = dist.get_world_size(group)
    if tensor.shape[dim] % world != 0:
        raise ValueError(
            f"reduce_scatter dim {dim} of shape {tuple(tensor.shape)} does not divide "
            f"across {world} ranks"
        )
    # reduce_scatter_tensor splits along dim 0, so the shard dim moves there first.
    moved = tensor.movedim(dim, 0).contiguous()
    shard = torch.empty(
        moved.shape[0] // world, *moved.shape[1:], dtype=moved.dtype, device=moved.device
    )
    dist.reduce_scatter_tensor(shard, moved, op=dist.ReduceOp.SUM, group=group)
    CollectiveStats.record(Collective.REDUCE_SCATTER, _payload(moved))
    return shard.movedim(0, dim)


def all_to_all(
    tensor: torch.Tensor, *, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """Equal-split exchange: slice ``j`` of rank ``i``'s tensor lands on rank ``j``.

    The EP token-exchange primitive: with tokens sorted by expert id, one
    all_to_all routes every token to the rank owning its expert. Splits are along
    dim 0, whose size must divide by the world size.

    Raises:
        ValueError: If dim 0 does not divide across the group's ranks.
    """
    if _world_of_one(group):
        return tensor
    group = _resolve(group)
    world = dist.get_world_size(group)
    if tensor.shape[0] % world != 0:
        raise ValueError(
            f"all_to_all dim 0 of shape {tuple(tensor.shape)} does not divide across "
            f"{world} ranks"
        )
    tensor = tensor.contiguous()
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor, group=group)
    CollectiveStats.record(Collective.ALL_TO_ALL, _payload(tensor))
    return output


def send(tensor: torch.Tensor, dst: int, *, group: dist.ProcessGroup | None = None) -> None:
    """Point-to-point handoff to rank ``dst`` *within the group* (CP/PD plumbing).

    Pairs with :func:`recv`; a world of one has no peer, so both are no-ops there.
    """
    if _world_of_one(group):
        return
    dist.send(tensor.contiguous(), group_dst=dst, group=_resolve(group))
    CollectiveStats.record(Collective.SEND, _payload(tensor))


def recv(
    tensor: torch.Tensor, src: int, *, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """Fill ``tensor`` from rank ``src`` *within the group*; pairs with :func:`send`."""
    if _world_of_one(group):
        return tensor
    dist.recv(tensor, group_src=src, group=_resolve(group))
    CollectiveStats.record(Collective.RECV, _payload(tensor))
    return tensor


def broadcast(
    tensor: torch.Tensor, src: int = 0, *, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """Broadcast ``tensor`` from the group-local ``src`` rank to the whole group.

    Used after sampling: rank 0 draws the next token and broadcasts it so every
    rank feeds the same input_ids on the next decode step. Without this, each
    rank would run ``torch.multinomial`` with an independent RNG state and
    diverge on the first non-greedy sample.
    """
    if _world_of_one(group):
        return tensor
    # ``group_src`` keeps ``src`` a group-local rank for every group, which the old
    # hand-rolled ``dp_rank * tp_size + src`` arithmetic only managed for the TP one.
    dist.broadcast(tensor, group_src=src, group=_resolve(group))
    CollectiveStats.record(Collective.BROADCAST, _payload(tensor))
    return tensor


def broadcast_object(
    obj: Any = None, src: int = 0, *, group: dist.ProcessGroup | None = None
) -> Any:
    """Broadcast any picklable object from group-local ``src`` to every rank.

    This is the *control* plane. A tensor-parallel step is decided once — on the
    rank that owns the scheduler — and then run everywhere, so the decision has
    to travel: which slots, which token ids, which sampling parameters. Sending
    it as an object rather than as a pile of padded tensors means the receiving
    ranks reconstruct exactly what the sender planned, with no encoding to keep
    in sync on both sides; and ``group=None`` goes over the gloo TP group, so a
    few hundred bytes of control never touch device memory or serialise behind
    the NCCL stream that the data plane is using. An explicit group is used
    as-is — its owner chooses a backend that can carry host objects.

    Non-root ranks ignore ``obj`` and return what the root sent. Returns ``obj``
    unchanged for a world of one, which is what lets the single-process path
    call this without a branch.
    """
    if _world_of_one(group):
        return obj
    payload = [obj]
    dist.broadcast_object_list(
        payload, group_src=src, group=_TP_CPU_GROUP if group is None else group
    )
    # Sizing a plan means pickling it a second time, so it happens only while a window
    # is open: observability may cost something when asked for, never when not. Sized
    # after the broadcast so a follower reports the same bytes the driver sent.
    if CollectiveStats.collecting():
        CollectiveStats.record(Collective.BROADCAST_OBJECT, len(pickle.dumps(payload[0])))
    return payload[0]


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
    CollectiveStats.record(Collective.ALL_REDUCE_MIN, _payload(tensor))
    return int(tensor.item())


def all_ranks_agree(value: int) -> bool:
    """Whether every TP rank passed the same ``value``. ``True`` when TP is off.

    A consensus primitive rather than a reduction, for decisions that must come
    out the same on every rank *or not be taken at all*. Whether to keep a set of
    captured CUDA graphs is one: the graphs contain collectives, so ranks that
    disagree about which graph to replay do not produce different answers — they
    stop, one of them waiting in an all-reduce its peer never issues. A caller can
    therefore branch on this result and know its peers branch with it.

    Both extremes come back from one collective: reducing ``[value, -value]`` with
    MIN yields ``min`` and ``-max``, which are equal exactly when every rank
    contributed the same number.
    """
    if _TP_WORLD_SIZE <= 1:
        return True
    on_gpu = torch.cuda.is_available()
    device = torch.device("cuda", torch.cuda.current_device()) if on_gpu else None
    tensor = torch.tensor([value, -value], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=_TP_GROUP)
    CollectiveStats.record(Collective.ALL_REDUCE_MIN, _payload(tensor))
    low, negated_high = tensor.tolist()
    return low == -negated_high


def warmup_collectives() -> None:
    """Force this rank's communicator into existence. No-op when TP is off.

    NCCL builds a communicator's device resources on its first collective, and a
    CUDA graph capture cannot allocate — so the first all-reduce inside a capture
    region is the one that fails, or worse, hangs while one rank allocates and the
    others wait. Issuing one throwaway all-reduce beforehand moves that
    initialisation outside every capture.

    The value is checked rather than discarded: a reduction over ones must come
    back as the rank count. That makes this a cheap assertion that the group about
    to be baked into a graph is the group this rank thinks it is — a mismatch
    found here raises, whereas the same mismatch found during capture surfaces as
    a hang with no message.

    Not reported to :class:`CollectiveStats`: every other collective here is
    traffic a *step* pays, and folding a one-off initialisation into that total
    would overstate the per-step cost by a constant.
    """
    if _TP_WORLD_SIZE <= 1:
        return
    on_gpu = torch.cuda.is_available()
    device = torch.device("cuda", torch.cuda.current_device()) if on_gpu else None
    probe = torch.ones(1, dtype=torch.float32, device=device)
    dist.all_reduce(probe, op=dist.ReduceOp.SUM, group=_TP_GROUP)
    total = int(probe.item())
    if total != _TP_WORLD_SIZE:
        raise RuntimeError(
            f"collective warmup summed {total} over a group of {_TP_WORLD_SIZE} ranks; "
            "the process group does not span the ranks this process believes it does"
        )
