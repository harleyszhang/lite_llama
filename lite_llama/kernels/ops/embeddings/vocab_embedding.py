"""Vocabulary-parallel embedding lookup: gather this rank's rows, zero the rest.

Under tensor parallelism the embedding table is split along the vocabulary, so a
rank's contribution to a token's hidden vector is either its own row or exact
zeros — the ``all_reduce`` over ranks then sums exactly one real embedding per
token. This module is that contribution in a single kernel: the id->row mapping
(subtract the shard start, range-check) is two scalar register ops inside the
kernel, and the gather and the zeroing are one masked load/store pair.

The fused form replaces the eager chain of seven kernels (subtract, compare,
compare, and, clamp, gather, multiply) the lookup used to launch. Under TP there
are no CUDA graphs to hide those launches behind, so every one of them was paid
in full on every step; here the same arithmetic costs one launch plus the
collective.

Usage:
    out = vocab_parallel_embedding(input_ids, weight, shard_start, local_vocab)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ..utils import calculate_settings


@triton.jit
def _vocab_parallel_embedding_kernel(
    ids_ptr,
    weight_ptr,
    out_ptr,
    shard_start,
    local_vocab,
    stride_w,
    stride_o,
    hidden: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per token: the whole shard arithmetic is two scalar ops in
    # registers, which is what "hoist it out of forward" means in practice —
    # not a precomputed table, just no separate kernels for it.
    token = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    col_mask = offs < hidden

    local_row = tl.load(ids_ptr + token) - shard_start
    owned = (local_row >= 0) & (local_row < local_vocab)

    # An id another rank owns leaves the load mask empty, so ``other=0.0`` flows
    # to the store: a zero row is this rank's contribution for that token. The
    # negative pointer arithmetic for ``local_row == -1`` is never dereferenced
    # because the mask is what guards the access, not the pointer value.
    row = tl.load(weight_ptr + local_row * stride_w + offs, mask=col_mask & owned, other=0.0)
    tl.store(out_ptr + token * stride_o + offs, row, mask=col_mask)


def _eager_embedding(
    flat: torch.Tensor, weight: torch.Tensor, shard_start: int, local_vocab: int
) -> torch.Tensor:
    """Torch-native path for CPU tensors, semantically identical to the kernel.

    The gloo test tier (``tests/distributed``) exercises the sharded layers on
    CPU, where Triton cannot run; this is the same subtract/mask/gather chain
    the fused kernel performs in one launch.
    """
    local_ids = flat - shard_start
    owned = (local_ids >= 0) & (local_ids < local_vocab)
    out = F.embedding(local_ids.clamp(0, local_vocab - 1), weight)
    return out * owned.unsqueeze(-1).to(out.dtype)


@torch.no_grad()
def vocab_parallel_embedding(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    shard_start: int,
    local_vocab: int,
) -> torch.Tensor:
    """Look ``input_ids`` up in this rank's ``[local_vocab, hidden]`` slice.

    Ids inside ``[shard_start, shard_start + local_vocab)`` gather their rows;
    every other id produces a zero row, so summing the outputs of all TP ranks
    (``all_reduce_tp``) yields the full unsharded embedding.

    Args:
        input_ids: Token ids of any shape; each is looked up independently.
        weight: This rank's slice of the embedding table, contiguous rows.
        shard_start: First global token id the slice covers.
        local_vocab: Number of rows in the slice.

    Returns:
        ``[input_ids.numel(), hidden]``; the caller views it back to
        ``[*input_ids.shape, hidden]``. CPU tensors take a torch-native path
        with the same semantics — Triton cannot run there.
    """
    flat = input_ids.reshape(-1)
    if not input_ids.is_cuda:
        return _eager_embedding(flat, weight, shard_start, local_vocab)

    hidden = weight.shape[1]
    out = torch.empty(flat.shape[0], hidden, dtype=weight.dtype, device=weight.device)
    if flat.shape[0] == 0:
        return out

    BLOCK, num_warps = calculate_settings(hidden)
    _vocab_parallel_embedding_kernel[(flat.shape[0],)](
        flat,
        weight,
        out,
        shard_start,
        local_vocab,
        weight.stride(0),
        out.stride(0),
        hidden=hidden,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
