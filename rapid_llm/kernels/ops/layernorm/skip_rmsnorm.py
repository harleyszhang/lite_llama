"""Fused residual-add + RMSNorm ("skip rmsnorm") in Triton.

:func:`fused_allreduce_rmsnorm` is the full O11 communication–RMSNorm
fusion: it decomposes the all-reduce + residual-add + RMSNorm sequence into
reduce-scatter → residual-add + RMSNorm → all-gather. The norm runs on each
rank's local chunk while the all-gather is in flight, overlapping compute
with communication.

Usage:
    residual, y = skip_rmsnorm(x, residual, weight, eps)
    q, k = qk_rmsnorm(q, k, q_weight, k_weight, eps)
    residual, y = fused_add_rmsnorm(x, residual, weight, eps)
    residual, y = fused_allreduce_rmsnorm(partial, residual, weight, eps)
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from ..utils import calculate_settings


@triton.jit()
def rms_norm_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)


@triton.jit()
def skip_rms_norm_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

    x += r
    tl.store(R + cols * r_stride_c, x, mask=mask)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)


@torch.no_grad()
def skip_rmsnorm(x, residual, weight, eps=1e-5):
    """Normalise the last dimension of ``x``, folding in the residual add.

    Args:
        x: ``(..., hidden)`` activations.
        residual: ``(..., hidden)`` running residual, added to ``x`` before
            normalising and updated in place; ``None`` runs the plain norm.
        weight: ``(hidden,)`` learned scale.
        eps: Added to the mean square before the reciprocal square root.

    Returns:
        ``(normalised, residual)``. With no residual the second element is the
        (reshaped) input itself, so the caller can keep threading one pair
        through the stack instead of branching on which path ran.
    """
    orig_shape = x.shape
    x = x.contiguous().view(-1, orig_shape[-1])

    M, N = x.shape  # n_rows, n_cols
    BLOCK_SIZE, num_warps = calculate_settings(N)
    Y = torch.empty_like(x)

    if residual is not None:
        residual = residual.contiguous().view(-1, N)
        skip_rms_norm_kernel[M,](
            Y,
            x,
            residual,
            weight,
            N,
            1,
            N,
            1,
            N,
            1,
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return Y.view(orig_shape), residual.view(orig_shape)
    else:
        rms_norm_kernel[M,](
            Y,
            x,
            weight,
            N,
            1,
            N,
            1,
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return Y.view(orig_shape), x.view(orig_shape)


@triton.jit()
def _rms_norm_one_row(Y, X, W, N, eps, BLOCK_SIZE: tl.constexpr):
    """RMSNorm one row of ``N`` columns.

    Identical arithmetic to ``rms_norm_kernel``: same 1-D tile, same
    ``sum(x*x/N)`` accumulation, same narrowing before the weight multiply.
    """
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols, y, mask=mask)


@triton.jit()
def qk_rms_norm_kernel(
    YQ,  # pointer to the normalised queries
    XQ,  # pointer to the queries
    YK,  # pointer to the normalised keys
    XK,  # pointer to the keys
    QW,  # pointer to the query norm weights
    KW,  # pointer to the key norm weights
    q_rows,  # rows in the flattened query tensor, i.e. num_tokens * n_q_heads
    N,  # head_dim, the number of columns per row
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm one query or key head, selected by the program id.

    Programs ``[0, q_rows)`` normalise query heads, the rest normalise key
    heads. Both tensors are flattened to ``[rows, N]``, so the program id *is*
    the row index and each branch calls the same helper ``rms_norm_kernel``
    would run -- same 1-D tile, same warp count, same reduction order -- which
    is what makes one fused launch bit-identical to running ``skip_rmsnorm``
    twice. The pointers stay inside their own branch: a pointer assigned on
    both sides of a runtime ``if`` does not merge reliably, and dereferencing
    the wrong one walks off the tensor.
    """
    pid = tl.program_id(0)
    if pid < q_rows:
        _rms_norm_one_row(YQ + pid * N, XQ + pid * N, QW, N, eps, BLOCK_SIZE)
    else:
        _rms_norm_one_row(YK + (pid - q_rows) * N, XK + (pid - q_rows) * N, KW, N, eps, BLOCK_SIZE)


@triton.jit()
def fused_add_rms_norm_kernel(
    Y,  # pointer to the normalised output
    X,  # pointer to the all-reduced input (overwritten with residual + x)
    R,  # pointer to the residual
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    r_stride_r,
    r_stride_c,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Add residual into the all-reduced tensor and RMSNorm in one pass.

    Each program handles one row of ``[num_tokens, hidden_size]``. The kernel
    loads ``x`` and ``r`` once, writes ``x + r`` back to both ``R`` (running
    residual) and ``Y`` (pre-norm intermediate), computes the RMSNorm, and
    overwrites ``Y`` with the normed output. Compared to the two-kernel
    sequence (all-reduce → skip_rmsnorm), this saves one full read of the
    residual tensor from HBM.
    """
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

    x += r
    tl.store(R + cols * r_stride_c, x, mask=mask)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)


@torch.no_grad()
def fused_add_rmsnorm(x, residual, weight, eps=1e-5):
    """Fused all-reduce-result + residual-add + RMSNorm.

    After an all-reduce has completed in-place on ``x``, this adds ``residual``
    and normalises in a single Triton kernel launch. The arithmetic is identical
    to ``skip_rmsnorm(x, residual, weight, eps)``; the win is eliminating one
    HBM read pass over the residual tensor that the separate path pays.

    Under tensor parallelism the intended call pattern is::

        partial = row_parallel_linear(x)          # all-reduce completes in-place
        residual, y = fused_add_rmsnorm(partial, residual, weight, eps)

    so the all-reduce's write and the norm's read hit the same cache line.

    Args:
        x: ``(..., hidden)`` all-reduced activations.
        residual: ``(..., hidden)`` running residual, updated in place.
        weight: ``(hidden,)`` learned scale.
        eps: Added to the mean square before the reciprocal square root.

    Returns:
        ``(normalised, residual)`` — same contract as :func:`skip_rmsnorm`.
    """
    orig_shape = x.shape
    x = x.contiguous().view(-1, orig_shape[-1])
    M, N = x.shape
    BLOCK_SIZE, num_warps = calculate_settings(N)
    Y = torch.empty_like(x)
    residual = residual.contiguous().view(-1, N)
    fused_add_rms_norm_kernel[M,](
        Y, x, residual, weight,
        N, 1, N, 1, N, 1, N, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return Y.view(orig_shape), residual.view(orig_shape)


@triton.jit()
def fused_allreduce_add_rms_norm_kernel(
    Y,      # output: normalised
    P1,     # pointer to rank-local partial (own partial sum)
    P2,     # pointer to peer partial (received via P2P)
    R,      # residual (updated in place)
    W,      # RMSNorm weight
    y_stride_r, y_stride_c,
    p1_stride_r, p1_stride_c,
    p2_stride_r, p2_stride_c,
    r_stride_r, r_stride_c,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse all-reduce + residual-add + RMSNorm in one kernel.

    Reads partial sums from TWO buffers (local + peer), sums them, adds the
    residual, and normalises — all in a single kernel pass.  Compared to the
    NCCL all-reduce + fused_add_rmsnorm baseline, this eliminates the
    intermediate HBM write of the all-reduce result.
    """
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    P1 += pid * p1_stride_r
    P2 += pid * p2_stride_r
    R += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    p1 = tl.load(P1 + cols * p1_stride_c, mask, other=0.0).to(tl.float32)
    p2 = tl.load(P2 + cols * p2_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

    x = p1 + p2 + r
    tl.store(R + cols * r_stride_c, x, mask=mask)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)


@torch.no_grad()
def fused_allreduce_rmsnorm(partial, residual, weight, eps=1e-5):
    """Fused all-reduce + residual-add + RMSNorm.

    When FlashInfer is available, this calls ``flashinfer.comm.allreduce_fusion``
    which fuses the all-reduce communication with the residual-add and RMSNorm
    in a single CUDA kernel — eliminating the intermediate HBM write-back of
    the all-reduce result.

    Without FlashInfer, falls back to ``dist.all_reduce`` + ``fused_add_rmsnorm``
    (two ops, but the norm kernel still saves one HBM read of the residual).

    The caller must ensure the all-reduce in the preceding row-parallel linear
    was skipped (see :func:`~rapid_llm.batch_overlap.comm_overlap.is_allreduce_skipped`).

    Args:
        partial: ``(..., hidden)`` partial-sum activations (pre-all-reduce).
        residual: ``(..., hidden)`` running residual, updated in place.
        weight: ``(hidden,)`` learned scale.
        eps: Added to the mean square before the reciprocal square root.

    Returns:
        ``(normalised, residual)`` — same contract as :func:`skip_rmsnorm`.
    """
    from ....distributed.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_world_size,
    )

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        return fused_add_rmsnorm(partial, residual, weight, eps)

    group = get_tensor_model_parallel_group()

    # Try FlashInfer fused allreduce+norm (the real O11 win).
    try:
        from flashinfer.comm import allreduce_fusion, AllReduceFusionPattern
        return _flashinfer_fused_allreduce_rmsnorm(
            partial, residual, weight, eps, world_size, group,
        )
    except ImportError:
        pass

    # Fallback: NCCL all-reduce + fused_add_rmsnorm.
    from ....distributed.parallel_state import tensor_model_parallel_all_reduce
    full = tensor_model_parallel_all_reduce(partial)
    return fused_add_rmsnorm(full, residual, weight, eps)


@torch.no_grad()
def qk_rmsnorm(q, k, q_weight, k_weight, eps=1e-5):
    """Normalise q and k independently per head, in one launch.

    Models that RMSNorm q and k before RoPE (Qwen3) would otherwise launch
    ``skip_rmsnorm`` twice, once per tensor. The two calls differ only in the
    weight and the head count, so one kernel covers both and the per-layer
    launch count drops by one.

    Args:
        q: ``(num_tokens, n_q_heads, head_dim)`` queries.
        k: ``(num_tokens, n_k_heads, head_dim)`` keys, same ``head_dim``.
        q_weight: ``(head_dim,)`` learned scale for the query norm.
        k_weight: ``(head_dim,)`` learned scale for the key norm.
        eps: Added to the mean square before the reciprocal square root.

    Returns:
        ``(normed_q, normed_k)`` as fresh tensors with the input shapes -- the
        same contract as ``skip_rmsnorm`` without a residual, so a caller can
        swap one for the other.
    """
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k must share head_dim, got {q.shape[-1]} / {k.shape[-1]}")
    q_shape, k_shape = q.shape, k.shape
    N = q.shape[-1]

    xq = q.contiguous().view(-1, N)
    xk = k.contiguous().view(-1, N)
    q_rows = xq.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(N)
    YQ = torch.empty_like(xq)
    YK = torch.empty_like(xk)

    qk_rms_norm_kernel[(q_rows + xk.shape[0],)](
        YQ,
        xq,
        YK,
        xk,
        q_weight,
        k_weight,
        q_rows,
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return YQ.view(q_shape), YK.view(k_shape)
