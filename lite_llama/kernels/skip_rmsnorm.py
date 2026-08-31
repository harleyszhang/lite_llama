"""Fused residual-add + RMSNorm ("skip rmsnorm") in Triton.

Adds the residual and applies RMSNorm in a single pass, returning both the
normalised output and the updated residual.

Usage:
    out, residual = skip_rmsnorm(x, residual, weight, eps=1e-5)
"""

import torch

from ._compat import tl, triton
from .utils import calculate_settings


@triton.jit
def skip_rms_norm_kernel_no_view(
    Y_ptr,
    X_ptr,
    R_ptr,
    W_ptr,
    B,
    S,
    N,
    x_stride_b,
    x_stride_s,
    x_stride_n,
    r_stride_b,
    r_stride_s,
    r_stride_n,
    y_stride_b,
    y_stride_s,
    y_stride_n,
    w_stride,
    eps,
    has_residual: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # pid表示处理的行号: 行索引 = batch_idx * S + seq_idx
    pid = tl.program_id(0)
    batch_idx = pid // S
    seq_idx = pid % S

    X_ptr = X_ptr + batch_idx * x_stride_b + seq_idx * x_stride_s
    Y_ptr = Y_ptr + batch_idx * y_stride_b + seq_idx * y_stride_s
    # R_ptr只有在has_residual为True时才使用

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + cols * x_stride_n, mask=mask, other=0.0).to(tl.float32)

    # 当有residual时，加载并加上r，然后回写r
    if has_residual:
        R_ptr = R_ptr + batch_idx * r_stride_b + seq_idx * r_stride_s
        r = tl.load(R_ptr + cols * r_stride_n, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        tl.store(R_ptr + cols * r_stride_n, x, mask=mask)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + cols * w_stride, mask=mask, other=0.0)
    y = (x * rrms).to(tl.float16) * w

    tl.store(Y_ptr + cols * y_stride_n, y, mask=mask)


@torch.no_grad()
def skip_rmsnorm_no_view(X, residual, weight, eps=1e-5):
    # 假设X: [B, S, N]
    # 若X为[B,S,N]，不对其进行view
    B, S, N = X.shape
    Y = torch.empty_like(X)

    x_stride_b, x_stride_s, x_stride_n = X.stride()
    y_stride_b, y_stride_s, y_stride_n = Y.stride()
    w_stride = weight.stride(0)

    # 如果 residual 不为 None，则确保与X同shape和stride
    if residual is not None:
        residual = residual.contiguous()  # 确保是连续存储
        r_stride_b, r_stride_s, r_stride_n = residual.stride()
        has_residual = True
    else:
        # 如果 residual 是 None，则在kernel中不处理residual
        # 这里给r_stride_*赋默认值，但不会使用
        r_stride_b, r_stride_s, r_stride_n = 0, 0, 0
        has_residual = False

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (B * S,)

    skip_rms_norm_kernel_no_view[grid](
        Y,
        X,
        residual
        if residual is not None
        else X,  # 若无residual，这里传X只是占位，kernel中不使用R_ptr
        weight,
        B,
        S,
        N,
        x_stride_b,
        x_stride_s,
        x_stride_n,
        r_stride_b,
        r_stride_s,
        r_stride_n,
        y_stride_b,
        y_stride_s,
        y_stride_n,
        w_stride,
        eps,
        has_residual=has_residual,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (Y, residual) if residual is not None else (Y, X)


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
def skip_rmsnorm(X, residual, weight, eps=1e-5):
    orig_shape = X.shape
    X = X.contiguous().view(-1, orig_shape[-1])

    M, N = X.shape  # n_rows, n_cols
    BLOCK_SIZE, num_warps = calculate_settings(N)
    Y = torch.empty_like(X)

    if residual is not None:
        residual = residual.contiguous().view(-1, N)
        skip_rms_norm_kernel[M,](
            Y,
            X,
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
            X,
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
        return Y.view(orig_shape), X.view(orig_shape)
