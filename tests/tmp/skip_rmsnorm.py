"""
modified from https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/fused/skip_rms_norm.py
"""
import logging
import math

import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["eps"])
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

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)

@torch.no_grad()
def skip_rmsnorm(x, residual, normalized_shape, weight, eps=1e-5):
    shape = X.shape
    X = X.view(-1, shape[-1])
    M, N = X.shape

    BLOCK_SIZE = triton.next_power_of_2(N)
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)

    with torch.cuda.device(x.device):
        skip_rms_norm_kernel[M,](
            y, x, residual, weight, 
            N, 1, N, 1, N, 1, N, 
            eps, BLOCK_SIZE
        )
    return y