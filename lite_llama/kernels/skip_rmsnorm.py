"""
modified from https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/fused/skip_rms_norm.py
"""
import logging
import math

import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["eps"])
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
    tl.store(R + cols * r_stride_c, x, mask=mask)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)

@torch.no_grad()
def skip_rmsnorm(X, residual, weight, eps=1e-5):
    orig_shape = X.shape
    X = X.view(-1, orig_shape[-1])

    M, N = X.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    Y = torch.empty_like(X)

    if residual is not None:
        residual = residual.view(-1, N)
        skip_rms_norm_kernel[M,](
            Y, X, residual, weight, 
            N, 1, N, 1, N, 1, N, 
            eps, BLOCK_SIZE
        )
        return Y.view(orig_shape), residual.view(orig_shape)
    else:
        rms_norm_kernel[M,](
            Y, X, weight, 
            N, 1, N, 1, N, 
            eps, BLOCK_SIZE
        )
        return Y.view(orig_shape)

import torch
import pytest

def python_rmsnorm(x, w, eps=1e-5):
    # x: (B, N)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x / torch.sqrt(var + eps)
    return x_normed * w

def python_skip_rmsnorm(x, r, w, eps=1e-5):
    # x, r: (B, N)
    x = x + r
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x / torch.sqrt(var + eps)
    return (x_normed * w).half(), x.half()

@pytest.mark.parametrize("batch_size, N, hidden_size", [(4, 128, 4096), (2, 256, 4096), (8, 1024, 4096)])
def test_rmsnorm(batch_size, N, hidden_size):
    x = torch.randn(batch_size, N, hidden_size, device='cuda', dtype=torch.float16)
    w = torch.randn(hidden_size, device='cuda', dtype=torch.float16)

    y_ref = python_rmsnorm(x.float(), w.float()).half()
    y_triton = skip_rmsnorm(x, None, w)  # 不传residual，就走rms_norm_kernel分支

    assert torch.allclose(y_ref, y_triton, atol=1e-3, rtol=1e-3), "RMSNorm results do not match"

@pytest.mark.parametrize("batch_size, N, hidden_size", [(4, 128, 4096), (2, 256, 4096), (8, 1024, 4096)])
def test_skip_rmsnorm(batch_size, N, hidden_size):
    x = torch.randn(batch_size, N, hidden_size, device='cuda', dtype=torch.float16)
    r = torch.randn(batch_size, N, hidden_size, device='cuda', dtype=torch.float16)
    w = torch.randn(hidden_size, device='cuda', dtype=torch.float16)

    y_ref, py_residual = python_skip_rmsnorm(x.float(), r.float(), w.float())
    y_triton, triton_residual = skip_rmsnorm(x, r, w)

    assert torch.allclose(y_ref, y_triton, atol=1e-3, rtol=1e-3), "Skip RMSNorm results do not match"
    assert torch.allclose(py_residual, triton_residual, atol=1e-3, rtol=1e-3), "Skip RMSNorm residual results do not match"

import time

def benchmark_skip_rmsnorm(batch_size, N, iters=1000):
    x = torch.randn(batch_size, N, device='cuda', dtype=torch.float16)
    r = torch.randn(batch_size, N, device='cuda', dtype=torch.float16)
    w = torch.randn(N, device='cuda', dtype=torch.float16)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        skip_rmsnorm(x, r, w)
    torch.cuda.synchronize()
    end = time.time()
    avg_time = (end - start) / iters
    print(f"skip_rmsnorm: B={batch_size}, N={N}, avg_time={avg_time*1e3:.3f} ms/iter")

if __name__ == "__main__":
    # 示例运行
    benchmark_skip_rmsnorm(4, 128)
    benchmark_skip_rmsnorm(4, 1024)
    benchmark_skip_rmsnorm(32, 1024)