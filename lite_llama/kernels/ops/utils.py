"""Shared helpers for the Triton kernels (block sizing, contiguity, HIP detection).

Small utilities the kernels reuse: ``calculate_settings`` picks BLOCK_SIZE /
num_warps, ``ensure_contiguous`` guards kernel inputs, plus a HIP check.

Incorporates code from Unsloth (Apache-2.0, https://github.com/unslothai/unsloth)
and Liger-Kernel; modifications by Yanning Chen, 2024.

Usage:
    block_size, num_warps = calculate_settings(n)
"""

import functools

import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 65536


def is_hip() -> bool:
    return torch.version.hip is not None


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8)


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def calculate_settings(n):
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}
