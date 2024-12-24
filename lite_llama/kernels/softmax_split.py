# modified from https://github.com/iclementine/optimize_softmax/blob/master/softmax_split.py

import triton
from triton import language as tl
import torch

@triton.jit
def logsumexp_kernel(
    out_ptr,
    in_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    num_programs_n = tl.num_programs(0)
    pid_m = tl.program_id(1)

    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    offset = pid_m * N + n_offsets
    inp = tl.load(in_ptr + offset, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(inp, 0)
    e = tl.exp(inp - m)
    z = tl.sum(e, 0)
    logz = m + tl.log(z)

    output_ptrs = out_ptr + pid_m * num_programs_n + pid_n
    tl.store(output_ptrs, logz)

@triton.jit
def combine_logsumexp_kernel(out_ptr, inp_ptr, M, N, TILE_N: tl.constexpr):
    pid_m = tl.program_id(0)
    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N
    logzs = tl.load(inp_ptr + pid_m * N + n_offsets, other=-float("inf"), mask=mask).to(out_ptr.dtype.element_ty)
    m = tl.max(logzs, 0)
    e = tl.exp(logzs - m)
    z = tl.sum(e, 0)
    logz = m + tl.log(z)
    tl.store(out_ptr + pid_m, logz)

@triton.jit
def softmax_kernel(out_ptr, in_ptr, logz_ptr, M, N, TILE_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    offset = pid_m * N + n_offsets
    mask = n_offsets < N
    inp = tl.load(in_ptr + offset, mask=mask, other=-float("inf")).to(out_ptr.dtype.element_ty)
    logz = tl.load(logz_ptr + pid_m).to(tl.float32)
    out = tl.exp(inp - logz)
    tl.store(out_ptr + offset, out, mask=mask)


def softmax_split(x):
    M, N = x.shape

    # num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count

    TILE_N = min(4096, triton.next_power_of_2(N))
    num_tiles_n = triton.cdiv(N, TILE_N)
    logz = torch.empty((M, num_tiles_n), dtype=x.dtype, device=x.device)
    grid = (num_tiles_n, M, 1)
    logsumexp_kernel[grid](logz, x, M, N, TILE_N)

    combined_logz = torch.empty((M, ), dtype=x.dtype, device=x.device)
    TILE_N = triton.next_power_of_2(num_tiles_n)
    grid = (M, 1, 1)
    combine_logsumexp_kernel[grid](combined_logz, logz, M, num_tiles_n, TILE_N)

    out = torch.empty_like(x)
    TILE_N = min(4096, triton.next_power_of_2(N))
    num_tiles_n = triton.cdiv(N, TILE_N)
    grid = (num_tiles_n, M, 1)
    softmax_kernel[grid](out, x, combined_logz, M, N, TILE_N)
    return out
