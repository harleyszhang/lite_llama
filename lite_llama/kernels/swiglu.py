# reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py

import torch
import triton
import triton.language as tl

from .utils import calculate_settings


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, row_stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * row_stride
    b_ptr += program_id * row_stride
    c_ptr += program_id * row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


def swiglu_forward(a, b):
    ori_shape = a.shape  # ori_shape is [batch_size, seq_len, hidden_size]

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),  # c.stride(-2) = n_cols
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return c.view(*ori_shape)
