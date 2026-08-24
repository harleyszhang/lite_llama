"""SwiGLU activation as a fused Triton kernel.

Computes silu(gate) * up in one pass, reading both projection halves and writing
the product without a temporary in HBM. :func:`swiglu_forward` takes the halves
as two tensors; :func:`swiglu_forward_fused` reads them out of the single
``[..., 2 * n_cols]`` tensor the merged gate/up GEMM produces, so the halves are
never split into separate allocations.

Usage:
    out = swiglu_forward(gate, up)
    out = swiglu_forward_fused(torch.cat([gate, up], dim=-1))
"""

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


@triton.jit
def _swiglu_forward_fused_kernel(
    x_ptr, c_ptr, row_stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # One fused row holds gate then up, so the up half starts n_cols elements
    # into the row; row_stride is the whole 2 * n_cols width.
    x_ptr += program_id * row_stride
    c_ptr += program_id * n_cols

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    gate_row = tl.load(x_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    up_row = tl.load(x_ptr + n_cols + col_offsets, mask=mask, other=0)
    c_row = silu(gate_row) * up_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


def swiglu_forward_fused(x):
    """silu(gate) * up where ``x`` is ``concat([gate, up], dim=-1)``.

    The merged gate/up projection emits one ``[..., 2 * n_cols]`` tensor; this
    activates both halves in a single pass without slicing them apart first
    (a slice of a fused row is not contiguous, so the split would copy).
    """
    ori_shape = x.shape  # [..., 2 * n_cols]
    n_cols = ori_shape[-1] // 2
    x = x.reshape(-1, ori_shape[-1])  # GEMM output is contiguous: no copy
    c = torch.empty(x.shape[0], n_cols, dtype=x.dtype, device=x.device)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_fused_kernel[(x.shape[0],)](
        x,
        c,
        x.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return c.view(*ori_shape[:-1], n_cols)
