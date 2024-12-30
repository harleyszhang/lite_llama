"""
modified from https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/rms_norm.py
"""

import operator

import torch
import triton
import triton.language as tl

from ..utils import (
    calculate_settings,
    compare_version,
)

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt

@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    n_cols,
    eps,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    """
    y_i = (x_i / (RMS)) * (offset_wi + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    X_row = X_row.to(tl.float32) # On Llama, only rstd is computed on fp32

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    X_row = X_row * rstd
    X_row = X_row.to(X_row_dtype)
    Y_row = X_row * (W_row + offset)

    tl.store(Y_ptr + col_offsets, Y_row.to(X_row_dtype), mask=mask)


@torch.no_grad()
def rmsnorm_fwd(X, W, eps=1e-5, offset=0.0):

    shape = X.shape
    X = X.view(-1, shape[-1])
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)

    # Check constraints.
    assert (
        X.shape[1] == W.shape[0]
    ), "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"

    _rms_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        n_cols,
        eps,
        offset,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y.view(*shape)


def test_rms_layernorm(
    dim = 1024, eps = 1e-5, dtype = torch.float16,
    bsz = 21, random_state = 3407, seqlen = 3341,
):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    layernorm = LlamaRMSNorm((dim,), eps = eps).to("cuda")
    torch.cuda.manual_seed(random_state)
    torch.manual_seed(random_state)
    torch.nn.init.uniform_(layernorm.weight)
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
    Y = layernorm(X)
    Y2 = rmsnorm_fwd(X, layernorm.weight, eps)

    assert(torch.amax(Y - Y2).item() <= 0.05)
    print("max delta:", torch.max(torch.abs(Y - Y2)))


def testing_suite_layernorm():
    for dim in [512, 1024, 2048]:
        for dtype in [torch.float16, torch.bfloat16]:
            with torch.autocast(device_type = "cuda", dtype = dtype):
                for seqlen in [3341, 2048, 349]:
                    for random_state in [3407, 42]:
                        test_rms_layernorm(
                            dim = dim,
                            eps = 1e-5,
                            dtype = dtype,
                            bsz = 21,
                            random_state = random_state,
                            seqlen = seqlen,
                        )

if __name__ == "__main__":
    testing_suite_layernorm()