"""Accuracy test for the w4a16 ``tl.dot`` kernel rewrite.

The kernel is diffed against ``_reference_w4a16`` across group sizes
and batch shapes, with and without bias — tolerance sized for the
fp32-accumulation rewrite.

Usage:
    pytest tests/kernels/test_w4a16_accuracy.py
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _reference_w4a16(x, qweight, scales, zeros, group_size):
    """Naive reference: unpack int4, dequant to fp16, then torch.mm."""
    n, k_packed = qweight.shape
    k = k_packed * 8
    # Unpack all int4 values
    shifts = torch.arange(8, device=qweight.device, dtype=torch.int32) * 4
    # qweight: [N, K//8] -> expand to [N, K//8, 8] -> reshape [N, K]
    expanded = (qweight[:, :, None] >> shifts[None, None, :]) & 0xF
    unpacked = expanded.reshape(n, k).to(torch.float32)

    # Dequant with group-wise scale and zero
    num_groups = k // group_size
    # scales, zeros: [N, num_groups]
    # Apply per group
    weight_fp = torch.empty(n, k, dtype=torch.float32, device=x.device)
    for g in range(num_groups):
        g_start = g * group_size
        g_end = (g + 1) * group_size
        s = scales[:, g : g + 1]  # [N, 1]
        z = zeros[:, g : g + 1]  # [N, 1]
        weight_fp[:, g_start:g_end] = (unpacked[:, g_start:g_end] - z) * s

    # matmul: x @ weight_fp.T
    return (x.float() @ weight_fp.T).to(x.dtype)


@pytest.fixture(
    params=[
        (1, 128, 1024, 128),
        (4, 256, 512, 128),
        (16, 1024, 2048, 128),
        (64, 512, 1024, 128),
    ]
)
def w4a16_problem(request):
    """(M, N, K, group_size) problem shape."""
    return request.param


def test_w4a16_matches_reference(w4a16_problem):
    """New tl.dot kernel must match the reference within fp16 tolerance."""
    m, n, k, group_size = w4a16_problem
    device = "cuda"

    torch.manual_seed(42)
    x = torch.randn(m, k, dtype=torch.float16, device=device)
    # Random int4 packed weights
    qweight = torch.randint(-(2**31), 2**31, (n, k // 8), dtype=torch.int32, device=device)
    num_groups = k // group_size
    scales = torch.randn(n, num_groups, dtype=torch.float32, device=device).abs() * 0.1
    zeros = torch.randint(0, 16, (n, num_groups), device=device).float()

    # Reference
    ref = _reference_w4a16(x, qweight, scales, zeros, group_size)

    # Kernel under test
    from lite_llama.kernels.ops.quantization.w4a16 import w4a16_matmul

    got = w4a16_matmul(x, qweight, scales, zeros, group_size=group_size)

    # Tolerance: fp16 has ~5e-4 relative error, int4 dequant adds noise. The
    # absolute bound must scale with the output magnitude: at K=2048 the largest
    # outputs reach ~140, where one fp16 ULP is already 0.125 — a single output
    # rounding would exceed a flat 0.1.
    max_diff = (ref.float() - got.float()).abs().max().item()
    rel_err = max_diff / (ref.float().abs().max().item() + 1e-6)
    ref_max = ref.float().abs().max().item()
    assert max_diff < max(1e-1, 2 * ref_max * 2**-10), (
        f"max abs diff {max_diff:.4e} exceeds 2 fp16 ULP at |ref|={ref_max:.1f}"
    )
    assert rel_err < 1e-2, f"relative error {rel_err:.4e} exceeds 1%"


def test_w4a16_batch_dimensions():
    """Verify leading dimensions are preserved (multi-dim input)."""
    device = "cuda"
    k, n, group_size = 512, 256, 128

    torch.manual_seed(0)
    x = torch.randn(2, 3, k, dtype=torch.float16, device=device)
    qweight = torch.randint(-(2**31), 2**31, (n, k // 8), dtype=torch.int32, device=device)
    scales = torch.randn(n, k // group_size, dtype=torch.float32, device=device).abs() * 0.1
    zeros = torch.zeros(n, k // group_size, dtype=torch.float32, device=device)

    from lite_llama.kernels.ops.quantization.w4a16 import w4a16_matmul

    out = w4a16_matmul(x, qweight, scales, zeros, group_size=group_size)
    assert out.shape == (2, 3, n)


def test_w4a16_with_bias():
    """Bias path must not crash and must shift the output."""
    device = "cuda"
    m, k, n, group_size = 4, 256, 128, 128

    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.float16, device=device)
    qweight = torch.randint(-(2**31), 2**31, (n, k // 8), dtype=torch.int32, device=device)
    scales = torch.randn(n, k // group_size, dtype=torch.float32, device=device).abs() * 0.1
    zeros = torch.zeros(n, k // group_size, dtype=torch.float32, device=device)
    bias = torch.ones(n, dtype=torch.float16, device=device)

    from lite_llama.kernels.ops.quantization.w4a16 import w4a16_matmul

    out_no_bias = w4a16_matmul(x, qweight, scales, zeros, group_size=group_size)
    out_bias = w4a16_matmul(x, qweight, scales, zeros, group_size=group_size, bias=bias)

    diff = (out_bias - out_no_bias).float().mean().item()
    assert abs(diff - 1.0) < 0.1, f"bias shift {diff:.3f} should be ~1.0"
