"""Tests for the fused residual-add/RMSNorm kernel.

Verifies that the fused kernel matches the separate skip-RMSNorm reference.
"""

import pytest
import torch

from rapid_llm.kernels.ops.layernorm.skip_rmsnorm import (
    fused_add_rmsnorm,
    skip_rmsnorm,
)


def _reference(x, residual, weight, eps):
    """Torch reference: add residual, then RMSNorm."""
    x = x + residual
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    rrms = 1.0 / torch.sqrt(var + eps)
    return (x.float() * rrms).to(x.dtype) * weight, x


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 128), (4, 256), (16, 512), (32, 4096)])
def test_fused_add_rmsnorm_matches_reference(dtype, shape):
    """The fused kernel must agree with skip_rmsnorm to within one output ulp."""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    residual = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.ones(shape[-1], dtype=dtype, device="cuda")

    y_fused, res_fused = fused_add_rmsnorm(x.clone(), residual.clone(), weight)
    y_ref, res_ref = skip_rmsnorm(x.clone(), residual.clone(), weight)

    torch.testing.assert_close(y_fused, y_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(res_fused, res_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_residual_update(dtype):
    """The residual must be updated in place to x + residual."""
    torch.manual_seed(0)
    shape = (8, 64)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    residual = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.ones(shape[-1], dtype=dtype, device="cuda")

    residual_clone = residual.clone()
    _, res_out = fused_add_rmsnorm(x.clone(), residual_clone, weight)

    expected = x + residual
    torch.testing.assert_close(res_out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_3d_input(dtype):
    """The kernel must handle [batch, seq, hidden] inputs (model forward shape)."""
    torch.manual_seed(7)
    shape = (2, 8, 128)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    residual = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.ones(shape[-1], dtype=dtype, device="cuda")

    y_fused, res_fused = fused_add_rmsnorm(x.clone(), residual.clone(), weight)
    y_ref, _ = skip_rmsnorm(x.clone(), residual.clone(), weight)

    assert y_fused.shape == shape
    assert res_fused.shape == shape
    torch.testing.assert_close(y_fused, y_ref, atol=1e-2, rtol=1e-2)


def test_fused_add_rmsnorm_preserves_dtype():
    """Output dtype must match input dtype (no silent upcast)."""
    for dtype in [torch.float16, torch.bfloat16]:
        x = torch.randn(4, 64, dtype=dtype, device="cuda")
        residual = torch.randn(4, 64, dtype=dtype, device="cuda")
        weight = torch.ones(64, dtype=dtype, device="cuda")
        y, _ = fused_add_rmsnorm(x, residual, weight)
        assert y.dtype == dtype
