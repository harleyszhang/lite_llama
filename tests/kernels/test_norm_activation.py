"""Tests for the fused norm, activation and split-softmax kernels.

These three are the elementwise/reduction kernels on the transformer block's
critical path. What each test is really pinning down:

* ``skip_rmsnorm`` fuses the residual add into the norm and returns *both* the
  normed activation and the updated residual. Callers thread that second value
  into the next block, so a kernel that returned the pre-add residual would
  silently drop one skip connection per layer -- the model still runs and still
  emits plausible text, only worse. The residual output is therefore asserted
  as explicitly as the normed one.
* ``swiglu_forward`` must compute the sigmoid in fp32; doing it in fp16 drifts
  enough to matter after 28 layers.
* ``softmax_split`` computes logsumexp in tiles and combines them, so its whole
  reason to exist is numerical stability on wide rows. It is checked against
  torch on shifted inputs where a naive exp would overflow.
"""

from __future__ import annotations

import pytest
import torch

from lite_llama.kernels import skip_rmsnorm, softmax_split, swiglu_forward
from tests import reference

_RTOL, _ATOL = 2e-2, 2e-2


# --------------------------------------------------------------------------- #
# skip_rmsnorm
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 512), id="single-token"),
        pytest.param((2, 16, 1024), id="typical-prefill"),
        pytest.param((1, 7, 896), id="non-power-of-two-hidden"),
        pytest.param((4, 1, 2048), id="decode-batch"),
    ],
)
def test_skip_rmsnorm_matches_reference(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    residual = torch.randn(shape, device="cuda", dtype=torch.float16)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)

    # The kernel overwrites ``residual`` in place, so the reference has to run
    # against pristine copies (see test_skip_rmsnorm_writes_residual_in_place).
    ref_out, ref_residual = reference.skip_rmsnorm(x.clone(), residual.clone(), weight)
    out, new_residual = skip_rmsnorm(x, residual, weight)

    torch.testing.assert_close(out.float(), ref_out.float(), rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(new_residual.float(), ref_residual.float(), rtol=_RTOL, atol=_ATOL)


def test_skip_rmsnorm_returns_the_post_add_residual():
    """The second return value must be ``x + residual``, not ``residual``.

    Returning the un-added residual is the plausible-but-wrong variant: it keeps
    shapes and dtypes intact, so only a value assertion catches it.
    """
    shape = (1, 4, 256)
    x = torch.full(shape, 2.0, device="cuda", dtype=torch.float16)
    residual = torch.full(shape, 3.0, device="cuda", dtype=torch.float16)
    weight = torch.ones(shape[-1], device="cuda", dtype=torch.float16)

    _, new_residual = skip_rmsnorm(x, residual, weight)
    assert torch.allclose(
        new_residual.float(), torch.full(shape, 5.0, device="cuda"), rtol=_RTOL, atol=_ATOL
    )


def test_skip_rmsnorm_without_residual_is_plain_rmsnorm():
    """``residual=None`` degenerates to RMSNorm and echoes ``x`` back."""
    shape = (2, 8, 512)
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)

    out, passthrough = skip_rmsnorm(x, None, weight)
    ref_out, _ = reference.skip_rmsnorm(x, None, weight)

    torch.testing.assert_close(out.float(), ref_out.float(), rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(passthrough.float(), x.float(), rtol=_RTOL, atol=_ATOL)


def test_skip_rmsnorm_normalises_to_unit_rms():
    """With unit weight the output RMS must be ~1; that is the kernel's contract.

    Checked independently of the reference so a shared misunderstanding of the
    formula cannot make both sides agree on the wrong answer.
    """
    x = torch.randn(1, 32, 1024, device="cuda", dtype=torch.float16) * 5.0
    weight = torch.ones(1024, device="cuda", dtype=torch.float16)
    out, _ = skip_rmsnorm(x, None, weight)
    rms = out.float().pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=5e-2, atol=5e-2)


# --------------------------------------------------------------------------- #
# swiglu_forward
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 256), id="single-token"),
        pytest.param((2, 16, 4864), id="qwen2-intermediate"),
        pytest.param((1, 5, 300), id="non-power-of-two"),
    ],
)
def test_swiglu_matches_reference(shape):
    gate = torch.randn(shape, device="cuda", dtype=torch.float16)
    up = torch.randn(shape, device="cuda", dtype=torch.float16)

    out = swiglu_forward(gate, up)
    ref = reference.swiglu(gate, up)

    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)
    assert out.shape == gate.shape


def test_swiglu_zero_gate_gives_zero():
    """silu(0) == 0, so a zero gate must annihilate the up projection."""
    gate = torch.zeros(1, 4, 128, device="cuda", dtype=torch.float16)
    up = torch.randn(1, 4, 128, device="cuda", dtype=torch.float16)
    out = swiglu_forward(gate, up)
    assert torch.count_nonzero(out) == 0


# --------------------------------------------------------------------------- #
# softmax_split
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((4, 128), id="narrow"),
        pytest.param((2, 4096), id="one-tile"),
        pytest.param((2, 8192), id="two-tiles"),
        pytest.param((1, 151936), id="qwen-vocab"),
    ],
)
def test_softmax_split_matches_torch(shape):
    """Rows wider than TILE_N (4096) exercise the cross-tile logsumexp combine."""
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    out = softmax_split(x)
    torch.testing.assert_close(out, torch.softmax(x, dim=-1), rtol=1e-4, atol=1e-6)


def test_softmax_split_rows_sum_to_one():
    x = torch.randn(3, 8192, device="cuda", dtype=torch.float32)
    out = softmax_split(x)
    sums = out.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-4, atol=1e-5)


def test_softmax_split_does_not_overflow_where_naive_exp_would():
    """A +100 shift makes ``exp(logit)`` overflow fp32; the kernel must survive it.

    That is the entire reason this kernel subtracts a logsumexp instead of
    exponentiating raw logits. At this magnitude the tiled reduction is still
    accurate to a few parts per million, so the result is pinned tightly.
    """
    x = torch.randn(2, 8192, device="cuda", dtype=torch.float32)
    shifted = x + 100.0

    assert not torch.isfinite(torch.exp(shifted)).all(), "shift too small to prove anything"

    out = softmax_split(shifted)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.softmax(shifted, dim=-1), rtol=1e-4, atol=1e-6)


def test_softmax_split_stays_finite_at_extreme_magnitude():
    """At logits ~1e4 the kernel stays finite and normalised, but loses precision.

    ``logz = tile_max + log(tile_sum)`` is stored at the logits' own magnitude, so
    fp32 granularity there (~1e-3 near 1e4) leaks into ``exp(logit - logz)`` as a
    relative error of a few times 1e-4. Torch avoids this by never forming a
    large-magnitude logsumexp. Real logits are O(10), so this is documented
    rather than treated as a defect -- but it is why the tolerance here is 1e-3
    while the +100 case above holds to 1e-4.
    """
    x = torch.randn(2, 8192, device="cuda", dtype=torch.float32)
    out = softmax_split(x + 1e4)

    assert torch.isfinite(out).all()
    sums = out.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-3, atol=1e-4)


def test_skip_rmsnorm_writes_residual_in_place():
    """The kernel stores ``x + residual`` back into the caller's tensor.

    This aliasing is load-bearing: the decoder layer relies on it to thread the
    running residual forward without an extra allocation. It also means a caller
    that needs the original residual afterwards must clone it first, so the
    behaviour is pinned here rather than left as folklore.
    """
    shape = (1, 4, 256)
    x = torch.full(shape, 2.0, device="cuda", dtype=torch.float16)
    residual = torch.full(shape, 3.0, device="cuda", dtype=torch.float16)
    weight = torch.ones(shape[-1], device="cuda", dtype=torch.float16)

    _, returned = skip_rmsnorm(x, residual, weight)

    expected = torch.full(shape, 5.0, device="cuda")
    torch.testing.assert_close(residual.float(), expected, rtol=_RTOL, atol=_ATOL)
    assert returned.data_ptr() == residual.data_ptr()
