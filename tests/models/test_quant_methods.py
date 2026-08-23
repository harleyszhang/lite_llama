"""Tests for the quant-method strategies and the runtime quantisation schemes.

The Triton kernels themselves are covered in ``tests/kernels/test_quantization.py``;
what is tested here is everything around them, on CPU: the format -> method
registry, the parameter layout each method allocates, and the fp16 -> low-bit
conversion that backs ``--quantization <scheme>``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lite_llama.modules.linear import ReplicatedLinear
from lite_llama.models.quantization import (
    FP8,
    INT4,
    INT8,
    SMOOTHQUANT,
    RUNTIME_SCHEMES,
    QuantConfig,
    get_linear_method,
    get_moe_method,
    quantize_fp8_per_channel,
    quantize_int8_groupwise,
)
from lite_llama.models.quantization.methods import (
    AWQLinearMethod,
    Fp8LinearMethod,
    GPTQLinearMethod,
    SmoothQuantLinearMethod,
    UnquantizedLinearMethod,
    UnquantizedMoeMethod,
    W4A16LinearMethod,
    W8A16LinearMethod,
    W8A16MoeMethod,
)


class _StubMoeBlock(nn.Module):
    """Minimum surface a MoeQuantMethod touches: sizes, ``quant``, ``experts``."""

    def __init__(
        self,
        num_experts: int = 4,
        hidden_size: int = 256,
        moe_intermediate_size: int = 128,
        quant: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.quant = quant


def _fill_fp16(layer: ReplicatedLinear, scale: float = 0.05) -> torch.Tensor:
    """Give an fp16 layer a known weight and return a float copy of it."""
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight, dtype=torch.float32) * scale)
    return layer.weight.data.float().clone()


# --------------------------------------------------------------------------- #
# Registry: format -> method
# --------------------------------------------------------------------------- #
def test_linear_method_registry():
    assert isinstance(get_linear_method(None), UnquantizedLinearMethod)
    # fp8 linears run true W8A8 (per-token fp8 activations); int8 stays W8A16.
    assert isinstance(get_linear_method(QuantConfig(FP8, 128, 128)), Fp8LinearMethod)
    assert isinstance(get_linear_method(QuantConfig(INT8, 128, 128)), W8A16LinearMethod)
    assert isinstance(
        get_linear_method(QuantConfig.smoothquant_per_channel()), SmoothQuantLinearMethod
    )
    assert isinstance(get_linear_method(QuantConfig.int4_groupwise()), W4A16LinearMethod)


def test_int4_method_dispatches_on_checkpoint_method():
    """AWQ/GPTQ checkpoints get their named (vLLM-style) method classes."""
    assert isinstance(get_linear_method(QuantConfig(INT4, 1, 128, method="awq")), AWQLinearMethod)
    assert isinstance(
        get_linear_method(QuantConfig(INT4, 1, 128, method="gptq")), GPTQLinearMethod
    )
    # Runtime int4 (no checkpoint layout) stays on the generic method.
    assert isinstance(get_linear_method(QuantConfig(INT4, 1, 128)), W4A16LinearMethod)


def test_moe_method_registry():
    assert isinstance(get_moe_method(None), UnquantizedMoeMethod)
    for fmt in (FP8, INT8, SMOOTHQUANT):
        assert isinstance(get_moe_method(QuantConfig(fmt, 1, 1 << 30)), W8A16MoeMethod)


def test_moe_method_rejects_int4():
    """There is no grouped int4 GEMM kernel; the registry must fail loudly."""
    with pytest.raises(ValueError, match="not supported for MoE experts"):
        get_moe_method(QuantConfig.int4_groupwise())


# --------------------------------------------------------------------------- #
# create_weights: parameter layout per scheme
# --------------------------------------------------------------------------- #
def test_create_weights_unquantized():
    layer = ReplicatedLinear(64, 128)
    assert layer.weight.shape == (128, 64)
    assert layer.weight.dtype == torch.float16
    assert not hasattr(layer, "weight_scale_inv")


def test_create_weights_int8_per_channel():
    layer = ReplicatedLinear(64, 128, quant=QuantConfig.int8_per_channel())
    assert layer.weight.shape == (128, 64)
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale_inv.shape == (128, 1)
    assert layer.weight_scale_inv.dtype == torch.float32


def test_create_weights_int8_blockwise():
    layer = ReplicatedLinear(64, 128, quant=QuantConfig.int8_groupwise(group_size=32))
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale_inv.shape == (128, 2)  # 64 / 32


def test_create_weights_fp8_per_channel():
    layer = ReplicatedLinear(64, 128, quant=QuantConfig.fp8_per_channel())
    assert layer.weight.shape == (128, 64)
    assert layer.weight.dtype == torch.uint8  # e4m3 bit pattern container
    assert layer.weight_scale_inv.shape == (128, 1)


def test_create_weights_int4():
    layer = ReplicatedLinear(256, 128, quant=QuantConfig.int4_groupwise(group_size=128))
    assert layer.weight.shape == (128, 32)  # 8 int4 values per int32 word
    assert layer.weight.dtype == torch.int32
    assert layer.weight_scale.shape == (128, 2)
    assert layer.weight_zeros.shape == (128, 2)


def test_create_weights_smoothquant():
    layer = ReplicatedLinear(64, 128, quant=QuantConfig.smoothquant_per_channel())
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale_inv.shape == (128, 1)


def test_moe_create_weights_int8():
    quant = QuantConfig.int8_per_channel()
    block = _StubMoeBlock(quant=quant)
    params = W8A16MoeMethod().create_weights(block)
    assert params["gate_up_proj"].shape == (4, 256, 256)  # [E, 2I, H]
    assert params["down_proj"].shape == (4, 256, 128)  # [E, H, I]
    assert params["gate_up_proj"].dtype == torch.int8
    assert params["gate_up_proj_scale_inv"].shape == (4, 256, 1)
    assert params["down_proj_scale_inv"].shape == (4, 256, 1)


# --------------------------------------------------------------------------- #
# quantize_: fp16 -> low-bit conversion per runtime scheme
# --------------------------------------------------------------------------- #
def test_quantize_int8_per_channel_roundtrip():
    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(QuantConfig.int8_per_channel())
    assert isinstance(layer.quant_method, W8A16LinearMethod)
    assert layer.weight.dtype == torch.int8
    deq = layer.weight.float() * layer.weight_scale_inv
    torch.testing.assert_close(deq, w, rtol=2e-2, atol=2e-3)


def test_quantize_int8_blockwise_roundtrip():
    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(QuantConfig.int8_groupwise(group_size=128))
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale_inv.shape == (128, 2)
    s = layer.weight_scale_inv.repeat_interleave(128, dim=1)
    torch.testing.assert_close(layer.weight.float() * s, w, rtol=2e-2, atol=2e-3)


def test_quantize_fp8_per_channel_roundtrip():
    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(QuantConfig.fp8_per_channel())
    assert isinstance(layer.quant_method, Fp8LinearMethod)
    assert layer.weight.dtype == torch.uint8
    deq = layer.weight.view(torch.float8_e4m3fn).float() * layer.weight_scale_inv
    # e4m3 has 3 mantissa bits, so the tolerance is wider than int8's.
    torch.testing.assert_close(deq, w, rtol=1e-1, atol=1e-2)


def test_quantize_int4_roundtrip():
    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(QuantConfig.int4_groupwise(group_size=128))
    assert isinstance(layer.quant_method, W4A16LinearMethod)
    assert layer.weight.dtype == torch.int32
    deq = _dequant_int4(layer.weight, layer.weight_scale, layer.weight_zeros, 128, 128, 256)
    # int4's quantisation step is ~amax/7 per group, so the honest bound is the
    # same one the kernel test uses.
    torch.testing.assert_close(deq, w, rtol=5e-2, atol=5e-2)


def test_quantize_smoothquant_roundtrip():
    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(QuantConfig.smoothquant_per_channel())
    assert isinstance(layer.quant_method, SmoothQuantLinearMethod)
    assert layer.weight.dtype == torch.int8
    deq = layer.weight.float() * layer.weight_scale_inv
    torch.testing.assert_close(deq, w, rtol=2e-2, atol=2e-3)


def test_quantize_is_idempotent_guard():
    """A layer that already carries quantised weights is left alone."""
    layer = ReplicatedLinear(256, 128)
    _fill_fp16(layer)
    layer.quantize_(QuantConfig.int8_per_channel())
    weight_after_first = layer.weight
    layer.quantize_(QuantConfig.fp8_per_channel())  # must be a no-op
    assert layer.weight is weight_after_first
    assert layer.quant.format == INT8


def test_unquantized_method_cannot_convert():
    layer = ReplicatedLinear(64, 128)
    with pytest.raises(NotImplementedError, match="cannot be computed from fp16"):
        UnquantizedLinearMethod().convert_from_fp16(layer, QuantConfig.int8_per_channel())


def test_moe_convert_from_fp16_int8():
    block = _StubMoeBlock()
    block.experts = nn.ParameterDict(UnquantizedMoeMethod().create_weights(block))
    with torch.no_grad():
        for p in block.experts.values():
            p.copy_(torch.randn_like(p, dtype=torch.float32) * 0.05)
    ref = block.experts["gate_up_proj"].data.float().clone()

    quant = QuantConfig.int8_per_channel()
    W8A16MoeMethod().convert_from_fp16(block, quant)

    assert block.experts["gate_up_proj"].dtype == torch.int8
    assert block.experts["down_proj"].dtype == torch.int8
    deq = block.experts["gate_up_proj"].float() * block.experts["gate_up_proj_scale_inv"]
    torch.testing.assert_close(deq, ref, rtol=2e-2, atol=2e-3)


def test_moe_convert_from_fp16_fp8():
    block = _StubMoeBlock()
    block.experts = nn.ParameterDict(UnquantizedMoeMethod().create_weights(block))
    W8A16MoeMethod().convert_from_fp16(block, QuantConfig.fp8_per_channel())
    assert block.experts["gate_up_proj"].dtype == torch.uint8
    assert block.experts["down_proj_scale_inv"].shape == (4, 256, 1)


def _dequant_int4(qw, scales, zeros, group_size, n, k):
    """Unpack ``[N, K//8]`` int32 nibbles and apply ``(q - zero) * scale``."""
    words = qw.to(torch.int64)
    shifts = 4 * torch.arange(8)
    w = ((words.unsqueeze(-1) >> shifts) & 0xF).flatten(-2).float()[:, :k]
    w = w.unflatten(-1, (k // group_size, group_size))
    return ((w - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)).flatten(-2)


# --------------------------------------------------------------------------- #
# Parameter factories
# --------------------------------------------------------------------------- #
def test_quantize_int8_groupwise_shapes_and_accuracy():
    w = torch.randn(64, 256) * 0.05
    qw, scale = quantize_int8_groupwise(w, group_size=128)
    assert qw.dtype == torch.int8
    assert qw.shape == (64, 256)
    assert scale.shape == (64, 2)
    deq = qw.float() * scale.repeat_interleave(128, dim=1)
    torch.testing.assert_close(deq, w, rtol=1e-2, atol=1e-3)


def test_quantize_int8_groupwise_rejects_uneven_k():
    with pytest.raises(ValueError, match="multiple of group_size"):
        quantize_int8_groupwise(torch.randn(8, 100), group_size=128)


def test_quantize_fp8_per_channel():
    w = torch.randn(64, 128) * 0.05
    qw, scale = quantize_fp8_per_channel(w)
    assert qw.dtype == torch.uint8
    assert scale.shape == (64, 1)
    deq = qw.view(torch.float8_e4m3fn).float() * scale
    torch.testing.assert_close(deq, w, rtol=1e-1, atol=1e-2)


def test_quantize_fp8_per_channel_zero_row():
    """An all-zero channel must not produce a zero scale (division guard)."""
    w = torch.randn(4, 32)
    w[1].zero_()
    qw, scale = quantize_fp8_per_channel(w)
    assert torch.isfinite(scale).all()
    assert (scale > 0).all()
    assert qw[1].view(torch.float8_e4m3fn).float().abs().max() == 0


# --------------------------------------------------------------------------- #
# QuantConfig: runtime schemes and shard alignment
# --------------------------------------------------------------------------- #
def test_for_runtime_scheme_covers_every_registered_name():
    expectations = {
        "int8": (INT8, 1 << 30, False),
        "int8-blockwise": (INT8, 128, False),
        "fp8": (FP8, 1 << 30, False),
        "int4": (INT4, 128, False),
        "smoothquant": (SMOOTHQUANT, 1 << 30, True),
    }
    assert set(expectations) == set(RUNTIME_SCHEMES)
    for name, (fmt, group_k, is_dynamic) in expectations.items():
        quant = QuantConfig.for_runtime_scheme(name)
        assert quant.format == fmt
        assert quant.group_k == group_k
        assert quant.is_dynamic == is_dynamic


def test_for_runtime_scheme_rejects_unknown():
    with pytest.raises(ValueError, match="unknown runtime quantisation"):
        QuantConfig.for_runtime_scheme("int2")


def test_shard_is_aligned_per_channel_always():
    """One scale per output row: no block for a TP shard to cut."""
    quant = QuantConfig.int8_per_channel()
    assert quant.shard_is_aligned(96)
    assert quant.shard_is_aligned(1)


def test_shard_is_aligned_blockwise():
    for quant in (
        QuantConfig.int8_groupwise(group_size=128),
        QuantConfig.int4_groupwise(group_size=128),
        QuantConfig(FP8, 128, 128),
    ):
        assert quant.shard_is_aligned(256)
        assert not quant.shard_is_aligned(96)


# --------------------------------------------------------------------------- #
# End-to-end: a quantised layer's forward (needs the Triton kernels)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_replicated_linear_int8_forward_matches_reference():
    layer = ReplicatedLinear(256, 128).cuda()
    w = _fill_fp16(layer)
    layer.quantize_(QuantConfig.int8_per_channel())
    x = torch.randn(8, 256, device="cuda", dtype=torch.float16) * 0.5

    out = layer(x)

    ref = x.float() @ (layer.weight.float() * layer.weight_scale_inv).T
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    # The reference dequantises the stored int8, so the layer must sit within
    # int8's rounding noise of the original fp16 product.
    fp16_ref = x.float() @ w.cuda().T
    torch.testing.assert_close(out.float(), fp16_ref, rtol=2e-2, atol=2e-2)
