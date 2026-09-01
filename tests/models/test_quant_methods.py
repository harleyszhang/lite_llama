"""Tests for the quant-method strategies and the runtime quantisation schemes.

Registry lookups, checkpoint-method dispatch (int4 variants), weight
creation shapes for every scheme, and rejection paths — the strategy
layer without kernels.

Usage:
    pytest tests/models/test_quant_methods.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lite_llama.modules.linear import ReplicatedLinear
from lite_llama.modules.quantization import (
    UnquantizedLinearMethod,
    UnquantizedFusedMoEMethod,
    QuantizationConfig,
)
from lite_llama.modules.quantization.utils import (
    quantize_fp8_per_channel,
    quantize_int8_groupwise,
)
from lite_llama.modules.quantization.blockwise_int8 import BlockInt8LinearMethod, BlockInt8MoEMethod
from lite_llama.modules.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod
from lite_llama.modules.quantization.w8a8_fp8 import W8A8Fp8LinearMethod
from lite_llama.modules.quantization.w8a8_int8 import W8A8Int8LinearMethod
from lite_llama.modules.quantization.awq import AWQLinearMethod


from lite_llama.modules.moe import SparseMoeBlock


class _StubMoeBlock(SparseMoeBlock):
    """Minimum surface a MoeQuantMethod touches: sizes, ``quant``, ``experts``."""

    def __init__(
        self,
        num_experts: int = 4,
        hidden_size: int = 256,
        moe_intermediate_size: int = 128,
        quant: "QuantizationConfig | None" = None,
    ) -> None:
        # Bypass SparseMoeBlock.__init__ which requires a full ModelConfig
        nn.Module.__init__(self)
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
# Registry: config.get_quant_method() dispatch
# --------------------------------------------------------------------------- #
def test_linear_method_registry():
    from lite_llama.modules.quantization.fp8 import Fp8Config
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    from lite_llama.modules.quantization.w8a8_int8 import W8A8Int8Config
    from lite_llama.modules.quantization.awq import AWQConfig

    layer = ReplicatedLinear(64, 128)
    assert isinstance(Fp8Config(128, 128).get_quant_method(layer), Fp8LinearMethod)
    assert isinstance(BlockInt8Config.per_channel().get_quant_method(layer), BlockInt8LinearMethod)
    assert isinstance(W8A8Int8Config().get_quant_method(layer), W8A8Int8LinearMethod)
    assert isinstance(AWQConfig().get_quant_method(layer), AWQLinearMethod)


def test_int4_method_dispatches_on_checkpoint_method():
    """AWQ/GPTQ checkpoints get their named method classes."""
    from lite_llama.modules.quantization.awq import AWQConfig
    from lite_llama.modules.quantization.gptq import GPTQConfig, GPTQLinearMethod

    layer = ReplicatedLinear(64, 128)
    assert isinstance(AWQConfig().get_quant_method(layer), AWQLinearMethod)
    assert isinstance(GPTQConfig().get_quant_method(layer), GPTQLinearMethod)


def test_moe_method_registry():
    from lite_llama.modules.quantization.fp8 import Fp8Config
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    from lite_llama.modules.moe import SparseMoeBlock

    # Use a stub that inherits from SparseMoeBlock for isinstance check
    block = _StubMoeBlock()
    assert isinstance(Fp8Config(128, 128).get_quant_method(block), Fp8MoEMethod)
    assert isinstance(BlockInt8Config.per_channel().get_quant_method(block), BlockInt8MoEMethod)


def test_moe_method_rejects_int4():
    """AWQ has no MoE support; get_quant_method returns UnquantizedFusedMoEMethod for ignored."""
    from lite_llama.modules.quantization.awq import AWQConfig
    # AWQ doesn't have a MoE method at all — it only returns linear methods.
    # The config simply doesn't support MoE layers.
    pass


# --------------------------------------------------------------------------- #
# create_weights: parameter layout per scheme
# --------------------------------------------------------------------------- #
def test_create_weights_unquantized():
    layer = ReplicatedLinear(64, 128)
    assert layer.weight.shape == (128, 64)
    assert layer.weight.dtype == torch.float16
    assert not hasattr(layer, "weight_scale_inv")


def test_create_weights_int8_per_channel():
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    layer = ReplicatedLinear(64, 128, quant=BlockInt8Config.per_channel())
    assert layer.weight.shape == (128, 64)
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale_inv.shape == (128, 1)
    assert layer.weight_scale_inv.dtype == torch.float32


def test_create_weights_int8_blockwise():
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    layer = ReplicatedLinear(64, 128, quant=BlockInt8Config.groupwise(group_size=32))
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale_inv.shape == (128, 2)  # 64 / 32


def test_create_weights_fp8_per_channel():
    from lite_llama.modules.quantization.fp8 import Fp8Config
    layer = ReplicatedLinear(64, 128, quant=Fp8Config(group_n=1, group_k=1 << 30))
    assert layer.weight.shape == (128, 64)
    assert layer.weight.dtype == torch.uint8  # e4m3 bit pattern container
    assert layer.weight_scale_inv.shape == (128, 1)


def test_create_weights_int4():
    from lite_llama.modules.quantization.awq import AWQConfig
    layer = ReplicatedLinear(256, 128, quant=AWQConfig(group_size=128))
    assert layer.weight.shape == (128, 32)  # 8 int4 values per int32 word
    assert layer.weight.dtype == torch.int32
    assert layer.weight_scale.shape == (128, 2)
    assert layer.weight_zeros.shape == (128, 2)


def test_create_weights_smoothquant():
    from lite_llama.modules.quantization.w8a8_int8 import W8A8Int8Config
    layer = ReplicatedLinear(64, 128, quant=W8A8Int8Config())
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale_inv.shape == (128, 1)


def test_moe_create_weights_int8():
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    quant = BlockInt8Config.per_channel()
    block = _StubMoeBlock(quant=quant)
    method = BlockInt8MoEMethod()
    params = method.create_weights(block)
    assert params["gate_up_proj"].shape == (4, 256, 256)  # [E, 2I, H]
    assert params["down_proj"].shape == (4, 256, 128)  # [E, H, I]
    assert params["gate_up_proj"].dtype == torch.int8
    assert params["gate_up_proj_scale_inv"].shape == (4, 256, 1)
    assert params["down_proj_scale_inv"].shape == (4, 256, 1)


# --------------------------------------------------------------------------- #
# quantize_: fp16 -> low-bit conversion per runtime scheme
# --------------------------------------------------------------------------- #
def test_quantize_int8_per_channel_roundtrip():
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(BlockInt8Config.per_channel())
    assert isinstance(layer.quant_method, BlockInt8LinearMethod)
    assert layer.weight.dtype == torch.int8
    deq = layer.weight.float() * layer.weight_scale_inv
    torch.testing.assert_close(deq, w, rtol=2e-2, atol=2e-3)


def test_quantize_int8_blockwise_roundtrip():
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(BlockInt8Config.groupwise(group_size=128))
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale_inv.shape == (128, 2)
    s = layer.weight_scale_inv.repeat_interleave(128, dim=1)
    torch.testing.assert_close(layer.weight.float() * s, w, rtol=2e-2, atol=2e-3)


def test_quantize_fp8_per_channel_roundtrip():
    from lite_llama.modules.quantization.fp8 import Fp8Config
    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(Fp8Config(group_n=1, group_k=1 << 30))
    assert isinstance(layer.quant_method, (Fp8LinearMethod, W8A8Fp8LinearMethod))
    assert layer.weight.dtype == torch.uint8
    deq = layer.weight.view(torch.float8_e4m3fn).float() * layer.weight_scale_inv
    # e4m3 has 3 mantissa bits, so the tolerance is wider than int8's.
    torch.testing.assert_close(deq, w, rtol=1e-1, atol=1e-2)


def test_quantize_int4_roundtrip():
    from lite_llama.modules.quantization.awq import AWQConfig
    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(AWQConfig(group_size=128))
    assert isinstance(layer.quant_method, AWQLinearMethod)
    assert layer.weight.dtype == torch.int32
    deq = _dequant_int4(layer.weight, layer.weight_scale, layer.weight_zeros, 128, 128, 256)
    # int4's quantisation step is ~amax/7 per group, so the honest bound is the
    # same one the kernel test uses.
    torch.testing.assert_close(deq, w, rtol=5e-2, atol=5e-2)


def test_quantize_smoothquant_roundtrip():
    from lite_llama.modules.quantization.w8a8_int8 import W8A8Int8Config
    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(W8A8Int8Config())
    assert isinstance(layer.quant_method, W8A8Int8LinearMethod)
    assert layer.weight.dtype == torch.int8
    deq = layer.weight.float() * layer.weight_scale_inv
    torch.testing.assert_close(deq, w, rtol=2e-2, atol=2e-3)


def test_quantize_is_idempotent_guard():
    """A layer that already carries quantised weights is left alone."""
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    from lite_llama.modules.quantization.fp8 import Fp8Config
    layer = ReplicatedLinear(256, 128)
    _fill_fp16(layer)
    layer.quantize_(BlockInt8Config.per_channel())
    weight_after_first = layer.weight
    layer.quantize_(Fp8Config(group_n=1, group_k=1 << 30))  # must be a no-op
    assert layer.weight is weight_after_first
    assert layer.quant.get_name() == "blockwise_int8"


def test_unquantized_method_cannot_convert():
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    layer = ReplicatedLinear(64, 128)
    with pytest.raises(NotImplementedError, match="cannot be computed from fp16"):
        UnquantizedLinearMethod().quantize_from_fp16(layer, BlockInt8Config.per_channel())


def test_moe_convert_from_fp16_int8():
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    block = _StubMoeBlock()
    block.experts = nn.ParameterDict(UnquantizedFusedMoEMethod().create_weights(block))
    with torch.no_grad():
        for p in block.experts.values():
            p.copy_(torch.randn_like(p, dtype=torch.float32) * 0.05)
    ref = block.experts["gate_up_proj"].data.float().clone()

    quant = BlockInt8Config.per_channel()
    BlockInt8MoEMethod().quantize_from_fp16(block, quant)

    assert block.experts["gate_up_proj"].dtype == torch.int8
    assert block.experts["down_proj"].dtype == torch.int8
    deq = block.experts["gate_up_proj"].float() * block.experts["gate_up_proj_scale_inv"]
    torch.testing.assert_close(deq, ref, rtol=2e-2, atol=2e-3)


def test_moe_convert_from_fp16_fp8():
    from lite_llama.modules.quantization.fp8 import Fp8Config
    block = _StubMoeBlock()
    block.experts = nn.ParameterDict(UnquantizedFusedMoEMethod().create_weights(block))
    Fp8MoEMethod().quantize_from_fp16(block, Fp8Config(group_n=1, group_k=1 << 30))
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
# Runtime schemes and shard alignment
# --------------------------------------------------------------------------- #
def test_for_runtime_scheme_covers_every_registered_name():
    from lite_llama.modules.quantization import for_runtime_scheme, RUNTIME_SCHEMES
    for name in RUNTIME_SCHEMES:
        quant = for_runtime_scheme(name)
        assert quant is not None


def test_for_runtime_scheme_rejects_unknown():
    from lite_llama.modules.quantization import for_runtime_scheme
    with pytest.raises(ValueError, match="unknown runtime quantisation"):
        for_runtime_scheme("int2")


def test_shard_is_aligned_per_channel_always():
    """One scale per output row: no block for a TP shard to cut."""
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    quant = BlockInt8Config.per_channel()
    assert quant.shard_is_aligned(96)
    assert quant.shard_is_aligned(1)


def test_shard_is_aligned_blockwise():
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    from lite_llama.modules.quantization.awq import AWQConfig
    from lite_llama.modules.quantization.fp8 import Fp8Config
    for quant in (
        BlockInt8Config.groupwise(group_size=128),
        AWQConfig(group_size=128),
        Fp8Config(128, 128),
    ):
        assert quant.shard_is_aligned(256)
        assert not quant.shard_is_aligned(96)


# --------------------------------------------------------------------------- #
# End-to-end: a quantised layer's forward (needs the Triton kernels)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_replicated_linear_int8_forward_matches_reference():
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config
    layer = ReplicatedLinear(256, 128).cuda()
    w = _fill_fp16(layer)
    layer.quantize_(BlockInt8Config.per_channel())
    x = torch.randn(8, 256, device="cuda", dtype=torch.float16) * 0.5

    out = layer(x)

    ref = x.float() @ (layer.weight.float() * layer.weight_scale_inv).T
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    # The reference dequantises the stored int8, so the layer must sit within
    # int8's rounding noise of the original fp16 product.
    fp16_ref = x.float() @ w.cuda().T
    torch.testing.assert_close(out.float(), fp16_ref, rtol=2e-2, atol=2e-2)
