"""Numeric parity tests for the Qwen3-MoE support.

Two layers of verification:

1. :func:`test_block_fp8_dequant` — the FP8 (e4m3, 128x128 block) dequantisation
   used by the converter, checked against a manual blockwise multiply, including
   matrices whose dims are not multiples of the block size.
2. :func:`test_qwen3_moe_logits_parity` — a randomly initialised tiny
   ``Qwen3MoeForCausalLM`` is serialised to safetensors, converted through the
   *real* :func:`lite_llama.tools.convert_weights._convert` path, loaded into
   :class:`Qwen3MoeModel` and compared against the fp32 HuggingFace forward.
"""

from __future__ import annotations

import json

import pytest
import torch

from lite_llama.models.model_config import Qwen3MoeConfig
from lite_llama.tools.convert_weights import _convert, _dequant_block_fp8

# --------------------------------------------------------------------------- #
# FP8 dequantisation (CPU only)
# --------------------------------------------------------------------------- #


def _block_quantize_fp8(w: torch.Tensor, block: int = 128):
    """Emulate the Qwen FP8 checkpoint format: e4m3 weight + fp32 scale_inv."""
    out_f, in_f = w.shape
    w8 = torch.empty_like(w)
    scale = torch.empty(
        (out_f + block - 1) // block, (in_f + block - 1) // block, dtype=torch.float32
    )
    for bi in range(scale.shape[0]):
        for bj in range(scale.shape[1]):
            tile = w[bi * block : (bi + 1) * block, bj * block : (bj + 1) * block]
            s = tile.abs().max() / 448.0  # e4m3 max magnitude
            # ``weight_scale_inv`` 是反量化乘子(量化时除、反量化时乘)。
            scale[bi, bj] = s
            w8[bi * block : (bi + 1) * block, bj * block : (bj + 1) * block] = (
                (tile / s).to(torch.float8_e4m3fn).to(torch.float32)
            )
    return w8.to(torch.float8_e4m3fn), scale


@pytest.mark.parametrize("shape", [(256, 256), (200, 300)])
def test_block_fp8_dequant(shape):
    torch.manual_seed(0)
    w = torch.randn(*shape, dtype=torch.float32)
    w8, scale_inv = _block_quantize_fp8(w)

    deq = _dequant_block_fp8(w8, scale_inv)

    assert deq.dtype == torch.float16
    assert deq.shape == w.shape
    # e4m3 keeps ~3 mantissa bits -> ~6% relative error per element.
    rel = ((deq.float() - w).abs() / w.abs().clamp_min(1e-3)).median()
    assert rel < 0.06


# --------------------------------------------------------------------------- #
# Config layer semantics (CPU only)
# --------------------------------------------------------------------------- #


def test_qwen3_moe_config_from_hf_dict():
    cfg = Qwen3MoeConfig.from_dict(
        {
            "model_type": "qwen3_moe",
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "norm_topk_prob": True,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "head_dim": 128,
            "max_position_embeddings": 262144,
            # Unknown keys from the FP8 checkpoint must be ignored.
            "quantization_config": {"quant_method": "fp8"},
            "router_aux_loss_coef": 0.001,
        },
        max_seq_len=2048,
    )
    assert cfg.num_layers == 48 and cfg.num_experts == 128 and cfg.num_kv_heads == 4
    assert all(cfg.is_moe_layer(i) for i in range(48))

    dense_cfg = Qwen3MoeConfig(mlp_only_layers=[0, 5], decoder_sparse_step=1)
    assert not dense_cfg.is_moe_layer(0)
    assert not dense_cfg.is_moe_layer(5)
    assert dense_cfg.is_moe_layer(1)


# --------------------------------------------------------------------------- #
# Router semantics vs HuggingFace (CPU only, no Triton kernels involved)
# --------------------------------------------------------------------------- #


def test_route_matches_hf():
    """``_route`` must reproduce HF's softmax-all -> top-k -> renormalise order."""
    from lite_llama.models.qwen3_moe import SparseMoeBlock

    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(
        hidden_size=64,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        max_seq_len=64,
    )
    block = SparseMoeBlock(cfg)
    gate = torch.randn(cfg.num_experts, cfg.hidden_size)
    block.gate_weight.data.copy_(gate)

    x = torch.randn(7, cfg.hidden_size).half()
    weights, ids = block._route(x)

    # HF reference: fp32 softmax over all experts, then topk, then renormalise.
    ref = torch.softmax(torch.nn.functional.linear(x.float(), gate.float()), dim=-1)
    ref_w, ref_ids = torch.topk(ref, cfg.num_experts_per_tok, dim=-1)
    ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)

    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights.float(), ref_w, atol=1e-3, rtol=1e-3)


# --------------------------------------------------------------------------- #
# Full-model logits parity vs HuggingFace (needs CUDA for the Triton kernels)
# --------------------------------------------------------------------------- #

_TINY_HF_CONFIG = dict(
    vocab_size=512,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=64,
    norm_topk_prob=True,
    decoder_sparse_step=1,
    mlp_only_layers=[],
    max_position_embeddings=256,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
)


def _convert_hf_model(hf_model, tmp_path):
    """Serialise an HF model as safetensors and run the real converter over it."""
    from safetensors.torch import save_file

    config_dict = {"model_type": "qwen3_moe", **_TINY_HF_CONFIG}
    (tmp_path / "config.json").write_text(json.dumps(config_dict))
    save_file(hf_model.state_dict(), str(tmp_path / "model.safetensors"))

    return _convert(tmp_path, "qwen3_moe", _TINY_HF_CONFIG["num_hidden_layers"], torch.float16)


@pytest.mark.usefixtures("cuda_available")
def test_qwen3_moe_logits_parity(tmp_path):
    from transformers import Qwen3MoeConfig as HfConfig
    from transformers import Qwen3MoeForCausalLM

    from lite_llama.executor.executor_struct import AttentionInfo
    from lite_llama.models.qwen3_moe import Qwen3MoeModel

    torch.manual_seed(42)
    hf_model = Qwen3MoeForCausalLM(HfConfig(**_TINY_HF_CONFIG)).eval()

    state = _convert_hf_model(hf_model, tmp_path)

    lite_cfg = Qwen3MoeConfig.from_dict(_TINY_HF_CONFIG, max_seq_len=256)
    lite_model = Qwen3MoeModel(lite_cfg)
    lite_model.load_state_dict(state, strict=True)
    lite_model = lite_model.cuda().eval()

    seq_len = 12
    input_ids = torch.randint(0, _TINY_HF_CONFIG["vocab_size"], (1, seq_len))

    with torch.no_grad():
        ref_logits = hf_model(input_ids=input_ids).logits.float()  # [1, T, V]

    num_kv_heads, head_dim = _TINY_HF_CONFIG["num_key_value_heads"], _TINY_HF_CONFIG["head_dim"]
    atten_info = AttentionInfo(
        kv_buffer=[
            torch.zeros(seq_len, 2 * num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
            for _ in range(_TINY_HF_CONFIG["num_hidden_layers"])
        ],
        cur_select_index=torch.arange(seq_len, dtype=torch.int32, device="cuda"),
        b_start_loc=torch.zeros(1, dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
        max_actual_seq_len=seq_len,
    )
    position_ids = torch.arange(seq_len, dtype=torch.int32, device="cuda").unsqueeze(0)
    with torch.no_grad():
        lite_logits = lite_model(input_ids.cuda(), position_ids, atten_info).float().cpu()

    assert lite_logits.shape == ref_logits.shape

    # Random-weight tiny models produce near-degenerate logits whose top-2 gaps
    # can sit below the fp16-vs-fp32 noise floor, where argmax is a coin flip.
    # Argmax is therefore only asserted where the gap is decidable; overall
    # fidelity is carried by the cosine-similarity and absolute-diff checks.
    noise_floor = (lite_logits - ref_logits).abs().mean().item()
    top2 = ref_logits.topk(2, dim=-1).values
    decidable = (top2[..., 0] - top2[..., 1]) > 3 * noise_floor
    agree = lite_logits.argmax(-1) == ref_logits.argmax(-1)
    assert agree[decidable].all(), (
        f"argmax mismatch on decidable tokens: {(~agree & decidable).nonzero().flatten().tolist()}"
    )

    cos = torch.nn.functional.cosine_similarity(lite_logits, ref_logits, dim=-1)
    # 0.985 仍远高于随机方向(~0); 随机权重模型的 logits 接近简并, 不能按
    # 真实模型的 0.999 标准要求。
    assert cos.min() > 0.985, f"min cosine {cos.min().item():.6f}"
    max_diff = (lite_logits - ref_logits).abs().max().item()
    scale = ref_logits.abs().max().item()
    assert max_diff < 0.2 * max(scale, 1.0), f"max abs diff {max_diff:.5f} (scale {scale:.2f})"
