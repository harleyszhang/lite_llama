"""DeepSeek-V2 numerical alignment: the MoE layer against HuggingFace.

The attention layer's alignment lives in ``tests/tools/test_harness.py``; this
file pins the part that only the weight-mapping work exposed: the MoE layer's
forward semantics — the router that scales instead of renormalises, the shared
expert added unscaled, and the expert weights arriving through the stacked
checkpoint layout — against ``DeepseekV2Moe`` with identical parameters.

Usage:
    pytest tests/models/test_deepseek_v2.py
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from rapid_llm.models.config import ModelConfig

#: V2-Lite's relations at test size: one dense layer, then MoE with two shared
#: experts and a 2.5 routed scaling factor. ``norm_topk_prob`` is False — the
#: real config's spelling, and the behaviour half of this file exists to pin.
_BODY = {
    "model_type": "deepseek_v2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "n_shared_experts": 2,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "routed_scaling_factor": 2.5,
    "first_k_dense_replace": 1,
    "norm_topk_prob": False,
    "kv_lora_rank": 16,
    "q_lora_rank": 32,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 64,
    "v_head_dim": 32,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


def _loaded_pair(tmp_path, seed: int = 0):
    """Build the HF model, checkpoint it, and load the same weights into rapid_llm.

    Returns:
        ``(hf_model, lite_model, config)`` with both sides in eval mode on
        CUDA, parameters parity-verified by :func:`load_weights`'s coverage
        check on the way in.
    """
    from transformers import DeepseekV2ForCausalLM

    from rapid_llm.executor.loader import materialise_parameters
    from rapid_llm.executor.weight_utils import hf_weights_iterator
    from rapid_llm.models.registry import ModelRegistry

    (tmp_path / "config.json").write_text(json.dumps(_BODY))
    config = ModelConfig.from_pretrained(tmp_path, max_seq_len=128)
    torch.manual_seed(seed)
    hf_model = DeepseekV2ForCausalLM(config.hf_config).eval()
    state = {key: value.detach().clone() for key, value in hf_model.state_dict().items()}
    save_file(state, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})

    model = ModelRegistry.resolve("deepseek_v2").load_class()(config)
    materialise_parameters(model, "cuda", dtype=config.dtype)
    model.load_weights(hf_weights_iterator(tmp_path, dequant_dtype=config.dtype))
    return hf_model, model.eval(), config


def test_router_scales_instead_of_renormalising(tmp_path):
    """``norm_topk_prob=False`` semantics: raw softmax topk weights x factor.

    DeepSeek's router has no renormalise step at all (that is qwen3-moe's
    behaviour); the topk scores ride straight into the routed_scaling_factor
    multiply, so the weights sum to ``factor`` times the softmax mass of the
    selected experts — never to 1.
    """
    from rapid_llm.modules.moe import SparseMoeBlock

    (tmp_path / "config.json").write_text(json.dumps(_BODY))
    config = ModelConfig.from_pretrained(tmp_path, max_seq_len=128)
    block = SparseMoeBlock(config)
    torch.manual_seed(0)
    block.gate_weight.data.copy_(torch.randn(config.num_experts, config.hidden_size))

    x = torch.randn(9, config.hidden_size, dtype=config.dtype)
    weights, ids = block._route(x)

    # HF DeepseekV2TopkRouter: fp32 linear, fp32 softmax, topk, x factor.
    gate = block.gate_weight.float()
    scores = torch.softmax(torch.nn.functional.linear(x.float(), gate), dim=-1)
    ref_w, ref_ids = torch.topk(scores, config.num_experts_per_tok, dim=-1)
    ref_w = ref_w * config.routed_scaling_factor

    assert torch.equal(ids, ref_ids)
    # The fp32 values are identical (same ops, same inputs, CPU); the block
    # returns them cast to the activation dtype, so compare after the same
    # cast rather than demanding fp32 tolerance of a bf16 tensor.
    assert torch.equal(weights, ref_w.to(weights.dtype))


@pytest.mark.usefixtures("cuda_available")
def test_moe_layer_agrees_with_transformers(tmp_path):
    """The whole MoE forward — routing, scaling, shared expert — vs HF.

    Parity tests prove the weights land; this proves they are *used* the same
    way: the expert gather must apply the scaled (not renormalised) weights,
    and the shared expert joins after the routed half, unscaled.
    """
    hf_model, model, _ = _loaded_pair(tmp_path)
    layer = 1  # layer 0 is dense; layer 1 is the MoE one
    hf_moe = hf_model.model.layers[layer].mlp.to(model.layers[layer].mlp.gate_weight.dtype).cuda()
    lite_moe = model.layers[layer].mlp

    torch.manual_seed(1)
    x = torch.randn(3, 7, 64, dtype=lite_moe.gate_weight.dtype, device="cuda")
    with torch.no_grad():
        expected = hf_moe(x.clone())
        actual = lite_moe(x.clone())

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.usefixtures("cuda_available")
def test_shared_expert_output_is_unscaled(tmp_path):
    """A zero router score must still leave the shared expert's contribution.

    With every routed weight forced to (near) zero the block degenerates to
    the shared MLP alone — pinning that the scaling factor never touches the
    shared half, and that the shared expert really is wired in.
    """
    hf_model, model, _ = _loaded_pair(tmp_path)
    lite_moe = model.layers[1].mlp
    hf_moe = hf_model.model.layers[1].mlp.to(lite_moe.gate_weight.dtype).cuda()

    # Zero the router: topk weights become uniform (1/E each, sum 2/E after
    # the factor with E=4 -> 1.25 total). Instead of fighting the softmax,
    # zero the expert weights themselves: the routed half then contributes
    # exactly zero everywhere and the output is the shared expert's alone.
    lite_moe.experts["gate_up_proj"].data.zero_()
    lite_moe.experts["down_proj"].data.zero_()

    torch.manual_seed(2)
    x = torch.randn(5, 64, dtype=lite_moe.gate_weight.dtype, device="cuda")
    with torch.no_grad():
        actual = lite_moe(x.clone())
        # Same surgery on the HF side: its experts keep one stacked parameter
        # per projection, not one module per expert. transformers >= 5 routes
        # with a 3-D "[batch, seq, hidden]" shape, so the row-major test input
        # rides in as a single sequence.
        hf_moe.experts.gate_up_proj.data.zero_()
        hf_moe.experts.down_proj.data.zero_()
        expected = hf_moe(x.clone().unsqueeze(0)).squeeze(0)

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=2e-2,
        rtol=2e-2,
    )
