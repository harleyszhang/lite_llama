"""Tests for the single-layer harness: assembly, weight routing, bookkeeping.

The harness builds only the requested layer (negative indices count
from the end), routes MoE layers to the routed MLP, fuses QKV with
block ids, and reports what it ran.

Usage:
    pytest tests/tools/test_harness.py
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from rapid_llm.models.config import ModelConfig
from rapid_llm.tools.harness import (
    Diff,
    LayerReport,
    ModuleTimer,
    OpTiming,
    SingleLayerCache,
    SingleLayerHarness,
    hf_decoder_layer_class,
)

_DENSE = {
    "model_type": "qwen3",
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 256,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}

_MOE = {
    **_DENSE,
    "model_type": "qwen3_moe",
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
}

#: DeepSeek-V2-Lite geometry, shrunk to test size but keeping every relation
#: that makes MLA MLA: rope only on the pe slice, equal-width kv_b halves, the
#: mscale yarn pair, and one dense layer before the routed stack. ``factor`` is
#: the true ratio of the two position bounds, so AutoConfig round-trips without
#: the mismatch warning; ``mscale != mscale_all_dim`` keeps the cos/sin yarn
#: ratio non-trivial (it cancels to 1.0 when the pair matches). The rope width
#: is the real 64: the decode kernel's op contract carries no dimension
#: argument, so the latent row splits at that fixed boundary.
_MLA = {
    "model_type": "deepseek_v2",
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "n_shared_experts": 0,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "routed_scaling_factor": 2.5,
    "first_k_dense_replace": 1,
    # V2-Lite's real spelling: DeepSeek's router does not renormalise the
    # topk weights (that is qwen3-moe's norm_topk_prob=True behaviour) — the
    # softmax scores ride straight into the routed_scaling_factor multiply.
    "norm_topk_prob": False,
    "kv_lora_rank": 16,
    "q_lora_rank": None,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 64,
    "v_head_dim": 32,
    "attention_bias": False,
    "max_position_embeddings": 256,
    "rope_theta": 10000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 64,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 0.707,
    },
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


def _config(tmp_path, body: dict, **overrides) -> ModelConfig:
    """Round-trip a config.json through AutoConfig, as the loader does."""
    (tmp_path / "config.json").write_text(json.dumps({**body, **overrides}))
    return ModelConfig.from_pretrained(tmp_path, max_seq_len=128)


def _harness(tmp_path, body: dict = _DENSE, layer_index: int = 1) -> SingleLayerHarness:
    return SingleLayerHarness(_config(tmp_path, body), layer_index, device="cpu")


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def test_builds_only_the_requested_layer(tmp_path):
    """The skeleton is meta; the layer that comes back has real storage."""
    harness = _harness(tmp_path)
    assert harness.layer_index == 1
    assert all(not p.is_meta for p in harness.layer.parameters())
    assert harness.param_bytes() > 0


def test_negative_index_counts_from_the_end(tmp_path):
    assert _harness(tmp_path, layer_index=-1).layer_index == 2


def test_index_outside_the_stack_is_rejected(tmp_path):
    with pytest.raises(IndexError, match="3-layer stack"):
        _harness(tmp_path, layer_index=3)


def test_moe_layer_gets_the_routed_mlp(tmp_path):
    """The per-layer MLP choice comes from the model class, not from the harness."""
    from rapid_llm.modules import SparseMoeBlock

    assert isinstance(_harness(tmp_path, _MOE).layer.mlp, SparseMoeBlock)


# --------------------------------------------------------------------------- #
# Key translation: which checkpoint key reaches which parameter block
# --------------------------------------------------------------------------- #
def test_translate_fuses_qkv_with_block_ids(tmp_path):
    harness = _harness(tmp_path)
    assert harness.translate("model.layers.1.self_attn.q_proj.weight") == (
        "self_attn.qkv_proj.weight",
        0,
    )
    assert harness.translate("model.layers.1.self_attn.k_proj.weight") == (
        "self_attn.qkv_proj.weight",
        1,
    )
    assert harness.translate("model.layers.1.self_attn.v_proj.weight") == (
        "self_attn.qkv_proj.weight",
        2,
    )


def test_translate_flattens_norm_names(tmp_path):
    harness = _harness(tmp_path)
    assert harness.translate("model.layers.1.input_layernorm.weight") == (
        "input_layernorm_weight",
        None,
    )
    assert harness.translate("model.layers.1.self_attn.q_norm.weight") == (
        "self_attn.q_norm_weight",
        None,
    )


def test_translate_drops_every_other_key(tmp_path):
    """Another layer, and anything outside the stack, belong to no parameter here."""
    harness = _harness(tmp_path)
    assert harness.translate("model.layers.0.self_attn.q_proj.weight") is None
    assert harness.translate("model.layers.2.mlp.down_proj.weight") is None
    assert harness.translate("model.embed_tokens.weight") is None
    assert harness.translate("model.norm.weight") is None
    assert harness.translate("lm_head.weight") is None


def test_translate_stacks_moe_experts(tmp_path):
    """The expert rule needs the layer path in front of ``.mlp.experts`` to match.

    Filtering before translating would leave this key unrecognised, which is the whole
    reason the harness translates first and filters on the parameter name.
    """
    harness = _harness(tmp_path, _MOE)
    assert harness.translate("model.layers.1.mlp.experts.2.gate_proj.weight") == (
        "mlp.experts.gate_up_proj",
        (2, 0),
    )
    assert harness.translate("model.layers.1.mlp.experts.2.up_proj.weight") == (
        "mlp.experts.gate_up_proj",
        (2, 1),
    )
    assert harness.translate("model.layers.1.mlp.experts.2.down_proj.weight") == (
        "mlp.experts.down_proj",
        (2, 2),
    )
    assert harness.translate("model.layers.1.mlp.gate.weight") == ("mlp.gate_weight", None)


def test_checkpoint_prefix_names_this_layer(tmp_path):
    assert _harness(tmp_path).checkpoint_prefix() == "model.layers.1."


# --------------------------------------------------------------------------- #
# Weight loading
# --------------------------------------------------------------------------- #
def test_mirrored_weights_land_in_the_right_blocks(tmp_path):
    """A HF layer's own state dict must fill the fused parameters block by block.

    This is the load path a checkpoint takes, exercised without a checkpoint: if the
    fused-QKV block order were wrong, the layer would still run and every generation
    from it would be garbage.
    """
    from rapid_llm.tools.harness import HFLayerReference

    config = _config(tmp_path, _DENSE)
    harness = SingleLayerHarness(config, 1, device="cpu")
    reference = HFLayerReference(config, 1, device="cpu")
    harness.load_state_dict(reference.state_dict(), source="test")

    hf = dict(reference.state_dict())
    fused = harness.layer.self_attn.qkv_proj.weight
    q_size, kv_size = harness.layer.self_attn.q_size, harness.layer.self_attn.kv_size
    torch.testing.assert_close(fused[:q_size], hf["self_attn.q_proj.weight"])
    torch.testing.assert_close(fused[q_size : q_size + kv_size], hf["self_attn.k_proj.weight"])
    torch.testing.assert_close(fused[q_size + kv_size :], hf["self_attn.v_proj.weight"])

    gate_up = harness.layer.mlp.gate_up_proj.weight
    rows = hf["mlp.gate_proj.weight"].shape[0]
    torch.testing.assert_close(gate_up[:rows], hf["mlp.gate_proj.weight"])
    torch.testing.assert_close(gate_up[rows:], hf["mlp.up_proj.weight"])

    torch.testing.assert_close(harness.layer.input_layernorm_weight, hf["input_layernorm.weight"])
    assert harness.weights == "test"


def test_a_missing_key_fails_loudly(tmp_path):
    """Coverage checking is why a one-layer load goes through ``weights.load_weights``."""
    from rapid_llm.tools.harness import HFLayerReference

    config = _config(tmp_path, _DENSE)
    harness = SingleLayerHarness(config, 1, device="cpu")
    state = HFLayerReference(config, 1, device="cpu").state_dict()
    state.pop("self_attn.v_proj.weight")

    with pytest.raises(ValueError, match="partially written"):
        harness.load_state_dict(state, source="test")


def test_randomise_leaves_norms_at_one(tmp_path):
    """Norms start at one so activations stay in the range the kernels see in service."""
    harness = _harness(tmp_path)
    harness.randomise(seed=3)
    assert torch.equal(
        harness.layer.input_layernorm_weight, torch.ones_like(harness.layer.input_layernorm_weight)
    )
    assert harness.layer.self_attn.qkv_proj.weight.float().abs().max() < 1.0
    assert "random" in harness.weights


def test_randomise_is_reproducible(tmp_path):
    first, second = _harness(tmp_path), _harness(tmp_path)
    first.randomise(seed=7)
    second.randomise(seed=7)
    assert torch.equal(
        first.layer.self_attn.qkv_proj.weight, second.layer.self_attn.qkv_proj.weight
    )


def test_run_refuses_uninitialised_memory(tmp_path):
    """Otherwise every number in the report reads whatever the allocator handed out."""
    with pytest.raises(RuntimeError, match="uninitialised"):
        _harness(tmp_path).run(batch=1, seq_len=4, decode_steps=1)


# --------------------------------------------------------------------------- #
# KV bookkeeping
# --------------------------------------------------------------------------- #
def _cache(batch=2, seq_len=3, decode_steps=2) -> SingleLayerCache:
    return SingleLayerCache(
        batch,
        seq_len,
        decode_steps,
        kv_row=(4, 8),  # the paged shape: 2 * num_kv_heads, head_dim
        dtype=torch.float16,
        device="cpu",
    )


def test_cache_holds_one_layer_and_the_padded_row_layout():
    cache = _cache()
    assert len(cache.meta.kv_buffer) == 1
    assert cache.meta.kv_buffer[0].shape == (2 * 5, 4, 8)
    # Request i owns a contiguous run of max_seq rows, so the padded grid the layer
    # flattens and the table agree row for row.
    assert cache.table.tolist() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def test_prefill_metadata_matches_the_runner():
    cache = _cache()
    meta = cache.begin_prefill()
    assert meta.is_prefill
    assert meta.b_seq_len.tolist() == [3, 3]
    assert meta.max_actual_seq_len == 3
    # Offsets into the flattened token batch, not into cache rows.
    assert meta.b_start_loc.tolist() == [0, 3]
    assert meta.cur_select_index.tolist() == [0, 1, 2, 5, 6, 7]
    assert meta.b_req_idx.tolist() == [0, 1]


def test_decode_reserves_one_row_per_sequence_and_grows_first():
    cache = _cache()
    cache.begin_prefill()
    meta = cache.step_decode()
    assert not meta.is_prefill
    assert meta.cur_select_index.tolist() == [3, 8]
    # Length includes the token just written: the decode kernel reads history up to it.
    assert meta.b_seq_len.tolist() == [4, 4]
    assert meta.max_actual_seq_len == 4

    meta = cache.step_decode()
    assert meta.cur_select_index.tolist() == [4, 9]
    assert meta.b_seq_len.tolist() == [5, 5]


def test_decode_past_the_reservation_raises():
    cache = _cache(decode_steps=1)
    cache.begin_prefill()
    cache.step_decode()
    with pytest.raises(RuntimeError, match="decode_steps"):
        cache.step_decode()


# --------------------------------------------------------------------------- #
# MLA (DeepseekV2): the latent cache, rope only on the pe slice
# --------------------------------------------------------------------------- #
def test_mla_layer_builds_the_latent_attention(tmp_path):
    """Layer 0 pairs MLA with the dense MLP; layers past the cutover route.

    The absorbed-decode views are checked by shape because that is their whole
    contract: zero-copy reads of kv_b's per-head ``[k_nope | v]`` layout.
    """
    from rapid_llm.modules import DeepseekV2MLAAttention, FusedMLP, SparseMoeBlock

    config = _config(tmp_path, _MLA)
    harness = SingleLayerHarness(config, 0, device="cpu")
    assert isinstance(harness.layer.self_attn, DeepseekV2MLAAttention)
    assert isinstance(harness.layer.mlp, FusedMLP)
    assert isinstance(SingleLayerHarness(config, 1, device="cpu").layer.mlp, SparseMoeBlock)

    attn = harness.layer.self_attn
    assert attn.w_uk.shape == (4, 16, 32)  # heads, kv_lora, qk_nope
    assert attn.w_uv.shape == (4, 16, 32)  # heads, kv_lora, v


def test_translate_flattens_mla_norms(tmp_path):
    """The layernorms fold; the MLA projections are real modules and stay put."""
    harness = _harness(tmp_path, _MLA)
    assert harness.translate("model.layers.1.self_attn.kv_a_layernorm.weight") == (
        "self_attn.kv_a_layernorm_weight",
        None,
    )
    assert harness.translate("model.layers.1.self_attn.q_a_layernorm.weight") == (
        "self_attn.q_a_layernorm_weight",
        None,
    )
    assert harness.translate("model.layers.1.self_attn.kv_a_proj_with_mqa.weight") == (
        "self_attn.kv_a_proj_with_mqa.weight",
        None,
    )


def test_mla_weights_mirror_into_the_right_blocks(tmp_path):
    """A HF MLA layer's state dict fills the module block by block.

    kv_b lands whole, so the w_uk/w_uv views read the real checkpoint layout —
    if the per-head halves were transposed at load time this copy would pass and
    every later kernel would read the wrong halves.
    """
    from rapid_llm.tools.harness import HFLayerReference

    config = _config(tmp_path, _MLA)
    harness = SingleLayerHarness(config, 0, device="cpu")
    reference = HFLayerReference(config, 0, device="cpu")
    harness.load_state_dict(reference.state_dict(), source="test")

    hf = dict(reference.state_dict())
    attn = harness.layer.self_attn
    torch.testing.assert_close(attn.q_proj.weight, hf["self_attn.q_proj.weight"])
    torch.testing.assert_close(
        attn.kv_a_proj_with_mqa.weight, hf["self_attn.kv_a_proj_with_mqa.weight"]
    )
    torch.testing.assert_close(attn.kv_b_proj.weight, hf["self_attn.kv_b_proj.weight"])
    torch.testing.assert_close(attn.o_proj.weight, hf["self_attn.o_proj.weight"])
    torch.testing.assert_close(attn.kv_a_layernorm_weight, hf["self_attn.kv_a_layernorm.weight"])

    gate_up = harness.layer.mlp.gate_up_proj.weight
    rows = hf["mlp.gate_proj.weight"].shape[0]
    torch.testing.assert_close(gate_up[:rows], hf["mlp.gate_proj.weight"])
    torch.testing.assert_close(gate_up[rows:], hf["mlp.up_proj.weight"])


def test_mla_q_lora_weights_mirror(tmp_path):
    """The V2-full query path (q_a + layernorm + q_b) loads the same way."""
    from rapid_llm.tools.harness import HFLayerReference

    config = _config(tmp_path, _MLA, q_lora_rank=16)
    harness = SingleLayerHarness(config, 0, device="cpu")
    reference = HFLayerReference(config, 0, device="cpu")
    harness.load_state_dict(reference.state_dict(), source="test")

    hf = dict(reference.state_dict())
    attn = harness.layer.self_attn
    torch.testing.assert_close(attn.q_a_proj.weight, hf["self_attn.q_a_proj.weight"])
    torch.testing.assert_close(attn.q_a_layernorm_weight, hf["self_attn.q_a_layernorm.weight"])
    torch.testing.assert_close(attn.q_b_proj.weight, hf["self_attn.q_b_proj.weight"])


def test_new_cache_allocates_the_latent_pool(tmp_path):
    """MLA's cache row is the latent itself: ``[c_kv | k_pe]``, no head axis."""
    config = _config(tmp_path, _MLA)
    harness = SingleLayerHarness(config, 0, device="cpu")
    cache = harness.new_cache(2, 3, 1)
    assert cache.meta.kv_buffer[0].shape == (2 * 4, 1, 16 + 64)


# --------------------------------------------------------------------------- #
# Timing and reporting
# --------------------------------------------------------------------------- #
def test_module_timer_records_one_row_per_module_within_the_depth_budget():
    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x)

    class Outer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = Inner()
            self.b = Inner()

        def forward(self, x):
            return self.b(self.a(x))

    model = Outer()
    with ModuleTimer(model, "cpu") as timer:
        model(torch.zeros(1, 4))
        model(torch.zeros(1, 4))

    rows = {row.name: row for row in timer.results()}
    # Depth 2 keeps the children of the root and their direct children, and stops there.
    assert set(rows) == {"a", "b", "a.lin", "b.lin"}
    assert all(row.calls == 2 for row in rows.values())
    assert all(row.ms >= 0.0 for row in rows.values())


def test_module_timer_removes_its_hooks():
    model = nn.Sequential(nn.Linear(4, 4))
    with ModuleTimer(model, "cpu"):
        pass
    assert not model[0]._forward_pre_hooks
    assert not model[0]._forward_hooks


def test_diff_scales_by_the_reference_magnitude():
    reference = torch.tensor([[10.0, -20.0]])
    diff = Diff.between(torch.tensor([[10.5, -20.0]]), reference)
    assert diff.max_abs == pytest.approx(0.5)
    assert diff.mean_abs == pytest.approx(0.25)
    assert diff.rel == pytest.approx(0.5 / 20.0)


def test_diff_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        Diff.between(torch.zeros(2, 2), torch.zeros(2, 3))


def test_report_renders_every_section():
    report = LayerReport(
        model_type="qwen3",
        layer_index=3,
        mlp_kind="FusedMLP",
        device="cuda",
        dtype="bfloat16",
        weights="random (seed=0)",
        batch=2,
        seq_len=64,
        decode_steps=4,
        param_bytes=2**30,
        peak_mem_gb=1.5,
        prefill_ms=2.5,
        decode_ms=0.25,
        ops=(OpTiming("self_attn", 1, 1.5),),
        kernels=("native/linear_torch",),
        prefill_diff=Diff(1e-3, 1e-4, 2e-5),
    )
    text = report.render()
    assert "layer 3 of qwen3 (mlp=FusedMLP)" in text
    assert "1.000 GiB on cuda as bfloat16" in text
    assert "self_attn" in text
    assert "native/linear_torch" in text
    assert "prefill vs reference" in text
    # No decode diff was measured, so none is claimed.
    assert "decode vs reference" not in text


# --------------------------------------------------------------------------- #
# Reference resolution
# --------------------------------------------------------------------------- #
def test_reference_layer_class_comes_from_the_config(tmp_path):
    dense = hf_decoder_layer_class(_config(tmp_path, _DENSE).text_config)
    assert dense.__name__ == "Qwen3DecoderLayer"
    moe = hf_decoder_layer_class(_config(tmp_path, _MOE).text_config)
    assert moe.__name__ == "Qwen3MoeDecoderLayer"


def test_reference_decode_before_prefill_is_an_error(tmp_path):
    from rapid_llm.tools.harness import HFLayerReference

    config = _config(tmp_path, _DENSE)
    reference = HFLayerReference(config, 0, device="cpu")
    with pytest.raises(RuntimeError, match="prefill"):
        reference.decode(torch.zeros(1, 1, 64), (torch.zeros(1, 1, 16), torch.zeros(1, 1, 16)))


# --------------------------------------------------------------------------- #
# The numeric check, which needs the Triton kernels
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.usefixtures("cuda_available")
def test_layer_agrees_with_transformers(tmp_path):
    """Prefill and decode must both match the same layer as transformers builds it.

    Both phases, because they run different kernels over different cache layouts: a
    paged-decode bug is invisible in a prefill-only comparison, and catching it here is
    the point of the harness.
    """
    from rapid_llm.tools.harness import HFLayerReference

    config = _config(tmp_path, _DENSE)
    harness = SingleLayerHarness(config, 1, device="cuda")
    reference = HFLayerReference(config, 1, device="cuda")

    report = harness.run(batch=2, seq_len=16, decode_steps=2, iters=1, reference=reference)
    assert report.prefill_diff is not None and report.decode_diff is not None
    # bf16 through a fused-QKV GEMM and a Triton softmax accumulates more than fp32
    # would; 2% of the output's own peak is the band the whole-model parity tests use.
    assert report.prefill_diff.rel < 2e-2
    assert report.decode_diff.rel < 2e-2
    assert report.prefill_ms > 0.0
    assert report.ops


@pytest.mark.gpu
@pytest.mark.usefixtures("cuda_available")
def test_forward_leaves_its_input_alone(tmp_path):
    """The fused norm accumulates into the residual it was handed, which is the input.

    In a full stack that in-place add is the whole point. Here the same tensor goes to the
    reference next, so a layer that kept the mutation would be compared against its own
    post-attention sum — which reads as a 20% accuracy failure with no bad arithmetic
    anywhere in it.
    """
    harness = SingleLayerHarness(_config(tmp_path, _DENSE), 1, device="cuda")
    harness.randomise()
    prompt = harness.hidden_states(1, 8)
    before = prompt.clone()

    harness.forward(prompt, harness.new_cache(1, 8, 1).begin_prefill())
    assert torch.equal(prompt, before)


@pytest.mark.gpu
@pytest.mark.usefixtures("cuda_available")
def test_mla_layer_agrees_with_transformers(tmp_path):
    """The MLA layer matches transformers' own, in both phases.

    Prefill and decode take different paths — up-projected per head vs q kept
    absorbed against the latent — so two green numbers prove two different
    kernel compositions, plus the rope-on-pe-slice handling between them.
    """
    from rapid_llm.tools.harness import HFLayerReference

    config = _config(tmp_path, _MLA)
    harness = SingleLayerHarness(config, 0, device="cuda")
    reference = HFLayerReference(config, 0, device="cuda")

    report = harness.run(batch=2, seq_len=16, decode_steps=2, iters=1, reference=reference)
    assert report.prefill_diff is not None and report.decode_diff is not None
    assert report.prefill_diff.rel < 2e-2
    assert report.decode_diff.rel < 2e-2


@pytest.mark.gpu
@pytest.mark.usefixtures("cuda_available")
def test_mla_q_lora_layer_agrees_with_transformers(tmp_path):
    """Same alignment through the down-projected query path (V2-full shape)."""
    from rapid_llm.tools.harness import HFLayerReference

    config = _config(tmp_path, _MLA, q_lora_rank=16)
    harness = SingleLayerHarness(config, 0, device="cuda")
    reference = HFLayerReference(config, 0, device="cuda")

    report = harness.run(batch=2, seq_len=16, decode_steps=2, iters=1, reference=reference)
    assert report.prefill_diff is not None and report.decode_diff is not None
    assert report.prefill_diff.rel < 2e-2
    assert report.decode_diff.rel < 2e-2
