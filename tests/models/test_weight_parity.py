"""Round-trip parity: HuggingFace checkpoint in, identical lite_llama parameters out.

For each registered architecture a tiny HuggingFace model is randomly initialised,
saved as a real ``model.safetensors``, and loaded through the production path
(``ModelConfig`` -> registry -> ``materialise_parameters`` -> ``load_weights``).
Every parameter is then compared element by element against the HF tensor it came
from.

Coverage accounting inside ``load_weights`` already guarantees that *something*
wrote every parameter; what these tests add is that each tensor landed *where it
belongs*. A K/V swap, an off-by-one expert index or a gate/up transposition would
all satisfy the coverage check and silently corrupt the model, and none of them
would be visible in a key-set comparison.

Tiny configs keep the whole file on the CPU in a few seconds, which is what makes
it affordable to cover LLaVA and Qwen3-VL here rather than only against a 7B
checkpoint someone has to download.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from lite_llama.executor.loader import materialise_parameters
from lite_llama.executor.weight_utils import hf_weights_iterator
from lite_llama.models.config import ModelConfig
from lite_llama.models.registry import ModelRegistry

_TEXT_BODY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
}

_CLIP_VISION_BODY = {
    "model_type": "clip_vision_model",
    "hidden_size": 32,
    "intermediate_size": 37,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "image_size": 32,
    "patch_size": 16,
}

_QWEN3_VL_VISION_BODY = {
    "model_type": "qwen3_vl_vision",
    "hidden_size": 32,
    "intermediate_size": 37,
    "num_heads": 4,
    "depth": 2,
    "out_hidden_size": 64,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 1,
    "deepstack_visual_indexes": [0],
}

#: ``model_type`` -> (config.json body, HF implementation). Every registered
#: architecture appears here, so a model cannot be added to the registry without
#: someone noticing that its weight mapping is untested.
CASES: dict[str, tuple[dict, str]] = {
    "llama": (
        {"model_type": "llama", **_TEXT_BODY, "tie_word_embeddings": False},
        "transformers:LlamaForCausalLM",
    ),
    # Qwen2 is the only family with a bias on q/k/v, so it is also the only one
    # that exercises the fused ``qkv_proj.bias``.
    "qwen2": (
        {"model_type": "qwen2", **_TEXT_BODY, "tie_word_embeddings": True},
        "transformers:Qwen2ForCausalLM",
    ),
    # head_dim != hidden_size // num_heads, plus per-head q/k norm.
    "qwen3": (
        {"model_type": "qwen3", **_TEXT_BODY, "head_dim": 32, "tie_word_embeddings": True},
        "transformers:Qwen3ForCausalLM",
    ),
    "qwen3_moe": (
        {
            "model_type": "qwen3_moe",
            **_TEXT_BODY,
            "head_dim": 32,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "norm_topk_prob": True,
            "tie_word_embeddings": False,
        },
        "transformers:Qwen3MoeForCausalLM",
    ),
    "llava": (
        {
            "model_type": "llava",
            "text_config": {"model_type": "llama", **_TEXT_BODY, "tie_word_embeddings": False},
            "vision_config": _CLIP_VISION_BODY,
            "image_token_index": 127,
            "projector_hidden_act": "gelu",
            "vision_feature_layer": -2,
            "vision_feature_select_strategy": "default",
        },
        "transformers:LlavaForConditionalGeneration",
    ),
    "qwen3_vl": (
        {
            "model_type": "qwen3_vl",
            "text_config": {
                "model_type": "qwen3_vl_text",
                **_TEXT_BODY,
                "head_dim": 32,
                "tie_word_embeddings": True,
                # transformers 5.x nests the RoPE base and the mrope section here,
                # with no loose ``rope_theta`` alongside them.
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 5000000,
                    "mrope_section": [6, 5, 5],
                    "mrope_interleaved": True,
                },
            },
            "vision_config": _QWEN3_VL_VISION_BODY,
            "image_token_id": 126,
            "video_token_id": 125,
        },
        "transformers:Qwen3VLForConditionalGeneration",
    ),
}


def test_every_registered_model_has_a_parity_case():
    """A new architecture must not slip in without its weight mapping being checked."""
    assert set(CASES) == set(ModelRegistry.supported_types())


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _import(path: str):
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def _save(state: dict[str, torch.Tensor], directory: Path) -> None:
    # metadata={"format": "pt"} is what transformers writes; safetensors also
    # refuses tensors that share storage, which is why the caller clones.
    save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})


def write_hf_checkpoint(directory: Path, model_type: str) -> tuple[dict, ModelConfig]:
    """Random-init the HF model for ``model_type`` and save it as a real checkpoint.

    Returns:
        ``(hf state dict, ModelConfig)``. The state dict is returned so parity is
        asserted against the source tensors rather than a re-read of the file.
    """
    body, hf_path = CASES[model_type]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(body))

    config = ModelConfig.from_pretrained(directory, max_seq_len=128)
    torch.manual_seed(0)
    hf_model = _import(hf_path)(config.hf_config).eval()
    state = {key: value.detach().clone() for key, value in hf_model.state_dict().items()}
    _save(state, directory)
    return state, config


def load_lite_model(config: ModelConfig, directory: Path) -> nn.Module:
    """Build and fill the lite_llama model exactly as ``DefaultModelLoader`` does."""
    model_cls = ModelRegistry.resolve(config.model_type).load_class()
    model = model_cls(config)
    materialise_parameters(model, "cpu", dtype=config.dtype)
    model.load_weights(hf_weights_iterator(directory, dequant_dtype=config.dtype))
    return model.eval()


def _text_prefix(model_type: str) -> str:
    """lite_llama parameter prefix of the decoder stack for this architecture."""
    return "language_model." if ModelRegistry.resolve(model_type).is_multimodal else ""


# --------------------------------------------------------------------------- #
# Parity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_type", sorted(CASES))
def test_decoder_weights_land_in_the_right_place(model_type: str, tmp_path: Path):
    state, config = write_hf_checkpoint(tmp_path, model_type)
    params = dict(load_lite_model(config, tmp_path).named_parameters())
    lite = _text_prefix(model_type)
    # Every HF layout in CASES nests the decoder under ``model.`` (and multimodal
    # ones under ``model.language_model.``).
    hf = f"model.{lite}" if lite else "model."

    def same(hf_key: str, lite_key: str, index=(...,)) -> None:
        # fp32 source tensors go through the same round-to-nearest-even cast the
        # loader's ``copy_`` performs, so parity stays bit-exact in the model dtype.
        assert torch.equal(params[lite_key].data[index], state[hf_key].to(config.dtype)), lite_key

    same(f"{hf}embed_tokens.weight", f"{lite}embed_tokens.weight")
    same(f"{hf}norm.weight", f"{lite}norm_weight")

    q, kv = config.q_size, config.kv_size
    for i in range(config.num_layers):
        attn, lite_attn = f"{hf}layers.{i}.self_attn", f"{lite}layers.{i}.self_attn"
        same(f"{attn}.o_proj.weight", f"{lite_attn}.o_proj.weight")
        # The fused blocks are the whole reason a mapping layer exists: under GQA they
        # have different widths, yet permuting them keeps every shape and every count
        # valid, so only an element-wise check can tell them apart.
        same(f"{attn}.q_proj.weight", f"{lite_attn}.qkv_proj.weight", (slice(0, q),))
        same(f"{attn}.k_proj.weight", f"{lite_attn}.qkv_proj.weight", (slice(q, q + kv),))
        same(f"{attn}.v_proj.weight", f"{lite_attn}.qkv_proj.weight", (slice(q + kv, q + 2 * kv),))

        norms = f"{attn}.q_norm.weight" in state
        assert norms == (f"{lite_attn}.q_norm_weight" in params)
        if norms:
            same(f"{attn}.q_norm.weight", f"{lite_attn}.q_norm_weight")
            same(f"{attn}.k_norm.weight", f"{lite_attn}.k_norm_weight")

        # Qwen2 is the only family here with a q/k/v bias, i.e. the only one that
        # exercises the fused ``qkv_proj.bias``.
        if f"{attn}.q_proj.bias" in state:
            same(f"{attn}.q_proj.bias", f"{lite_attn}.qkv_proj.bias", (slice(0, q),))
            same(f"{attn}.k_proj.bias", f"{lite_attn}.qkv_proj.bias", (slice(q, q + kv),))
            same(
                f"{attn}.v_proj.bias",
                f"{lite_attn}.qkv_proj.bias",
                (slice(q + kv, q + 2 * kv),),
            )

        mlp, lite_mlp = f"{hf}layers.{i}.mlp", f"{lite}layers.{i}.mlp"
        if f"{mlp}.gate.weight" not in state:  # dense SwiGLU
            inter = config.intermediate_size
            # gate/up fuse like the attention blocks, except the two halves are
            # equal, so the width alone places them.
            same(
                f"{mlp}.gate_proj.weight",
                f"{lite_mlp}.gate_up_proj.weight",
                (slice(0, inter),),
            )
            same(
                f"{mlp}.up_proj.weight",
                f"{lite_mlp}.gate_up_proj.weight",
                (slice(inter, 2 * inter),),
            )
            same(f"{mlp}.down_proj.weight", f"{lite_mlp}.down_proj.weight")
            continue

        # transformers >= 5 stacks the experts into the same ``[E, 2*inter, hidden]``
        # / ``[E, hidden, inter]`` layout lite_llama uses, so these are identity
        # copies. The per-expert layout the published checkpoints ship is covered
        # by ``test_per_expert_checkpoint_matches_the_stacked_one``.
        same(f"{mlp}.gate.weight", f"{lite_mlp}.gate_weight")
        same(f"{mlp}.experts.gate_up_proj", f"{lite_mlp}.experts.gate_up_proj")
        same(f"{mlp}.experts.down_proj", f"{lite_mlp}.experts.down_proj")


def test_per_expert_checkpoint_matches_the_stacked_one(tmp_path: Path):
    """The published Qwen3-MoE checkpoints store one matrix per expert.

    ``model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight`` has to be stacked at
    load time into the three tensors the grouped-GEMM kernel wants. Exploding the
    stacked tensors and reloading gives an exact reference for that stacking: an
    off-by-one expert index or a gate/up swap changes nothing about shapes or
    counts, so only a value comparison can catch it.
    """
    stacked_state, config = write_hf_checkpoint(tmp_path, "qwen3_moe")
    stacked = dict(load_lite_model(config, tmp_path).named_parameters())

    inter = config.moe_intermediate_size
    per_expert = dict(stacked_state)
    for i in range(config.num_layers):
        prefix = f"model.layers.{i}.mlp.experts"
        gate_up = per_expert.pop(f"{prefix}.gate_up_proj")
        down = per_expert.pop(f"{prefix}.down_proj")
        for e in range(config.num_experts):
            per_expert[f"{prefix}.{e}.gate_proj.weight"] = gate_up[e, :inter].clone()
            per_expert[f"{prefix}.{e}.up_proj.weight"] = gate_up[e, inter:].clone()
            per_expert[f"{prefix}.{e}.down_proj.weight"] = down[e].clone()
    _save(per_expert, tmp_path)

    reloaded = dict(load_lite_model(config, tmp_path).named_parameters())
    assert set(reloaded) == set(stacked)
    for name, param in reloaded.items():
        assert torch.equal(param.data, stacked[name].data), name


@pytest.mark.parametrize("model_type", ["llava", "qwen3_vl"])
def test_vision_tower_weights_land_in_the_right_place(model_type: str, tmp_path: Path):
    """Vision towers *are* HF modules, so their names pass through untouched.

    Only the prefix moves (``model.visual.`` -> ``vision_tower.``), and the whole
    tower has to arrive: a prefix rule that misses would leave every vision
    parameter unwritten.
    """
    state, config = write_hf_checkpoint(tmp_path, model_type)
    params = dict(load_lite_model(config, tmp_path).named_parameters())

    hf_prefix = "model.visual." if model_type == "qwen3_vl" else "model.vision_tower."
    tower = {k[len(hf_prefix) :]: v for k, v in state.items() if k.startswith(hf_prefix)}
    assert tower, f"no {hf_prefix}* keys in the HF checkpoint"

    lite_tower = {
        k[len("vision_tower.") :]: v for k, v in params.items() if k.startswith("vision_tower.")
    }
    assert set(tower) == set(lite_tower)
    for name, tensor in tower.items():
        assert torch.equal(lite_tower[name].data, tensor.to(config.dtype)), name


def test_llava_projector_weights_land_in_the_right_place(tmp_path: Path):
    state, config = write_hf_checkpoint(tmp_path, "llava")
    params = dict(load_lite_model(config, tmp_path).named_parameters())
    for leaf in ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias"):
        expected = state[f"model.multi_modal_projector.{leaf}"].to(config.dtype)
        assert torch.equal(params[f"multi_modal_projector.{leaf}"].data, expected), leaf


@pytest.mark.parametrize("model_type", sorted(CASES))
def test_lm_head_shipped_by_the_checkpoint_is_used_verbatim(model_type: str, tmp_path: Path):
    """When the file carries an ``lm_head``, it wins — the tie is only a fallback."""
    state, config = write_hf_checkpoint(tmp_path, model_type)
    params = dict(load_lite_model(config, tmp_path).named_parameters())
    lite = _text_prefix(model_type)

    assert torch.equal(params[f"{lite}lm_head.weight"].data, state["lm_head.weight"].to(config.dtype))


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "qwen3_vl"])
def test_tied_checkpoint_without_lm_head_shares_the_embedding_tensor(
    model_type: str, tmp_path: Path
):
    """The published Qwen2.5 / Qwen3 / Qwen3-VL checkpoints genuinely omit ``lm_head.weight``.

    The loader answers with an *alias* rather than a copy, so the assertion is identity,
    not equality: one tensor read two ways costs one vocabulary table instead of two,
    and under vocabulary parallelism a copy could not even stay consistent.
    """
    state, config = write_hf_checkpoint(tmp_path, model_type)
    assert config.tie_word_embeddings
    del state["lm_head.weight"]
    _save(state, tmp_path)

    model = load_lite_model(config, tmp_path)
    lite = _text_prefix(model_type)
    head = model.get_parameter(f"{lite}lm_head.weight")
    assert head is model.get_parameter(f"{lite}embed_tokens.weight")
    # ``named_parameters`` de-duplicates shared tensors, so the head disappears from it:
    # that is the memory saving showing up in the introspection surface.
    assert f"{lite}lm_head.weight" not in dict(model.named_parameters())


@pytest.mark.parametrize("model_type", sorted(CASES))
def test_a_dropped_checkpoint_key_fails_loudly(model_type: str, tmp_path: Path):
    """The check that makes every assertion above worth trusting.

    A silently unloaded parameter yields a model that runs and returns nonsense, so
    a truncated or mis-mapped checkpoint has to raise instead of loading.
    """
    state, config = write_hf_checkpoint(tmp_path, model_type)
    victim = next(k for k in state if k.endswith("self_attn.v_proj.weight"))
    del state[victim]
    _save(state, tmp_path)

    with pytest.raises(ValueError, match="does not cover every parameter"):
        load_lite_model(config, tmp_path)


@pytest.mark.parametrize("model_type", sorted(CASES))
def test_an_unexpected_checkpoint_key_fails_loudly(model_type: str, tmp_path: Path):
    """A key no rule understands must not be dropped on the floor."""
    state, config = write_hf_checkpoint(tmp_path, model_type)
    state["some.future.module.weight"] = torch.zeros(2, 2)
    _save(state, tmp_path)

    with pytest.raises(ValueError, match="unknown parameter"):
        load_lite_model(config, tmp_path)


# --------------------------------------------------------------------------- #
# dtype handling
# --------------------------------------------------------------------------- #


def test_checkpoints_without_a_dtype_default_to_bf16(tmp_path: Path):
    """No ``torch_dtype`` in config.json -> bf16 parameters, in one copy, no extra pass."""
    state, config = write_hf_checkpoint(tmp_path, "qwen3")
    assert state["model.embed_tokens.weight"].dtype == torch.float32
    model = load_lite_model(config, tmp_path)
    assert config.dtype == torch.bfloat16
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())


def test_bf16_checkpoints_stay_bf16_verbatim(tmp_path: Path):
    """An explicit bf16 ``torch_dtype`` loads bit-identically: no fp16 detour.

    Published Qwen3 checkpoints ship this exact shape of config, so the default
    path is also the exact-value path.
    """
    state, _ = write_hf_checkpoint(tmp_path, "llama")
    body, _ = CASES["llama"]
    (tmp_path / "config.json").write_text(json.dumps({**body, "torch_dtype": "bfloat16"}))
    config = ModelConfig.from_pretrained(tmp_path, max_seq_len=128)
    assert config.dtype == torch.bfloat16
    bf16 = {k: v.to(torch.bfloat16) for k, v in state.items()}
    _save(bf16, tmp_path)

    params = dict(load_lite_model(config, tmp_path).named_parameters())
    assert torch.equal(params["embed_tokens.weight"].data, bf16["model.embed_tokens.weight"])


def test_fp16_checkpoints_stay_fp16(tmp_path: Path):
    """The legacy fp16 path is kept: an fp16 checkpoint is not widened to bf16."""
    state, _ = write_hf_checkpoint(tmp_path, "llama")
    body, _ = CASES["llama"]
    (tmp_path / "config.json").write_text(json.dumps({**body, "torch_dtype": "float16"}))
    config = ModelConfig.from_pretrained(tmp_path, max_seq_len=128)
    assert config.dtype == torch.float16
    fp16 = {k: v.to(torch.float16) for k, v in state.items()}
    _save(fp16, tmp_path)

    params = dict(load_lite_model(config, tmp_path).named_parameters())
    assert params["embed_tokens.weight"].data.dtype == torch.float16
    assert torch.equal(params["embed_tokens.weight"].data, fp16["model.embed_tokens.weight"])


def test_qwen3_vl_language_model_gets_mrope_and_the_right_base(tmp_path: Path):
    """The config has to reach the RoPE layer, not just parse correctly.

    Qwen3-VL is the case that used to break: its text config nests both the RoPE
    base and ``mrope_section`` inside ``rope_parameters``, the old layer read the
    (absent) top-level ``rope_theta`` and defaulted to 10000, and the model then ran
    a 500x-wrong rotation while looking perfectly healthy.
    """
    _, config = write_hf_checkpoint(tmp_path, "qwen3_vl")
    model = load_lite_model(config, tmp_path)
    rope = model.language_model.rotary_emb

    assert rope.mrope_section == [6, 5, 5]
    assert rope.config["rope_theta"] == 5000000
    # And the base actually shaped the table: theta=10000 would give much larger
    # high-index frequencies.
    dim = rope.inv_freq.numel() * 2
    expected = 1.0 / (5000000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    torch.testing.assert_close(rope.inv_freq, expected)


def test_rope_inv_freq_is_not_dragged_down_to_fp16(tmp_path: Path):
    """Loading must not touch buffers: ``inv_freq`` is fp32 and absent from checkpoints.

    The previous loader finished with a blanket ``model.half()``, which also cast
    this buffer. fp16 frequencies cost ~0.1 rad of phase error by position 500.
    """
    _, config = write_hf_checkpoint(tmp_path, "qwen3")
    model = load_lite_model(config, tmp_path)
    assert model.rotary_emb.inv_freq.dtype == torch.float32


def test_torch_bin_checkpoints_still_load(tmp_path: Path):
    """Some Hub repos predate safetensors; the ``.bin`` fallback keeps them usable."""
    state, config = write_hf_checkpoint(tmp_path, "llama")
    (tmp_path / "model.safetensors").unlink()
    torch.save(state, tmp_path / "pytorch_model.bin")

    params = dict(load_lite_model(config, tmp_path).named_parameters())
    assert torch.equal(
        params["embed_tokens.weight"].data, state["model.embed_tokens.weight"].to(config.dtype)
    )
