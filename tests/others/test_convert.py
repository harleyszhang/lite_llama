import os
import sys
import torch
import json

from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from apply_weight_convert import convert


# ---------- 公共工具 ----------
def dummy_state_dict_qwen(num_layers: int, hidden: int = 4):
    """构造一个最小化的 Qwen state_dict，仅包含 KV/O/W1/W2 ... 层权重。"""
    sd = {
        "model.norm.weight": torch.ones(hidden),
        "model.embed_tokens.weight": torch.zeros(10, hidden),
        "lm_head.weight": torch.zeros(hidden, 10),
    }
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        sd.update({
            f"{prefix}.self_attn.q_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.k_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.v_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.mlp.gate_proj.weight": torch.randn(2 * hidden, hidden),
            f"{prefix}.mlp.up_proj.weight":   torch.randn(2 * hidden, hidden),
            f"{prefix}.mlp.down_proj.weight": torch.randn(hidden, 2 * hidden),
            f"{prefix}.input_layernorm.weight": torch.ones(hidden),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden),
        })
    return sd


def dummy_state_dict_llama(num_layers: int, hidden: int = 4):
    """构造最小化 Llama HF 权重，方便测试不进行 KV 合并。"""
    sd = {
        "model.norm.weight": torch.ones(hidden),
        "model.embed_tokens.weight": torch.zeros(10, hidden),
        "lm_head.weight": torch.zeros(hidden, 10),
    }
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        sd.update({
            f"{prefix}.self_attn.q_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.k_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.v_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.mlp.gate_proj.weight": torch.randn(2 * hidden, hidden),
            f"{prefix}.mlp.up_proj.weight":   torch.randn(2 * hidden, hidden),
            f"{prefix}.mlp.down_proj.weight": torch.randn(hidden, 2 * hidden),
            f"{prefix}.input_layernorm.weight": torch.ones(hidden),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden),
        })
    return sd


# ---------- fixtures ----------
@pytest.fixture(scope="function")
def tmp_ckpt_dir(tmp_path: Path):
    """创建临时 checkpoints 目录并写入最小 config.json。"""
    def _factory(model_type: str):
        ckpt = tmp_path / model_type
        ckpt.mkdir()
        (ckpt / "config.json").write_text(json.dumps({"model_type": model_type}))
        return ckpt
    return _factory


# ---------- 测试映射 ----------
@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_mapping_and_num_params(tmp_ckpt_dir, model_type):
    ckpt_dir = tmp_ckpt_dir(model_type)
    num_layers = 2

    # 构造假权重
    sd = (dummy_state_dict_qwen(num_layers) if model_type.startswith("qwen")
          else dummy_state_dict_llama(num_layers))

    # 跑转换
    new_sd = convert(ckpt_dir, sd, model_type, num_layers)

    # 基础 key 应当存在
    assert "embed_tokens.weight" in new_sd
    assert "norm_weight" in new_sd


# ---------- Qwen KV 合并 ----------
def test_qwen_kv_merge(tmp_ckpt_dir):
    model_type = "qwen3"
    num_layers = 1
    ckpt_dir = tmp_ckpt_dir(model_type)
    sd = dummy_state_dict_qwen(num_layers)

    new_sd = convert(ckpt_dir, sd, model_type, num_layers)

    # KV fused weight 应出现
    kv_key = "layers.0.self_attn.kv_proj.weight"
    assert kv_key in new_sd

    # 原 K、V 不应保留
    assert "layers.0.self_attn.k_proj_weight" not in new_sd
    assert "layers.0.self_attn.v_proj_weight" not in new_sd

    # 维度检查：concat 后第一维应为 2*hidden
    hidden = sd["model.norm.weight"].numel()     # 4
    assert new_sd[kv_key].shape[0] == 2 * hidden


# ---------- Llama 不合并 ----------
def test_llama_no_kv_merge(tmp_ckpt_dir):
    model_type = "llama"
    num_layers = 1
    ckpt_dir = tmp_ckpt_dir(model_type)
    sd = dummy_state_dict_llama(num_layers)

    new_sd = convert(ckpt_dir, sd, model_type, num_layers)

    # KV fused 不存在
    assert "layers.0.self_attn.kv_proj.weight" not in new_sd
    # 原 K/V 仍然存在
    assert "layers.0.self_attn.k_proj.weight" in new_sd
    assert "layers.0.self_attn.v_proj.weight" in new_sd