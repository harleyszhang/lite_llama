#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_weight_convert.py
~~~~~~~~~~~~~~~~~~~~
将 Qwen-2/3、Llama、LLaVA 等 HuggingFace / PyTorch-bin 权重
整理为 lite_llama 框架的自定义格式模型权重的小工具。

Usage
-----
python lite_llama/apply_weight_convert.py /path/to/weights [--model-type qwen3] [--device cuda]

Author: harleyszhang (2025-06-08)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Mapping

import torch
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM,
                          LlavaConfig, LlavaForConditionalGeneration)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

from lite_llama.utils.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 通用工具函数
# --------------------------------------------------------------------------- #
def ensure_dir(path: Path) -> Path:
    """若目录不存在则创建，最后返回自身。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_state_dict(out_dir: Path, model_id: str, state: dict[str, torch.Tensor]) -> None:
    """保存 state_dict 并打印信息。"""
    torch.save(state, out_dir / f"{model_id}.pth", _use_new_zipfile_serialization=True)
    logger.info("✅ 已保存权重到 %s", out_dir / f"{model_id}.pth")


def copy_metadata(src: Path, dst: Path) -> None:
    """复制 *.json 与 tokenizer.model 等辅助文件。"""
    for file in src.glob("*.json"):
        shutil.copy2(file, dst)
    tok = src / "tokenizer.model"
    if tok.exists():
        shutil.copy2(tok, dst)


# --------------------------------------------------------------------------- #
# 修订后的 merge_kv_weights —— 只生成 kv_proj_weight，下划线风格
# --------------------------------------------------------------------------- #
def merge_kv_weights(state: dict[str, torch.Tensor],
                     prefix: str,
                     with_bias: bool = False) -> None:
    """
    将 K/V 投影合并为 kv_proj_weight / kv_proj_bias（可选），
    完全使用 **下划线** 键名，保证与 lite-llama 的 Qwen3 实现一致。
    同时若此前转换过留下旧的 kv_proj.weight，也会被删除。
    """
    # ---------- 1. 找到现有 K/V ----------
    # 两种候选命名：点号风格  layers.0.self_attn.k_proj.weight
    #            下划线风格 layers.0.self_attn.k_proj_weight
    candidates = [
        (f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"),
        (f"{prefix}.k_proj_weight", f"{prefix}.v_proj_weight"),
    ]
    for k_key, v_key in candidates:
        if k_key in state and v_key in state:
            break
    else:  # 没有任何一对匹配
        return

    # ---------- 2. 合并权重 ----------
    fused_k = f"{prefix}.kv_proj_weight"            # 目标键（下划线）
    state[fused_k] = torch.cat([state[k_key], state[v_key]], dim=0)
    del state[k_key], state[v_key]

    # ---------- 3. 合并 bias（可选） ----------
    if with_bias:
        bias_cands = [
            (f"{prefix}.k_proj.bias",  f"{prefix}.v_proj.bias"),
            (f"{prefix}.k_proj_bias",  f"{prefix}.v_proj_bias"),
        ]
        for kb_key, vb_key in bias_cands:
            if kb_key in state and vb_key in state:
                fused_b = f"{prefix}.kv_proj_bias"
                state[fused_b] = torch.cat([state[kb_key], state[vb_key]], dim=0)
                del state[kb_key], state[vb_key]
                break

    # ---------- 4. 如有旧版 kv_proj.weight，顺带删掉 ----------
    old_key = f"{prefix}.kv_proj.weight"
    if old_key in state:
        del state[old_key]


def build_mapping(common: Mapping[str, str],
                  layer_tpl: Mapping[str, str],
                  num_layers: int) -> dict[str, str]:
    """根据层数展开模板映射表。"""
    mapping = dict(common)
    for i in range(num_layers):
        mapping.update({hf.format(i=i): custom.format(i=i) for hf, custom in layer_tpl.items()})
    return mapping

# --------------------------------------------------------------------------- #
# 具体各模型映射规则
# --------------------------------------------------------------------------- #
_SPEC = {
    # Qwen-2
    "qwen2": {
        "common": {
            "model.norm.weight":         "norm_weight",
            "model.embed_tokens.weight": "embed_tokens.weight",
            "lm_head.weight":            "lm_head_weight",
        },
        "layer": {
            # q_proj/k_proj/... 同下
            "model.layers.{i}.self_attn.q_proj.weight":  "layers.{i}.self_attn.q_proj_weight",
            "model.layers.{i}.self_attn.q_proj.bias":    "layers.{i}.self_attn.q_proj_bias",
            "model.layers.{i}.self_attn.k_proj.weight":  "layers.{i}.self_attn.k_proj_weight",
            "model.layers.{i}.self_attn.k_proj.bias":    "layers.{i}.self_attn.k_proj_bias",
            "model.layers.{i}.self_attn.v_proj.weight":  "layers.{i}.self_attn.v_proj_weight",
            "model.layers.{i}.self_attn.v_proj.bias":    "layers.{i}.self_attn.v_proj_bias",
            "model.layers.{i}.self_attn.o_proj.weight":  "layers.{i}.self_attn.o_proj_weight",
            "model.layers.{i}.mlp.gate_proj.weight":     "layers.{i}.mlp.gate_proj.weight",
            "model.layers.{i}.mlp.up_proj.weight":       "layers.{i}.mlp.up_proj.weight",
            "model.layers.{i}.mlp.down_proj.weight":     "layers.{i}.mlp.down_proj.weight",
            "model.layers.{i}.input_layernorm.weight":   "layers.{i}.input_layernorm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm_weight",
        },
        "merge_bias": True,
    },

    # Qwen-3
    "qwen3": {
        "common": {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight":         "norm_weight",
            "lm_head.weight":            "lm_head_weight",
        },
        "layer": {
            "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj_weight",
            "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj_weight",
            "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj_weight",
            "model.layers.{i}.self_attn.q_norm.weight": "layers.{i}.self_attn.q_norm_weight",
            "model.layers.{i}.self_attn.k_norm.weight": "layers.{i}.self_attn.k_norm_weight",
            "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj_weight",
            "model.layers.{i}.mlp.gate_proj.weight":    "layers.{i}.mlp.gate_proj.weight",
            "model.layers.{i}.mlp.up_proj.weight":      "layers.{i}.mlp.up_proj.weight",
            "model.layers.{i}.mlp.down_proj.weight":    "layers.{i}.mlp.down_proj.weight",
            "model.layers.{i}.input_layernorm.weight":  "layers.{i}.input_layernorm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm_weight",
        },
        "merge_bias": False,
    },

    # Llama-HF
    "llama": {
        "common": {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight":         "norm_weight",
            "lm_head.weight":            "lm_head.weight",
        },
        "layer": {
            "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj.weight",
            "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj.weight",
            "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj.weight",
            "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj.weight",
            "model.layers.{i}.mlp.gate_proj.weight":    "layers.{i}.mlp.gate_proj.weight",
            "model.layers.{i}.mlp.up_proj.weight":      "layers.{i}.mlp.up_proj.weight",
            "model.layers.{i}.mlp.down_proj.weight":    "layers.{i}.mlp.down_proj.weight",
            "model.layers.{i}.input_layernorm.weight":  "layers.{i}.attention_norm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.ffn_norm_weight",
        },
        "merge_bias": False,
    },

    # Llama-bin（原 Fairseq/Llama.PTH 格式）
    "llama-bin": {
        "common": {
            "tok_embeddings.weight": "embed_tokens.weight",
            "norm.weight":           "norm_weight",
            "output.weight":         "lm_head.weight",
        },
        "layer": {
            "layers.{i}.attention.wq.weight": "layers.{i}.attention.q_proj.weight",
            "layers.{i}.attention.wk.weight": "layers.{i}.attention.k_proj.weight",
            "layers.{i}.attention.wv.weight": "layers.{i}.attention.v_proj.weight",
            "layers.{i}.attention.wo.weight": "layers.{i}.attention.o_proj.weight",
            "layers.{i}.feed_forward.w1.weight": "layers.{i}.feed_forward.gate_proj.weight",
            "layers.{i}.feed_forward.w3.weight": "layers.{i}.feed_forward.up_proj.weight",
            "layers.{i}.feed_forward.w2.weight": "layers.{i}.feed_forward.down_proj.weight",
            "layers.{i}.attention_norm.weight":  "layers.{i}.attention_norm_weight",
            "layers.{i}.ffn_norm.weight":        "layers.{i}.ffn_norm_weight",
        },
        "merge_bias": False,
    },

    # LLaVA-Llama
    "llava": {
        "common": {
            "language_model.model.embed_tokens.weight": "language_model.embed_tokens.weight",
            "language_model.model.norm.weight":         "language_model.norm_weight",
            "language_model.lm_head.weight":            "language_model.lm_head.weight",
        },
        "layer": {
            "language_model.model.layers.{i}.self_attn.q_proj.weight": "language_model.layers.{i}.self_attn.q_proj_weight",
            "language_model.model.layers.{i}.self_attn.k_proj.weight": "language_model.layers.{i}.self_attn.k_proj_weight",
            "language_model.model.layers.{i}.self_attn.v_proj.weight": "language_model.layers.{i}.self_attn.v_proj_weight",
            "language_model.model.layers.{i}.self_attn.o_proj.weight": "language_model.layers.{i}.self_attn.o_proj_weight",
            "language_model.model.layers.{i}.mlp.gate_proj.weight":    "language_model.layers.{i}.mlp.gate_proj.weight",
            "language_model.model.layers.{i}.mlp.up_proj.weight":      "language_model.layers.{i}.mlp.up_proj.weight",
            "language_model.model.layers.{i}.mlp.down_proj.weight":    "language_model.layers.{i}.mlp.down_proj.weight",
            "language_model.model.layers.{i}.input_layernorm.weight":  "language_model.layers.{i}.attention_norm_weight",
            "language_model.model.layers.{i}.post_attention_layernorm.weight": "language_model.layers.{i}.ffn_norm_weight",
        },
        "merge_bias": False,
    },

    # Qwen3-VL (Vision-Language)
    "qwen3_vl": {
        "common": {
            "model.language_model.embed_tokens.weight": "language_model.embed_tokens.weight",
            "model.language_model.norm.weight":         "language_model.norm_weight",
            "lm_head.weight":                           "lm_head_weight",
        },
        "layer": {
            "model.language_model.layers.{i}.self_attn.q_proj.weight": "language_model.layers.{i}.self_attn.q_proj_weight",
            "model.language_model.layers.{i}.self_attn.k_proj.weight": "language_model.layers.{i}.self_attn.k_proj_weight",
            "model.language_model.layers.{i}.self_attn.v_proj.weight": "language_model.layers.{i}.self_attn.v_proj_weight",
            "model.language_model.layers.{i}.self_attn.q_norm.weight": "language_model.layers.{i}.self_attn.q_norm_weight",
            "model.language_model.layers.{i}.self_attn.k_norm.weight": "language_model.layers.{i}.self_attn.k_norm_weight",
            "model.language_model.layers.{i}.self_attn.o_proj.weight": "language_model.layers.{i}.self_attn.o_proj_weight",
            "model.language_model.layers.{i}.mlp.gate_proj.weight":    "language_model.layers.{i}.mlp.gate_proj.weight",
            "model.language_model.layers.{i}.mlp.up_proj.weight":      "language_model.layers.{i}.mlp.up_proj.weight",
            "model.language_model.layers.{i}.mlp.down_proj.weight":    "language_model.layers.{i}.mlp.down_proj.weight",
            "model.language_model.layers.{i}.input_layernorm.weight":  "language_model.layers.{i}.input_layernorm_weight",
            "model.language_model.layers.{i}.post_attention_layernorm.weight": "language_model.layers.{i}.post_attention_layernorm_weight",
        },
        "merge_bias": False,
        "kv_prefix": "language_model.layers.{i}.self_attn",
        "passthrough_prefix": "model.visual.",
    },
}

# --------------------------------------------------------------------------- #
# 核心转换逻辑
# --------------------------------------------------------------------------- #
def convert(checkpoints_dir: Path,
            hf_state: dict[str, torch.Tensor],
            model_type: str,
            num_layers: int) -> dict[str, torch.Tensor]:
    """执行主转换流程并把结果保存到 my_weight/<model_id>/ 目录。"""
    spec = _SPEC[model_type]
    mapping = build_mapping(spec["common"], spec["layer"], num_layers)
    new_sd: dict[str, torch.Tensor] = {}

    # 对于多模态模型，某些前缀的参数直接透传（如视觉编码器）
    passthrough_prefix = spec.get("passthrough_prefix", "")

    # ---------- 1. 重映射 ----------
    for k, v in tqdm(hf_state.items(), desc=f"[{model_type}] 权重重映射"):
        if passthrough_prefix and k.startswith(passthrough_prefix):
            # 视觉编码器权重：去掉 "model." 前缀，保留 "vision_tower." 前缀
            new_key = k.replace("model.visual.", "vision_tower.", 1)
            new_sd[new_key] = v
        elif (ck := mapping.get(k)) is not None:
            new_sd[ck] = v
        else:
            logger.debug("忽略未映射参数 %s", k)

    # ---------- 2. KV 合并 ----------
    kv_prefix_tpl = spec.get("kv_prefix", "layers.{i}.self_attn")
    if model_type.startswith("qwen") or model_type.startswith("llama"):
        for i in range(num_layers):
            prefix = kv_prefix_tpl.format(i=i)
            merge_kv_weights(new_sd, prefix, with_bias=spec["merge_bias"])

    # ---------- 3. 保存 ----------
    script_root = Path(__file__).resolve().parent
    out_dir = ensure_dir(script_root / "my_weight" / checkpoints_dir.name)
    save_state_dict(out_dir, checkpoints_dir.name, new_sd)
    copy_metadata(checkpoints_dir, out_dir)

    logger.info("🎉 转换完成，共 %d 个参数", len(new_sd))
    return new_sd



# --------------------------------------------------------------------------- #
# CLI 辅助：由 config.json 判别模型类型
# --------------------------------------------------------------------------- #
def detect_model_type(checkpoints_dir: Path) -> str:
    """
    读取 config.json 中的 model_type 字段。
    若该字段在 _SPEC 中无法找到，则抛出错误提示。
    """
    cfg = AutoConfig.from_pretrained(checkpoints_dir, trust_remote_code=True)
    mtype = cfg.model_type.lower()
    # 某些模型可能需要额外归一化 / 映射
    alias = {
        "qwen2":    "qwen2",
        "qwen3":    "qwen3",
        "qwen3_vl": "qwen3_vl",
        "llama":    "llama",
        "llava":    "llava",
    }.get(mtype, mtype)      # 默认原样返回
    if alias not in _SPEC:
        raise ValueError(f"暂不支持的 model_type '{mtype}'，请检查映射表")
    return alias


def load_hf_state(checkpoints_dir: Path,
                  model_type: str,
                  device: str = "cpu") -> dict[str, torch.Tensor]:
    """加载 HF / bin 权重到 state_dict。"""
    if model_type == "llava":
        model = (LlavaForConditionalGeneration
                 .from_pretrained(checkpoints_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
                 .to(device))
    elif model_type == "qwen3_vl":
        model = (Qwen3VLForConditionalGeneration
                 .from_pretrained(checkpoints_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
                 .to(device))
    else:
        model = (AutoModelForCausalLM
                 .from_pretrained(checkpoints_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
                 .to(device))
    return model.state_dict()


def get_num_layers(checkpoints_dir: Path, model_type: str) -> int:
    """从 config 中提取 Transformer 层数。"""
    if model_type == "llava":
        cfg = LlavaConfig.from_pretrained(checkpoints_dir)
        return cfg.text_config.num_hidden_layers
    if model_type == "qwen3_vl":
        cfg = AutoConfig.from_pretrained(checkpoints_dir, trust_remote_code=True)
        return cfg.text_config.num_hidden_layers

    cfg = AutoConfig.from_pretrained(checkpoints_dir)
    return cfg.num_hidden_layers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HF / bin checkpoints into Lite-LLaMA format.")
    parser.add_argument("checkpoints_dir", type=Path, help="模型权重目录")
    parser.add_argument("--model-type",
                        choices=_SPEC.keys(),
                        help="显式指定模型类型；默认根据目录名猜测")
    parser.add_argument("--device", default="cuda",
                        help="加载权重时使用的设备 (default: cuda)")
    args = parser.parse_args()

    ckpt_dir: Path = args.checkpoints_dir.resolve()
    
    # 1️⃣ **直接从 config.json 读取 model_type** ↓
    model_type = detect_model_type(ckpt_dir)
    logger.info("检测到 model_type = %s", model_type)

    # 2️⃣ 获取层数
    num_layers = get_num_layers(ckpt_dir, model_type)
    logger.info("Transformer 层数 %d", num_layers)

    # 3️⃣ 加载权重并执行转换
    hf_sd = load_hf_state(ckpt_dir, model_type, device=args.device)
    convert(ckpt_dir, hf_sd, model_type, num_layers)


if __name__ == "__main__":
    main()
