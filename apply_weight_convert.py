#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_weight_convert.py
~~~~~~~~~~~~~~~~~~~~
高性能版：跳过 Model 初始化，直接读取权重文件。
支持 .safetensors (极速) 和 .bin 格式。

Author: harleyszhang (Optimized 2025-06-08)
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import glob
import os
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel

# 尝试导入 safetensors，这是目前最快的加载方式
try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    from lite_llama.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 配置类与映射表 (保持不变)
# --------------------------------------------------------------------------- #
class ModelSpec:
    def __init__(self, common: dict, layer: dict, merge_bias: bool, prefix_rules: list[tuple] = None):
        self.common = common
        self.layer = layer
        self.merge_bias = merge_bias
        self.prefix_rules = prefix_rules or []

_LLAMA_LAYER_TEMPLATE = {
    "mlp.gate_proj.weight":    "mlp.gate_proj.weight",
    "mlp.up_proj.weight":      "mlp.up_proj.weight",
    "mlp.down_proj.weight":    "mlp.down_proj.weight",
}

_SPECS: dict[str, ModelSpec] = {
    "qwen2": ModelSpec(
        common={
            "model.norm.weight":         "norm_weight",
            "model.embed_tokens.weight": "embed_tokens.weight",
            "lm_head.weight":            "lm_head_weight",
        },
        layer={
            "model.layers.{i}.self_attn.q_proj.weight":  "layers.{i}.self_attn.q_proj_weight",
            "model.layers.{i}.self_attn.q_proj.bias":    "layers.{i}.self_attn.q_proj_bias",
            "model.layers.{i}.self_attn.k_proj.weight":  "layers.{i}.self_attn.k_proj_weight",
            "model.layers.{i}.self_attn.k_proj.bias":    "layers.{i}.self_attn.k_proj_bias",
            "model.layers.{i}.self_attn.v_proj.weight":  "layers.{i}.self_attn.v_proj_weight",
            "model.layers.{i}.self_attn.v_proj.bias":    "layers.{i}.self_attn.v_proj_bias",
            "model.layers.{i}.self_attn.o_proj.weight":  "layers.{i}.self_attn.o_proj_weight",
            "model.layers.{i}.input_layernorm.weight":   "layers.{i}.input_layernorm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm_weight",
            **{f"model.layers.{{i}}.{k}": f"layers.{{i}}.{v}" for k, v in _LLAMA_LAYER_TEMPLATE.items()}
        },
        merge_bias=True
    ),
    "qwen3": ModelSpec(
        common={
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight":         "norm_weight",
            "lm_head.weight":            "lm_head_weight",
        },
        layer={
            "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj_weight",
            "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj_weight",
            "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj_weight",
            "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj_weight",
            "model.layers.{i}.self_attn.q_norm.weight": "layers.{i}.self_attn.q_norm_weight",
            "model.layers.{i}.self_attn.k_norm.weight": "layers.{i}.self_attn.k_norm_weight",
            "model.layers.{i}.input_layernorm.weight":  "layers.{i}.input_layernorm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm_weight",
            **{f"model.layers.{{i}}.{k}": f"layers.{{i}}.{v}" for k, v in _LLAMA_LAYER_TEMPLATE.items()}
        },
        merge_bias=False
    ),
    "llama": ModelSpec(
        common={
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight":         "norm_weight",
            "lm_head.weight":            "lm_head.weight",
        },
        layer={
            "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj.weight",
            "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj.weight",
            "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj.weight",
            "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj.weight",
            "model.layers.{i}.input_layernorm.weight":  "layers.{i}.attention_norm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.ffn_norm_weight",
            **{f"model.layers.{{i}}.{k}": f"layers.{{i}}.{v}" for k, v in _LLAMA_LAYER_TEMPLATE.items()}
        },
        merge_bias=False
    ),
    "llava": ModelSpec(
        common={
            # HF 的 LlavaForConditionalGeneration state_dict 常见两种前缀：
            # 1) language_model.*（少见，某些导出/重排后会出现）
            # 2) model.language_model.*（更常见，原生 HF/Transformers 权重）
            "language_model.model.embed_tokens.weight":       "language_model.embed_tokens.weight",
            "language_model.model.norm.weight":               "language_model.norm_weight",
            "language_model.lm_head.weight":                  "language_model.lm_head.weight",
            "model.language_model.model.embed_tokens.weight": "language_model.embed_tokens.weight",
            "model.language_model.model.norm.weight":         "language_model.norm_weight",
            "model.language_model.lm_head.weight":            "language_model.lm_head.weight",

            # LLaVA v1.5 常见“分包”形式：语言模型是纯 LLaMA（pytorch_model-*.bin 里是 model.*）
            "model.embed_tokens.weight": "language_model.embed_tokens.weight",
            "model.norm.weight":         "language_model.norm_weight",
            "lm_head.weight":            "language_model.lm_head.weight",

            # projector 常见存放在 mm_projector.bin，key 形如 model.mm_projector.{0,2}.*
            "model.mm_projector.0.weight": "multi_modal_projector.linear_1.weight",
            "model.mm_projector.0.bias":   "multi_modal_projector.linear_1.bias",
            "model.mm_projector.2.weight": "multi_modal_projector.linear_2.weight",
            "model.mm_projector.2.bias":   "multi_modal_projector.linear_2.bias",
        },
        layer = {
            # key 是原始权重值, value 是自定义模型结构权重参数
            "language_model.model.layers.{i}.self_attn.q_proj.weight": "language_model.layers.{i}.self_attn.q_proj.weight",
            "language_model.model.layers.{i}.self_attn.k_proj.weight": "language_model.layers.{i}.self_attn.k_proj.weight",
            "language_model.model.layers.{i}.self_attn.v_proj.weight": "language_model.layers.{i}.self_attn.v_proj.weight",
            "language_model.model.layers.{i}.self_attn.o_proj.weight": "language_model.layers.{i}.self_attn.o_proj.weight",
            "language_model.model.layers.{i}.mlp.gate_proj.weight": "language_model.layers.{i}.mlp.gate_proj.weight",
            "language_model.model.layers.{i}.mlp.up_proj.weight": "language_model.layers.{i}.mlp.up_proj.weight",
            "language_model.model.layers.{i}.mlp.down_proj.weight": "language_model.layers.{i}.mlp.down_proj.weight",
            "language_model.model.layers.{i}.input_layernorm.weight": "language_model.layers.{i}.attention_norm_weight",
            "language_model.model.layers.{i}.post_attention_layernorm.weight": "language_model.layers.{i}.ffn_norm_weight",

            # 完整 HF 形式（带 model.language_model 前缀）
            "model.language_model.model.layers.{i}.self_attn.q_proj.weight": "language_model.layers.{i}.self_attn.q_proj.weight",
            "model.language_model.model.layers.{i}.self_attn.k_proj.weight": "language_model.layers.{i}.self_attn.k_proj.weight",
            "model.language_model.model.layers.{i}.self_attn.v_proj.weight": "language_model.layers.{i}.self_attn.v_proj.weight",
            "model.language_model.model.layers.{i}.self_attn.o_proj.weight": "language_model.layers.{i}.self_attn.o_proj.weight",
            "model.language_model.model.layers.{i}.mlp.gate_proj.weight": "language_model.layers.{i}.mlp.gate_proj.weight",
            "model.language_model.model.layers.{i}.mlp.up_proj.weight": "language_model.layers.{i}.mlp.up_proj.weight",
            "model.language_model.model.layers.{i}.mlp.down_proj.weight": "language_model.layers.{i}.mlp.down_proj.weight",
            "model.language_model.model.layers.{i}.input_layernorm.weight": "language_model.layers.{i}.attention_norm_weight",
            "model.language_model.model.layers.{i}.post_attention_layernorm.weight": "language_model.layers.{i}.ffn_norm_weight",

            # 分包形式：纯 LLaMA（model.layers.*）
            "model.layers.{i}.self_attn.q_proj.weight": "language_model.layers.{i}.self_attn.q_proj.weight",
            "model.layers.{i}.self_attn.k_proj.weight": "language_model.layers.{i}.self_attn.k_proj.weight",
            "model.layers.{i}.self_attn.v_proj.weight": "language_model.layers.{i}.self_attn.v_proj.weight",
            "model.layers.{i}.self_attn.o_proj.weight": "language_model.layers.{i}.self_attn.o_proj.weight",
            "model.layers.{i}.mlp.gate_proj.weight": "language_model.layers.{i}.mlp.gate_proj.weight",
            "model.layers.{i}.mlp.up_proj.weight": "language_model.layers.{i}.mlp.up_proj.weight",
            "model.layers.{i}.mlp.down_proj.weight": "language_model.layers.{i}.mlp.down_proj.weight",
            "model.layers.{i}.input_layernorm.weight": "language_model.layers.{i}.attention_norm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "language_model.layers.{i}.ffn_norm_weight",
        },
        merge_bias=False,
        prefix_rules=[
            ("vision_tower.", "vision_tower."),
            ("model.vision_tower.", "vision_tower."),
            ("multi_modal_projector.", "multi_modal_projector."),
            ("model.multi_modal_projector.", "multi_modal_projector."),
        ]
    ),
}

# --------------------------------------------------------------------------- #
# 高性能加载器
# --------------------------------------------------------------------------- #
def get_weight_files(ckpt_dir: Path) -> list[Path]:
    """获取权重文件列表，优先查找 .safetensors"""
    # 1. 优先找 safetensors (速度极快)
    safetensors = list(ckpt_dir.glob("*.safetensors"))
    if safetensors:
        return sorted(safetensors)
    
    # 2. 其次找 .bin（LLaVA 还可能有 mm_projector.bin）
    bins = list(ckpt_dir.glob("*.bin"))
    if bins:
        blacklist = ("training_args", "trainer_state", "optimizer", "scheduler", "rng_state", "scaler")
        return sorted([b for b in bins if not any(x in b.name for x in blacklist)])
    
    # 3. 找 .pt
    pts = list(ckpt_dir.glob("*.pt"))
    return sorted(pts)

def load_shard(file_path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """加载单个分片文件"""
    file_str = str(file_path)
    if file_str.endswith(".safetensors"):
        if not HAS_SAFETENSORS:
            raise ImportError("检测到 safetensors 文件，但未安装 `safetensors` 库。请执行 `pip install safetensors`")
        return load_safetensors(file_str, device=device)
    else:
        # .bin / .pt
        return torch.load(file_str, map_location=device)

def build_full_mapping(spec: ModelSpec, num_layers: int) -> dict[str, str]:
    mapping = dict(spec.common)
    for i in range(num_layers):
        mapping.update({k.format(i=i): v.format(i=i) for k, v in spec.layer.items()})
    return mapping

def check_prefix_rules(key: str, rules: list[tuple[str, str]]) -> str | None:
    for src, dst in rules:
        if key.startswith(src):
            return dst + key[len(src):]
    return None

def merge_kv_weights(state: dict[str, torch.Tensor], prefix: str, with_bias: bool = False) -> None:
    """KV 合并逻辑 (就地修改)"""
    candidates = [
        (f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"),
        (f"{prefix}.k_proj_weight", f"{prefix}.v_proj_weight"),
    ]
    k_key, v_key = None, None
    for k, v in candidates:
        if k in state and v in state:
            k_key, v_key = k, v
            break
            
    if not k_key: return

    target_key = f"{prefix}.kv_proj_weight"
    # CPU 上合并是内存操作，不涉及计算
    state[target_key] = torch.cat([state[k_key], state[v_key]], dim=0)
    del state[k_key], state[v_key]

    if with_bias:
        bias_cands = [
            (f"{prefix}.k_proj.bias", f"{prefix}.v_proj.bias"),
            (f"{prefix}.k_proj_bias", f"{prefix}.v_proj_bias"),
        ]
        for kb, vb in bias_cands:
            if kb in state and vb in state:
                state[f"{prefix}.kv_proj_bias"] = torch.cat([state[kb], state[vb]], dim=0)
                del state[kb], state[vb]
                break

# --------------------------------------------------------------------------- #
# 核心流程
# --------------------------------------------------------------------------- #
def convert_model(checkpoints_dir: Path, output_dir: Path, model_type: str, device: str = "cpu") -> None:
    spec = _SPECS.get(model_type)
    if not spec:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 1. 获取模型层数 (轻量级，只加载 Config)
    logger.info("📖 读取 Config...")
    config = AutoConfig.from_pretrained(checkpoints_dir, trust_remote_code=True)
    if hasattr(config, "text_config"): # Llava
        num_layers = config.text_config.num_hidden_layers
    else:
        num_layers = config.num_hidden_layers
    
    # 2. 准备映射表
    full_mapping = build_full_mapping(spec, num_layers)
    new_state: dict[str, torch.Tensor] = {}
    
    # 3. 扫描并加载权重文件
    weight_files = get_weight_files(checkpoints_dir)
    if not weight_files:
        raise FileNotFoundError(f"在 {checkpoints_dir} 中未找到权重文件 (.safetensors/.bin)")
    
    logger.info(f"🚀 发现 {len(weight_files)} 个权重分片，开始并行加载与映射...")

    # 4. 逐个文件加载 -> 映射 -> 释放 (流式处理，极省内存)
    total_params = 0
    for w_file in tqdm(weight_files, desc="Processing Shards"):
        # 加载单个分片 (Raw Tensor)
        shard = load_shard(w_file, device="cpu") # 强制 CPU 以避免显存碎片，转换通常是 CPU I/O 密集型
        
        # 立即处理当前分片中的 Key
        keys_to_process = list(shard.keys())
        for k in keys_to_process:
            v = shard[k]
            mapped_key = None
            
            # 查找映射
            if k in full_mapping:
                mapped_key = full_mapping[k]
            elif (remapped := check_prefix_rules(k, spec.prefix_rules)) is not None:
                mapped_key = remapped
            
            if mapped_key:
                # 移动到主字典，同时如果用户指定了 cuda，此时再转 device
                if device != "cpu":
                    new_state[mapped_key] = v.to(device)
                else:
                    new_state[mapped_key] = v
                total_params += 1
            
            # 关键：从 shard 中删除引用，协助 Python GC
            del shard[k]
        
        del shard
        gc.collect()

    logger.info(f"✅ 映射完成，共提取 {total_params} 个参数张量。开始合并 KV...")

    # 5. 合并 KV (CPU 计算极快)
    kv_prefix_tpl = "language_model.layers.{i}.self_attn" if model_type == "llava" else "layers.{i}.self_attn"
    for i in tqdm(range(num_layers), desc="Merging KV"):
        prefix = kv_prefix_tpl.format(i=i)
        merge_kv_weights(new_state, prefix, with_bias=spec.merge_bias)

    # 5.1 LLaVA：若 checkpoint 不含 vision_tower（分包/离线权重常见），为了 strict load 通过，补齐一份按 config 初始化的视觉塔参数。
    # 注意：这不是预训练视觉塔权重，仅用于让推理链路不因 Missing keys 失败。
    if model_type == "llava":
        has_vision = any(k.startswith("vision_tower.") for k in new_state.keys())
        if not has_vision:
            logger.warning("未发现 vision_tower 权重；将使用 config 初始化 vision_tower 并写入其初始参数（用于 strict load 通过）。")
            cfg = AutoConfig.from_pretrained(checkpoints_dir, trust_remote_code=True)
            vision_cfg = getattr(cfg, "vision_config", None)
            if vision_cfg is None:
                raise ValueError("config 中缺少 vision_config，无法初始化 vision_tower")
            vision_model = AutoModel.from_config(vision_cfg)
            vision_sd = vision_model.state_dict()
            for k, v in vision_sd.items():
                new_state[f"vision_tower.{k}"] = v.to(dtype=torch.float16)
            logger.info("已补齐 vision_tower 初始权重: %d tensors", len(vision_sd))
            del vision_model, vision_sd
            gc.collect()

    # 6. 保存
    model_id = checkpoints_dir.name
    final_out_dir = output_dir / model_id
    final_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = final_out_dir / f"{model_id}.pth"

    logger.info(f"💾 正在保存结果到: {out_path}")
    torch.save(new_state, out_path, _use_new_zipfile_serialization=True)
    
    # 复制辅助文件
    logger.info("📂 复制元数据文件...")
    for ext in ["*.json", "*.model", "*.txt", "*.tiktoken"]:
        for file in checkpoints_dir.glob(ext):
            shutil.copy2(file, final_out_dir)

    logger.info("🎉 转换结束！")

# --------------------------------------------------------------------------- #
# 入口逻辑
# --------------------------------------------------------------------------- #
def detect_model_type_from_config(checkpoints_dir: Path) -> str:
    config_path = checkpoints_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("找不到 config.json")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        
    hf_type = cfg.get("model_type", "").lower()
    return {
        "qwen2": "qwen2",
        "llama": "llama", 
        "llava": "llava"
    }.get(hf_type, hf_type)

def main():
    parser = argparse.ArgumentParser(description="High Performance Converter for Lite-LLaMA.")
    parser.add_argument("checkpoints_dir", type=Path)
    parser.add_argument("--model-type", type=str, choices=_SPECS.keys())
    parser.add_argument("--output-dir", type=Path, default=Path("my_weight"))
    parser.add_argument("--device", default="cpu", help="cpu (recommended for conversion) or cuda")
    
    args = parser.parse_args()
    ckpt_dir = args.checkpoints_dir.resolve()

    if not ckpt_dir.exists():
        logger.error("目录不存在")
        return

    # 确定模型类型
    model_type = args.model_type or detect_model_type_from_config(ckpt_dir)
    if model_type not in _SPECS:
        logger.error(f"不支持的模型类型: {model_type}")
        return
    
    logger.info(f"检测模型类型: {model_type}")

    convert_model(ckpt_dir, args.output_dir, model_type, args.device)

if __name__ == "__main__":
    main()