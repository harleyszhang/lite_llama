"""
GPTQ weight loading and dequantization utilities for lite_llama
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import time
from typing import Dict, Optional, Tuple, Any
import numpy as np

try:
    import safetensors.torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("Warning: safetensors not installed. Install with: pip install safetensors")


class GPTQConfig:
    """Configuration for GPTQ quantization parameters"""
    def __init__(self, bits: int = 4, group_size: int = 128,
                 desc_act: bool = False, sym: bool = True,
                 true_sequential: bool = True):
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.sym = sym
        self.true_sequential = true_sequential
        self.pack_num = 32 // self.bits  # number of weights packed in int32

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GPTQConfig":
        """Create GPTQConfig from dictionary"""
        return cls(
            bits=config_dict.get("bits", 4),
            group_size=config_dict.get("group_size", 128),
            desc_act=config_dict.get("desc_act", False),
            sym=config_dict.get("sym", True),
            true_sequential=config_dict.get("true_sequential", True)
        )


def load_gptq_quantize_config(model_path: str) -> Optional[GPTQConfig]:
    """Load GPTQ quantization config from model directory"""
    quantize_config_path = Path(model_path) / "quantization_config.json"
    if not quantize_config_path.exists():
        return None

    with open(quantize_config_path, 'r') as f:
        config_dict = json.load(f)

    return GPTQConfig.from_dict(config_dict)


def unpack_gptq_weights(qweight: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    Unpack GPTQ quantized weights from int32 format

    Args:
        qweight: Packed quantized weights [out_features, in_features // pack_num]
        bits: Number of bits per weight (4 or 8)

    Returns:
        Unpacked weights [out_features, in_features]
    """
    pack_num = 32 // bits
    out_features = qweight.shape[0]
    in_features = qweight.shape[1] * pack_num

    unpacked_weights = torch.zeros((out_features, in_features),
                                   dtype=torch.int32, device=qweight.device)

    for i in range(pack_num):
        shift = i * bits
        if bits == 4:
            mask = 0xF
        elif bits == 8:
            mask = 0xFF
        else:
            raise ValueError(f"Unsupported bits: {bits}")

        unpacked_weights[:, i::pack_num] = (qweight >> shift) & mask

    return unpacked_weights


def dequantize_gptq(qweight: torch.Tensor, qzeros: torch.Tensor,
                    scales: torch.Tensor, g_idx: Optional[torch.Tensor] = None,
                    bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """
    Dequantize GPTQ weights

    Args:
        qweight: Packed quantized weights
        qzeros: Packed zero points
        scales: Scale factors
        g_idx: Group indices (optional, for act-order)
        bits: Quantization bits
        group_size: Quantization group size

    Returns:
        Dequantized weights in fp16
    """
    # Unpack weights and zeros
    weight = unpack_gptq_weights(qweight, bits).to(torch.float16)
    zeros = unpack_gptq_weights(qzeros, bits).to(torch.float16)

    # Handle act-order if needed
    if g_idx is not None:
        weight = weight[:, g_idx]
        zeros = zeros[:, g_idx]

    # Reshape for group-wise dequantization
    out_features, in_features = weight.shape
    num_groups = in_features // group_size

    weight = weight.reshape(out_features, num_groups, group_size)
    zeros = zeros.reshape(-1, num_groups, 1)
    scales = scales.reshape(-1, num_groups, 1)

    # Dequantize: w = (w_q - z) * s
    weight = (weight - zeros) * scales

    # Reshape back
    weight = weight.reshape(out_features, in_features)

    return weight


def load_gptq_linear_weights(checkpoint_path: str, layer_name: str,
                            gptq_config: GPTQConfig) -> Dict[str, torch.Tensor]:
    """
    Load GPTQ quantized linear layer weights

    Args:
        checkpoint_path: Path to checkpoint file
        layer_name: Name prefix of the layer (e.g., "layers.0.self_attn.q_proj")
        gptq_config: GPTQ configuration

    Returns:
        Dictionary containing dequantized weight and bias (if exists)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load quantized components
    qweight = checkpoint.get(f"{layer_name}.qweight")
    qzeros = checkpoint.get(f"{layer_name}.qzeros")
    scales = checkpoint.get(f"{layer_name}.scales")
    g_idx = checkpoint.get(f"{layer_name}.g_idx", None)
    bias = checkpoint.get(f"{layer_name}.bias", None)

    if qweight is None or qzeros is None or scales is None:
        # Fallback to non-quantized weight
        weight = checkpoint.get(f"{layer_name}.weight")
        if weight is None:
            raise ValueError(f"No weight found for {layer_name}")
        return {"weight": weight, "bias": bias}

    # Dequantize
    weight = dequantize_gptq(qweight, qzeros, scales, g_idx,
                            gptq_config.bits, gptq_config.group_size)

    return {"weight": weight, "bias": bias}


def convert_gptq_to_lite_llama(checkpoints_dir: str, model_config) -> Dict[str, torch.Tensor]:
    """
    Convert GPTQ quantized model to lite_llama format

    Args:
        checkpoints_dir: Directory containing GPTQ model files
        model_config: Model configuration

    Returns:
        State dictionary in lite_llama format
    """
    import safetensors.torch

    # Load GPTQ config
    gptq_config = load_gptq_quantize_config(checkpoints_dir)
    if gptq_config is None:
        raise ValueError(f"No quantization_config.json found in {checkpoints_dir}")

    # Find checkpoint files
    checkpoint_files = sorted(Path(checkpoints_dir).glob("*.safetensors"))
    use_safetensors = len(checkpoint_files) > 0

    if not checkpoint_files:
        checkpoint_files = sorted(Path(checkpoints_dir).glob("*.bin"))

    if not checkpoint_files:
        checkpoint_files = sorted(Path(checkpoints_dir).glob("*.pth"))

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoints_dir}")

    # Load all checkpoints (handle sharded models)
    full_state_dict = {}
    for checkpoint_file in checkpoint_files:
        if use_safetensors:
            if not HAS_SAFETENSORS:
                raise ImportError("safetensors is required for loading .safetensors files. Install with: pip install safetensors")
            state_dict = safetensors.torch.load_file(str(checkpoint_file))
        else:
            state_dict = torch.load(str(checkpoint_file), map_location="cpu")
        full_state_dict.update(state_dict)

    # Check if already in lite_llama format
    is_lite_llama_format = any("kv_proj_weight" in key for key in full_state_dict.keys())

    if is_lite_llama_format:
        print("Model is already in lite_llama format")
        # Just dequantize if needed
        new_state_dict = {}
        for key, value in full_state_dict.items():
            # Check if this is a quantized weight
            base_key = key.replace(".qweight", "").replace(".qzeros", "").replace(".scales", "").replace(".g_idx", "")

            if key.endswith(".qweight"):
                # This is a quantized weight, dequantize it
                qweight = value
                qzeros = full_state_dict.get(base_key + ".qzeros")
                scales = full_state_dict.get(base_key + ".scales")
                g_idx = full_state_dict.get(base_key + ".g_idx", None)

                if qzeros is not None and scales is not None:
                    weight = dequantize_gptq(
                        qweight, qzeros, scales, g_idx,
                        gptq_config.bits, gptq_config.group_size
                    )
                    new_state_dict[base_key] = weight
            elif not any(key.endswith(suffix) for suffix in [".qzeros", ".scales", ".g_idx"]):
                # Regular weight, just copy
                new_state_dict[key] = value

        return new_state_dict

    # Otherwise, convert based on model type
    if model_config.model_type.lower() == "llama":
        new_state_dict = convert_gptq_llama_to_lite_llama(
            full_state_dict, gptq_config, model_config
        )
    elif model_config.model_type.lower() == "qwen2":
        new_state_dict = convert_gptq_qwen2_to_lite_llama(
            full_state_dict, gptq_config, model_config
        )
    else:
        raise ValueError(f"Unsupported model type for GPTQ: {model_config.model_type}")

    return new_state_dict


def convert_gptq_llama_to_lite_llama(
    checkpoint: Dict[str, torch.Tensor],
    gptq_config: GPTQConfig,
    model_config
) -> Dict[str, torch.Tensor]:
    """Convert GPTQ Llama model to lite_llama format"""
    new_state_dict = {}

    # Check if this is already in lite_llama format
    is_lite_llama_format = any("kv_proj_weight" in key for key in checkpoint.keys())

    if is_lite_llama_format:
        # Already in lite_llama format, just process the weights
        for key, value in checkpoint.items():
            new_state_dict[key] = value
        return new_state_dict

    # Load embeddings and norms (these are not quantized)
    new_state_dict["embed_tokens.weight"] = checkpoint.get("model.embed_tokens.weight")
    new_state_dict["norm_weight"] = checkpoint.get("model.norm.weight")
    new_state_dict["lm_head.weight"] = checkpoint.get("lm_head.weight")

    # Process each layer
    for i in range(model_config.num_layers):
        # Check if we have separate k_proj and v_proj or merged kv_proj
        has_separate_kv = f"model.layers.{i}.self_attn.k_proj.weight" in checkpoint or \
                         f"model.layers.{i}.self_attn.k_proj.qweight" in checkpoint

        if has_separate_kv:
            # Process separate K and V projections
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                prefix = f"model.layers.{i}.self_attn.{proj}"

                # Check if quantized weights exist
                if f"{prefix}.qweight" in checkpoint:
                    # Load and dequantize
                    qweight = checkpoint[f"{prefix}.qweight"]
                    qzeros = checkpoint[f"{prefix}.qzeros"]
                    scales = checkpoint[f"{prefix}.scales"]
                    g_idx = checkpoint.get(f"{prefix}.g_idx", None)

                    weight = dequantize_gptq(
                        qweight, qzeros, scales, g_idx,
                        gptq_config.bits, gptq_config.group_size
                    )
                else:
                    # Use original weight if not quantized
                    weight = checkpoint.get(f"{prefix}.weight")
                    if weight is None and proj in ["k_proj", "v_proj"]:
                        # Skip if k_proj/v_proj don't exist (might be merged already)
                        continue
                    elif weight is None:
                        raise ValueError(f"No weight found for {prefix}")

                if proj in ["k_proj", "v_proj"]:
                    # Store temporarily for merging
                    new_state_dict[f"_temp_{i}_{proj}_weight"] = weight
                else:
                    new_state_dict[f"layers.{i}.self_attn.{proj}.weight"] = weight

            # Merge k and v projections if they were separate
            if f"_temp_{i}_k_proj_weight" in new_state_dict:
                k_weight = new_state_dict.pop(f"_temp_{i}_k_proj_weight")
                v_weight = new_state_dict.pop(f"_temp_{i}_v_proj_weight")
                new_state_dict[f"layers.{i}.self_attn.kv_proj_weight"] = torch.cat([k_weight, v_weight], dim=0)
        else:
            # Already has merged kv_proj
            # Q projection
            prefix = f"model.layers.{i}.self_attn.q_proj"
            if f"{prefix}.qweight" in checkpoint:
                qweight = checkpoint[f"{prefix}.qweight"]
                qzeros = checkpoint[f"{prefix}.qzeros"]
                scales = checkpoint[f"{prefix}.scales"]
                g_idx = checkpoint.get(f"{prefix}.g_idx", None)

                weight = dequantize_gptq(
                    qweight, qzeros, scales, g_idx,
                    gptq_config.bits, gptq_config.group_size
                )
            else:
                weight = checkpoint.get(f"{prefix}.weight")

            new_state_dict[f"layers.{i}.self_attn.q_proj.weight"] = weight

            # O projection
            prefix = f"model.layers.{i}.self_attn.o_proj"
            if f"{prefix}.qweight" in checkpoint:
                qweight = checkpoint[f"{prefix}.qweight"]
                qzeros = checkpoint[f"{prefix}.qzeros"]
                scales = checkpoint[f"{prefix}.scales"]
                g_idx = checkpoint.get(f"{prefix}.g_idx", None)

                weight = dequantize_gptq(
                    qweight, qzeros, scales, g_idx,
                    gptq_config.bits, gptq_config.group_size
                )
            else:
                weight = checkpoint.get(f"{prefix}.weight")

            new_state_dict[f"layers.{i}.self_attn.o_proj.weight"] = weight

            # KV projection (already merged)
            prefix = f"model.layers.{i}.self_attn.kv_proj"
            if f"{prefix}.qweight" in checkpoint:
                qweight = checkpoint[f"{prefix}.qweight"]
                qzeros = checkpoint[f"{prefix}.qzeros"]
                scales = checkpoint[f"{prefix}.scales"]
                g_idx = checkpoint.get(f"{prefix}.g_idx", None)

                weight = dequantize_gptq(
                    qweight, qzeros, scales, g_idx,
                    gptq_config.bits, gptq_config.group_size
                )
            else:
                weight = checkpoint.get(f"{prefix}.weight",
                                       checkpoint.get(f"layers.{i}.self_attn.kv_proj_weight"))

            if weight is not None:
                new_state_dict[f"layers.{i}.self_attn.kv_proj_weight"] = weight

        # MLP projections
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            prefix = f"model.layers.{i}.mlp.{proj}"

            if f"{prefix}.qweight" in checkpoint:
                # Load and dequantize
                qweight = checkpoint[f"{prefix}.qweight"]
                qzeros = checkpoint[f"{prefix}.qzeros"]
                scales = checkpoint[f"{prefix}.scales"]
                g_idx = checkpoint.get(f"{prefix}.g_idx", None)

                weight = dequantize_gptq(
                    qweight, qzeros, scales, g_idx,
                    gptq_config.bits, gptq_config.group_size
                )
            else:
                weight = checkpoint.get(f"{prefix}.weight")
                if weight is None:
                    raise ValueError(f"No weight found for {prefix}")

            new_state_dict[f"layers.{i}.mlp.{proj}.weight"] = weight

        # Layer norms (not quantized) - handle different naming conventions
        attention_norm = checkpoint.get(f"model.layers.{i}.input_layernorm.weight") or \
                        checkpoint.get(f"layers.{i}.attention_norm_weight") or \
                        checkpoint.get(f"layers.{i}.input_layernorm_weight")

        ffn_norm = checkpoint.get(f"model.layers.{i}.post_attention_layernorm.weight") or \
                   checkpoint.get(f"layers.{i}.ffn_norm_weight") or \
                   checkpoint.get(f"layers.{i}.post_attention_layernorm_weight")

        if attention_norm is not None:
            new_state_dict[f"layers.{i}.attention_norm_weight"] = attention_norm
        if ffn_norm is not None:
            new_state_dict[f"layers.{i}.ffn_norm_weight"] = ffn_norm

    return new_state_dict


def convert_gptq_qwen2_to_lite_llama(
    checkpoint: Dict[str, torch.Tensor],
    gptq_config: GPTQConfig,
    model_config
) -> Dict[str, torch.Tensor]:
    """Convert GPTQ Qwen2 model to lite_llama format"""
    new_state_dict = {}

    # Load embeddings and norms
    new_state_dict["embed_tokens.weight"] = checkpoint.get("model.embed_tokens.weight")
    new_state_dict["norm_weight"] = checkpoint.get("model.norm.weight")
    new_state_dict["lm_head_weight"] = checkpoint.get("lm_head.weight")

    # Process each layer
    for i in range(model_config.num_layers):
        # Self attention - handle q_proj separately due to bias
        prefix = f"model.layers.{i}.self_attn.q_proj"
        if f"{prefix}.qweight" in checkpoint:
            qweight = checkpoint[f"{prefix}.qweight"]
            qzeros = checkpoint[f"{prefix}.qzeros"]
            scales = checkpoint[f"{prefix}.scales"]
            g_idx = checkpoint.get(f"{prefix}.g_idx", None)

            weight = dequantize_gptq(
                qweight, qzeros, scales, g_idx,
                gptq_config.bits, gptq_config.group_size
            )
        else:
            weight = checkpoint.get(f"{prefix}.weight")

        new_state_dict[f"layers.{i}.self_attn.q_proj_weight"] = weight
        new_state_dict[f"layers.{i}.self_attn.q_proj_bias"] = checkpoint.get(f"{prefix}.bias")

        # Handle k_proj and v_proj for merging
        for proj in ["k_proj", "v_proj"]:
            prefix = f"model.layers.{i}.self_attn.{proj}"

            if f"{prefix}.qweight" in checkpoint:
                qweight = checkpoint[f"{prefix}.qweight"]
                qzeros = checkpoint[f"{prefix}.qzeros"]
                scales = checkpoint[f"{prefix}.scales"]
                g_idx = checkpoint.get(f"{prefix}.g_idx", None)

                weight = dequantize_gptq(
                    qweight, qzeros, scales, g_idx,
                    gptq_config.bits, gptq_config.group_size
                )
            else:
                weight = checkpoint.get(f"{prefix}.weight")

            new_state_dict[f"_temp_{i}_{proj}_weight"] = weight
            new_state_dict[f"_temp_{i}_{proj}_bias"] = checkpoint.get(f"{prefix}.bias")

        # Merge k and v
        k_weight = new_state_dict.pop(f"_temp_{i}_k_proj_weight")
        v_weight = new_state_dict.pop(f"_temp_{i}_v_proj_weight")
        k_bias = new_state_dict.pop(f"_temp_{i}_k_proj_bias")
        v_bias = new_state_dict.pop(f"_temp_{i}_v_proj_bias")

        new_state_dict[f"layers.{i}.self_attn.kv_proj_weight"] = torch.cat([k_weight, v_weight], dim=0)
        new_state_dict[f"layers.{i}.self_attn.kv_proj_bias"] = torch.cat([k_bias, v_bias], dim=0)

        # O projection
        prefix = f"model.layers.{i}.self_attn.o_proj"
        if f"{prefix}.qweight" in checkpoint:
            qweight = checkpoint[f"{prefix}.qweight"]
            qzeros = checkpoint[f"{prefix}.qzeros"]
            scales = checkpoint[f"{prefix}.scales"]
            g_idx = checkpoint.get(f"{prefix}.g_idx", None)

            weight = dequantize_gptq(
                qweight, qzeros, scales, g_idx,
                gptq_config.bits, gptq_config.group_size
            )
        else:
            weight = checkpoint.get(f"{prefix}.weight")

        new_state_dict[f"layers.{i}.self_attn.o_proj_weight"] = weight

        # MLP layers
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            prefix = f"model.layers.{i}.mlp.{proj}"

            if f"{prefix}.qweight" in checkpoint:
                qweight = checkpoint[f"{prefix}.qweight"]
                qzeros = checkpoint[f"{prefix}.qzeros"]
                scales = checkpoint[f"{prefix}.scales"]
                g_idx = checkpoint.get(f"{prefix}.g_idx", None)

                weight = dequantize_gptq(
                    qweight, qzeros, scales, g_idx,
                    gptq_config.bits, gptq_config.group_size
                )
            else:
                weight = checkpoint.get(f"{prefix}.weight")

            new_state_dict[f"layers.{i}.mlp.{proj}.weight"] = weight

        # Layer norms
        new_state_dict[f"layers.{i}.input_layernorm_weight"] = checkpoint.get(
            f"model.layers.{i}.input_layernorm.weight"
        )
        new_state_dict[f"layers.{i}.post_attention_layernorm_weight"] = checkpoint.get(
            f"model.layers.{i}.post_attention_layernorm.weight"
        )

    return new_state_dict


class GPTQModelLoader:
    """Helper class to load GPTQ models"""

    @staticmethod
    def load(checkpoints_dir: str, model_config, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """
        Load GPTQ model and convert to lite_llama format

        Args:
            checkpoints_dir: Directory containing GPTQ model
            model_config: Model configuration
            device: Target device

        Returns:
            State dictionary ready for lite_llama
        """
        print(f"Loading GPTQ model from {checkpoints_dir}")
        start_time = time.time()

        state_dict = convert_gptq_to_lite_llama(checkpoints_dir, model_config)

        # Move to device and convert to fp16
        for key, value in state_dict.items():
            if value is not None:
                state_dict[key] = value.to(device).half()

        print(f"GPTQ model loaded and converted in {time.time() - start_time:.2f}s")
        return state_dict