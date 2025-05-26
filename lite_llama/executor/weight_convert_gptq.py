from tqdm.auto import tqdm
import torch
import os
import shutil
import glob
import os.path as osp
from typing import Dict, Optional
import gc
from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from auto_gptq.modeling import BaseGPTQForCausalLM
except ImportError:
    raise ImportError(
        "Please install auto-gptq: pip install auto-gptq"
    )


def get_calibration_data(model_id: str, dataset_name: str, tokenizer, nsamples: int = 128, seqlen: int = 2048):
    """
    Prepare calibration dataset for GPTQ quantization.
    """
    if dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        text_column = "text"
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        text_column = "text"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    calibration_data = []

    for data in tqdm(dataset, desc="Loading calibration data"):
        text = data[text_column]
        if len(text.strip()) > 10:  # Skip very short texts
            inputs = tokenizer(
                text,
                truncation=True,
                max_length=seqlen,
                return_tensors="pt"
            )
            if inputs["input_ids"].shape[1] >= seqlen // 2:  # Ensure reasonable length
                calibration_data.append({
                    "input_ids": inputs["input_ids"][0],
                    "attention_mask": inputs["attention_mask"][0]
                })

                if len(calibration_data) >= nsamples:
                    break

    return calibration_data


def build_new_weight_dir_gptq(checkpoints_dir: str, new_sd: Dict[str, torch.Tensor], bits: int):
    """
    Save GPTQ quantized model weights and build new weight directory.
    """
    model_id = osp.basename(osp.normpath(checkpoints_dir))
    current_dir = osp.dirname(osp.abspath(__file__))
    my_weight_dir = osp.join(
        current_dir, f"../../my_weight/{model_id}-{bits}bit-GPTQ"
    )
    os.makedirs(my_weight_dir, exist_ok=True)

    # Save quantized model state dict
    torch.save(
        new_sd,
        osp.join(my_weight_dir, f"{model_id}-{bits}bit-GPTQ.pth"),
        _use_new_zipfile_serialization=True,
    )

    # Copy JSON files
    json_files = glob.glob(osp.join(checkpoints_dir, "*.json"))
    for file_path in json_files:
        shutil.copy(file_path, my_weight_dir)
        print(f"已复制: {file_path} -> {my_weight_dir}")

    # Copy tokenizer files
    if osp.exists(osp.join(checkpoints_dir, "tokenizer.model")):
        shutil.copy(osp.join(checkpoints_dir, "tokenizer.model"), my_weight_dir)

    # Save quantization config
    quant_config = {
        "bits": bits,
        "quantization_method": "gptq",
        "model_id": model_id
    }

    import json
    with open(osp.join(my_weight_dir, "quantization_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)


def quantize_and_convert_weights(
        model,
        checkpoints_dir: str,
        bits: int = 4,
        group_size: int = 128,
        act_order: bool = False,
        calibration_dataset: str = "c4",
        nsamples: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Quantize model with GPTQ and return quantized state dict.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoints_dir)

    # Prepare quantization config
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=0.01,
        desc_act=act_order,
        static_groups=False,
        sym=True,
        true_sequential=True,
        model_name_or_path=checkpoints_dir,
        model_file_base_name="model"
    )

    # Get calibration data
    calibration_data = get_calibration_data(
        checkpoints_dir,
        calibration_dataset,
        tokenizer,
        nsamples=nsamples
    )

    # Clear GPU cache before quantization
    torch.cuda.empty_cache()
    gc.collect()

    # Quantize the model
    print(f"Starting GPTQ quantization with {bits} bits...")
    model.quantize(calibration_data, quantize_config)

    # Get quantized state dict
    quantized_sd = model.state_dict()

    # Clear memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return quantized_sd


def convert_qwen2_hf_to_litellama_gptq(
        checkpoints_dir: str,
        model,
        num_layers: int,
        bits: int = 4,
        group_size: int = 128,
        act_order: bool = False,
        calibration_dataset: str = "c4",
        nsamples: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Convert Qwen2 HF model to LiteLLaMA format with GPTQ quantization.
    """
    # First quantize the model
    quantized_sd = quantize_and_convert_weights(
        model,
        checkpoints_dir,
        bits=bits,
        group_size=group_size,
        act_order=act_order,
        calibration_dataset=calibration_dataset,
        nsamples=nsamples,
    )

    # Mapping for base layers
    mapping = {
        "model.norm.weight": "norm_weight",
        "model.embed_tokens.weight": "embed_tokens.weight",
        "lm_head.weight": "lm_head_weight",
    }

    # Mapping for transformer layers
    layers = {
        "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj_weight",
        "model.layers.{i}.self_attn.q_proj.bias": "layers.{i}.self_attn.q_proj_bias",
        "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj_weight",
        "model.layers.{i}.self_attn.k_proj.bias": "layers.{i}.self_attn.k_proj_bias",
        "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj_weight",
        "model.layers.{i}.self_attn.v_proj.bias": "layers.{i}.self_attn.v_proj_bias",
        "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj_weight",
        "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.mlp.gate_proj.weight",
        "model.layers.{i}.mlp.up_proj.weight": "layers.{i}.mlp.up_proj.weight",
        "model.layers.{i}.mlp.down_proj.weight": "layers.{i}.mlp.down_proj.weight",
        "model.layers.{i}.input_layernorm.weight": "layers.{i}.input_layernorm_weight",
        "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm_weight",
    }

    # Add GPTQ-specific mappings
    gptq_layers = {
        "model.layers.{i}.self_attn.q_proj.qweight": "layers.{i}.self_attn.q_proj_qweight",
        "model.layers.{i}.self_attn.q_proj.qzeros": "layers.{i}.self_attn.q_proj_qzeros",
        "model.layers.{i}.self_attn.q_proj.scales": "layers.{i}.self_attn.q_proj_scales",
        "model.layers.{i}.self_attn.k_proj.qweight": "layers.{i}.self_attn.k_proj_qweight",
        "model.layers.{i}.self_attn.k_proj.qzeros": "layers.{i}.self_attn.k_proj_qzeros",
        "model.layers.{i}.self_attn.k_proj.scales": "layers.{i}.self_attn.k_proj_scales",
        "model.layers.{i}.self_attn.v_proj.qweight": "layers.{i}.self_attn.v_proj_qweight",
        "model.layers.{i}.self_attn.v_proj.qzeros": "layers.{i}.self_attn.v_proj_qzeros",
        "model.layers.{i}.self_attn.v_proj.scales": "layers.{i}.self_attn.v_proj_scales",
        "model.layers.{i}.self_attn.o_proj.qweight": "layers.{i}.self_attn.o_proj_qweight",
        "model.layers.{i}.self_attn.o_proj.qzeros": "layers.{i}.self_attn.o_proj_qzeros",
        "model.layers.{i}.self_attn.o_proj.scales": "layers.{i}.self_attn.o_proj_scales",
        "model.layers.{i}.mlp.gate_proj.qweight": "layers.{i}.mlp.gate_proj_qweight",
        "model.layers.{i}.mlp.gate_proj.qzeros": "layers.{i}.mlp.gate_proj_qzeros",
        "model.layers.{i}.mlp.gate_proj.scales": "layers.{i}.mlp.gate_proj_scales",
        "model.layers.{i}.mlp.up_proj.qweight": "layers.{i}.mlp.up_proj_qweight",
        "model.layers.{i}.mlp.up_proj.qzeros": "layers.{i}.mlp.up_proj_qzeros",
        "model.layers.{i}.mlp.up_proj.scales": "layers.{i}.mlp.up_proj_scales",
        "model.layers.{i}.mlp.down_proj.qweight": "layers.{i}.mlp.down_proj_qweight",
        "model.layers.{i}.mlp.down_proj.qzeros": "layers.{i}.mlp.down_proj_qzeros",
        "model.layers.{i}.mlp.down_proj.scales": "layers.{i}.mlp.down_proj_scales",
    }

    # Generate mappings for all layers
    for i in range(num_layers):
        for hf_key, custom_key in layers.items():
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)
        for hf_key, custom_key in gptq_layers.items():
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)

    # Create new state dict with converted keys
    new_sd = {}
    for hf_key, tensor in tqdm(quantized_sd.items(), desc="Mapping GPTQ weights"):
        custom_key = mapping.get(hf_key, None)
        if custom_key is not None:
            new_sd[custom_key] = tensor
        else:
            print(f"Warning: Unmapped key {hf_key}")

    # Merge k_proj and v_proj for GPTQ
    for i in range(num_layers):
        # For regular weights (if they exist)
        k_key = f"layers.{i}.self_attn.k_proj_weight"
        v_key = f"layers.{i}.self_attn.v_proj_weight"
        k_bias_key = f"layers.{i}.self_attn.k_proj_bias"
        v_bias_key = f"layers.{i}.self_attn.v_proj_bias"

        if k_key in new_sd and v_key in new_sd:
            # Merge weights
            kv_tensor = torch.cat([new_sd[k_key], new_sd[v_key]], dim=0)
            new_sd[f"layers.{i}.self_attn.kv_proj_weight"] = kv_tensor
            del new_sd[k_key]
            del new_sd[v_key]

            # Merge biases if they exist
            if k_bias_key in new_sd and v_bias_key in new_sd:
                kv_bias_tensor = torch.cat([new_sd[k_bias_key], new_sd[v_bias_key]], dim=0)
                new_sd[f"layers.{i}.self_attn.kv_proj_bias"] = kv_bias_tensor
                del new_sd[k_bias_key]
                del new_sd[v_bias_key]

        # For GPTQ quantized weights
        k_qweight = f"layers.{i}.self_attn.k_proj_qweight"
        v_qweight = f"layers.{i}.self_attn.v_proj_qweight"
        k_qzeros = f"layers.{i}.self_attn.k_proj_qzeros"
        v_qzeros = f"layers.{i}.self_attn.v_proj_qzeros"
        k_scales = f"layers.{i}.self_attn.k_proj_scales"
        v_scales = f"layers.{i}.self_attn.v_proj_scales"

        if k_qweight in new_sd and v_qweight in new_sd:
            # Merge quantized weights
            kv_qweight = torch.cat([new_sd[k_qweight], new_sd[v_qweight]], dim=0)
            kv_qzeros = torch.cat([new_sd[k_qzeros], new_sd[v_qzeros]], dim=0)
            kv_scales = torch.cat([new_sd[k_scales], new_sd[v_scales]], dim=0)

            new_sd[f"layers.{i}.self_attn.kv_proj_qweight"] = kv_qweight
            new_sd[f"layers.{i}.self_attn.kv_proj_qzeros"] = kv_qzeros
            new_sd[f"layers.{i}.self_attn.kv_proj_scales"] = kv_scales

            # Remove original k and v projections
            del new_sd[k_qweight]
            del new_sd[v_qweight]
            del new_sd[k_qzeros]
            del new_sd[v_qzeros]
            del new_sd[k_scales]
            del new_sd[v_scales]

    # Save the quantized weights
    build_new_weight_dir_gptq(checkpoints_dir, new_sd, bits)

    print(f"GPTQ quantization complete. Model saved with {bits}-bit precision.")
    return new_sd


def convert_llama_hf_to_litellama_gptq(
        checkpoints_dir: str,
        model,
        num_layers: int,
        bits: int = 4,
        group_size: int = 128,
        act_order: bool = False,
        calibration_dataset: str = "c4",
        nsamples: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Convert Llama HF model to LiteLLaMA format with GPTQ quantization.
    """
    # First quantize the model
    quantized_sd = quantize_and_convert_weights(
        model,
        checkpoints_dir,
        bits=bits,
        group_size=group_size,
        act_order=act_order,
        calibration_dataset=calibration_dataset,
        nsamples=nsamples,
    )

    # Mapping for base layers
    mapping = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm_weight",
        "lm_head.weight": "lm_head.weight",
    }

    # Mapping for transformer layers
    layers = {
        "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj.weight",
        "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj.weight",
        "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj.weight",
        "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj.weight",
        "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.mlp.gate_proj.weight",
        "model.layers.{i}.mlp.up_proj.weight": "layers.{i}.mlp.up_proj.weight",
        "model.layers.{i}.mlp.down_proj.weight": "layers.{i}.mlp.down_proj.weight",
        "model.layers.{i}.input_layernorm.weight": "layers.{i}.attention_norm_weight",
        "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.ffn_norm_weight",
    }

    # Add GPTQ-specific mappings
    gptq_layers = {
        "model.layers.{i}.self_attn.q_proj.qweight": "layers.{i}.self_attn.q_proj_qweight",
        "model.layers.{i}.self_attn.q_proj.qzeros": "layers.{i}.self_attn.q_proj_qzeros",
        "model.layers.{i}.self_attn.q_proj.scales": "layers.{i}.self_attn.q_proj_scales",
        "model.layers.{i}.self_attn.k_proj.qweight": "layers.{i}.self_attn.k_proj_qweight",
        "model.layers.{i}.self_attn.k_proj.qzeros": "layers.{i}.self_attn.k_proj_qzeros",
        "model.layers.{i}.self_attn.k_proj.scales": "layers.{i}.self_attn.k_proj_scales",
        "model.layers.{i}.self_attn.v_proj.qweight": "layers.{i}.self_attn.v_proj_qweight",
        "model.layers.{i}.self_attn.v_proj.qzeros": "layers.{i}.self_attn.v_proj_qzeros",
        "model.layers.{i}.self_attn.v_proj.scales": "layers.{i}.self_attn.v_proj_scales",
        "model.layers.{i}.self_attn.o_proj.qweight": "layers.{i}.self_attn.o_proj_qweight",
        "model.layers.{i}.self_attn.o_proj.qzeros": "layers.{i}.self_attn.o_proj_qzeros",
        "model.layers.{i}.self_attn.o_proj.scales": "layers.{i}.self_attn.o_proj_scales",
        "model.layers.{i}.mlp.gate_proj.qweight": "layers.{i}.mlp.gate_proj_qweight",
        "model.layers.{i}.mlp.gate_proj.qzeros": "layers.{i}.mlp.gate_proj_qzeros",
        "model.layers.{i}.mlp.gate_proj.scales": "layers.{i}.mlp.gate_proj_scales",
        "model.layers.{i}.mlp.up_proj.qweight": "layers.{i}.mlp.up_proj_qweight",
        "model.layers.{i}.mlp.up_proj.qzeros": "layers.{i}.mlp.up_proj_qzeros",
        "model.layers.{i}.mlp.up_proj.scales": "layers.{i}.mlp.up_proj_scales",
        "model.layers.{i}.mlp.down_proj.qweight": "layers.{i}.mlp.down_proj_qweight",
        "model.layers.{i}.mlp.down_proj.qzeros": "layers.{i}.mlp.down_proj_qzeros",
        "model.layers.{i}.mlp.down_proj.scales": "layers.{i}.mlp.down_proj_scales",
    }

    # Generate mappings for all layers
    for i in range(num_layers):
        for hf_key, custom_key in layers.items():
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)
        for hf_key, custom_key in gptq_layers.items():
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)

    # Create new state dict with converted keys
    new_sd = {}
    for hf_key, tensor in tqdm(quantized_sd.items(), desc="Mapping GPTQ weights"):
        custom_key = mapping.get(hf_key, None)
        if custom_key is not None:
            new_sd[custom_key] = tensor
        else:
            print(f"Warning: Unmapped key {hf_key}")

    # Merge k_proj and v_proj
    for i in range(num_layers):
        # Handle regular weights if they exist
        k_key = f"layers.{i}.self_attn.k_proj.weight"
        v_key = f"layers.{i}.self_attn.v_proj.weight"
        if k_key in new_sd and v_key in new_sd:
            kv_tensor = torch.cat([new_sd[k_key], new_sd[v_key]], dim=0)
            new_sd[f"layers.{i}.self_attn.kv_proj_weight"] = kv_tensor
            del new_sd[k_key]
            del new_sd[v_key]

        # Handle GPTQ quantized weights
        k_qweight = f"layers.{i}.self_attn.k_proj_qweight"
        v_qweight = f"layers.{i}.self_attn.v_proj_qweight"
        k_qzeros = f"layers.{i}.self_attn.k_proj_qzeros"
        v_qzeros = f"layers.{i}.self_attn.v_proj_qzeros"
        k_scales = f"layers.{i}.self_attn.k_proj_scales"
        v_scales = f"layers.{i}.self_attn.v_proj_scales"

        if k_qweight in new_sd and v_qweight in new_sd:
            # Merge quantized weights
            kv_qweight = torch.cat([new_sd[k_qweight], new_sd[v_qweight]], dim=0)
            kv_qzeros = torch.cat([new_sd[k_qzeros], new_sd[v_qzeros]], dim=0)
            kv_scales = torch.cat([new_sd[k_scales], new_sd[v_scales]], dim=0)

            new_sd[f"layers.{i}.self_attn.kv_proj_qweight"] = kv_qweight
            new_sd[f"layers.{i}.self_attn.kv_proj_qzeros"] = kv_qzeros
            new_sd[f"layers.{i}.self_attn.kv_proj_scales"] = kv_scales

            # Remove original k and v projections
            del new_sd[k_qweight]
            del new_sd[v_qweight]
            del new_sd[k_qzeros]
            del new_sd[v_qzeros]
            del new_sd[k_scales]
            del new_sd[v_scales]

    # Save the quantized weights
    build_new_weight_dir_gptq(checkpoints_dir, new_sd, bits)

    print(f"GPTQ quantization complete. Model saved with {bits}-bit precision.")
    return new_sd


def convert_llavallama_hf_to_litellama_gptq(
        checkpoints_dir: str,
        model,
        num_layers: int,
        bits: int = 4,
        group_size: int = 128,
        act_order: bool = False,
        calibration_dataset: str = "c4",
        nsamples: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Convert LLaVA-Llama HF model to LiteLLaMA format with GPTQ quantization.
    """
    # First quantize the model
    quantized_sd = quantize_and_convert_weights(
        model,
        checkpoints_dir,
        bits=bits,
        group_size=group_size,
        act_order=act_order,
        calibration_dataset=calibration_dataset,
        nsamples=nsamples,
    )

    # Mapping for base layers
    mapping = {
        "language_model.model.embed_tokens.weight": "language_model.embed_tokens.weight",
        "language_model.model.norm.weight": "language_model.norm_weight",
        "language_model.lm_head.weight": "language_model.lm_head.weight",
    }

    # Mapping for transformer layers
    layers = {
        "language_model.model.layers.{i}.self_attn.q_proj.weight": "language_model.layers.{i}.self_attn.q_proj.weight",
        "language_model.model.layers.{i}.self_attn.k_proj.weight": "language_model.layers.{i}.self_attn.k_proj.weight",
        "language_model.model.layers.{i}.self_attn.v_proj.weight": "language_model.layers.{i}.self_attn.v_proj.weight",
        "language_model.model.layers.{i}.self_attn.o_proj.weight": "language_model.layers.{i}.self_attn.o_proj.weight",
        "language_model.model.layers.{i}.mlp.gate_proj.weight": "language_model.layers.{i}.mlp.gate_proj.weight",
        "language_model.model.layers.{i}.mlp.up_proj.weight": "language_model.layers.{i}.mlp.up_proj.weight",
        "language_model.model.layers.{i}.mlp.down_proj.weight": "language_model.layers.{i}.mlp.down_proj.weight",
        "language_model.model.layers.{i}.input_layernorm.weight": "language_model.layers.{i}.attention_norm_weight",
        "language_model.model.layers.{i}.post_attention_layernorm.weight": "language_model.layers.{i}.ffn_norm_weight",
    }

    # Add GPTQ-specific mappings
    gptq_layers = {
        "language_model.model.layers.{i}.self_attn.q_proj.qweight": "language_model.layers.{i}.self_attn.q_proj_qweight",
        "language_model.model.layers.{i}.self_attn.q_proj.qzeros": "language_model.layers.{i}.self_attn.q_proj_qzeros",
        "language_model.model.layers.{i}.self_attn.q_proj.scales": "language_model.layers.{i}.self_attn.q_proj_scales",
        "language_model.model.layers.{i}.self_attn.k_proj.qweight": "language_model.layers.{i}.self_attn.k_proj_qweight",
        "language_model.model.layers.{i}.self_attn.k_proj.qzeros": "language_model.layers.{i}.self_attn.k_proj_qzeros",
        "language_model.model.layers.{i}.self_attn.k_proj.scales": "language_model.layers.{i}.self_attn.k_proj_scales",
        "language_model.model.layers.{i}.self_attn.v_proj.qweight": "language_model.layers.{i}.self_attn.v_proj_qweight",
        "language_model.model.layers.{i}.self_attn.v_proj.qzeros": "language_model.layers.{i}.self_attn.v_proj_qzeros",
        "language_model.model.layers.{i}.self_attn.v_proj.scales": "language_model.layers.{i}.self_attn.v_proj_scales",
        "language_model.model.layers.{i}.self_attn.o_proj.qweight": "language_model.layers.{i}.self_attn.o_proj_qweight",
        "language_model.model.layers.{i}.self_attn.o_proj.qzeros": "language_model.layers.{i}.self_attn.o_proj_qzeros",
        "language_model.model.layers.{i}.self_attn.o_proj.scales": "language_model.layers.{i}.self_attn.o_proj_scales",
        "language_model.model.layers.{i}.mlp.gate_proj.qweight": "language_model.layers.{i}.mlp.gate_proj_qweight",
        "language_model.model.layers.{i}.mlp.gate_proj.qzeros": "language_model.layers.{i}.mlp.gate_proj_qzeros",
        "language_model.model.layers.{i}.mlp.gate_proj.scales": "language_model.layers.{i}.mlp.gate_proj_scales",
        "language_model.model.layers.{i}.mlp.up_proj.qweight": "language_model.layers.{i}.mlp.up_proj_qweight",
        "language_model.model.layers.{i}.mlp.up_proj.qzeros": "language_model.layers.{i}.mlp.up_proj_qzeros",
        "language_model.model.layers.{i}.mlp.up_proj.scales": "language_model.layers.{i}.mlp.up_proj_scales",
        "language_model.model.layers.{i}.mlp.down_proj.qweight": "language_model.layers.{i}.mlp.down_proj_qweight",
        "language_model.model.layers.{i}.mlp.down_proj.qzeros": "language_model.layers.{i}.mlp.down_proj_qzeros",
        "language_model.model.layers.{i}.mlp.down_proj.scales": "language_model.layers.{i}.mlp.down_proj_scales",
    }

    # Generate mappings for all layers
    for i in range(num_layers):
        for hf_key, custom_key in layers.items():
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)
        for hf_key, custom_key in gptq_layers.items():
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)

    # Create new state dict with converted keys
    new_sd = {}
    for hf_key, tensor in tqdm(quantized_sd.items(), desc="Mapping GPTQ weights"):
        custom_key = mapping.get(hf_key, None)
        if custom_key is not None:
            new_sd[custom_key] = tensor
        else:
            # Keep vision model and other components as-is
            new_sd[hf_key] = tensor
            print(f"Warning: Unmapped key {hf_key}")

    # Merge k_proj and v_proj for language model
    for i in tqdm(range(num_layers), desc="Mapping kv fused weights"):
        # Handle regular weights if they exist
        k_key = f"language_model.layers.{i}.self_attn.k_proj.weight"
        v_key = f"language_model.layers.{i}.self_attn.v_proj.weight"
        if k_key in new_sd and v_key in new_sd:
            kv_tensor = torch.cat([new_sd[k_key], new_sd[v_key]], dim=0)
            new_sd[f"language_model.layers.{i}.self_attn.kv_proj_weight"] = kv_tensor
            del new_sd[k_key]
            del new_sd[v_key]

        # Handle GPTQ quantized weights
        k_qweight = f"language_model.layers.{i}.self_attn.k_proj_qweight"
        v_qweight = f"language_model.layers.{i}.self_attn.v_proj_qweight"
        k_qzeros = f"language_model.layers.{i}.self_attn.k_proj_qzeros"
        v_qzeros = f"language_model.layers.{i}.self_attn.v_proj_qzeros"
        k_scales = f"language_model.layers.{i}.self_attn.k_proj_scales"
        v_scales = f"language_model.layers.{i}.self_attn.v_proj_scales"

        if k_qweight in new_sd and v_qweight in new_sd:
            # Merge quantized weights
            kv_qweight = torch.cat([new_sd[k_qweight], new_sd[v_qweight]], dim=0)
            kv_qzeros = torch.cat([new_sd[k_qzeros], new_sd[v_qzeros]], dim=0)
            kv_scales = torch.cat([new_sd[k_scales], new_sd[v_scales]], dim=0)

            new_sd[f"language_model.layers.{i}.self_attn.kv_proj_qweight"] = kv_qweight
            new_sd[f"language_model.layers.{i}.self_attn.kv_proj_qzeros"] = kv_qzeros
            new_sd[f"language_model.layers.{i}.self_attn.kv_proj_scales"] = kv_scales

            print(f"Merged GPTQ k/v projections for layer {i}")

            # Remove original k and v projections
            del new_sd[k_qweight]
            del new_sd[v_qweight]
            del new_sd[k_qzeros]
            del new_sd[v_qzeros]
            del new_sd[k_scales]
            del new_sd[v_scales]

    # Save the quantized weights
    build_new_weight_dir_gptq(checkpoints_dir, new_sd, bits)

    print(f"GPTQ quantization complete. Model saved with {bits}-bit precision.")
    return new_sd