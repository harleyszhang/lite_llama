from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import tqdm
import json, sys, os
from pathlib import Path

# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lite_llama.models.qwen2 import Qwen2Model, Qwen2Config


def load_config_from_json(json_file_path: str, device: str = "cuda") -> Qwen2Config:
    with open(json_file_path, "r") as f:
        config_dict = json.load(f)

    config = Qwen2Config(config_dict, max_seq_len=2048, device=device)
    return config


def load_original_llama(model_name_or_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.to(device)
    hf_sd = model.state_dict()

    return model, tokenizer, hf_sd


def load_custom_llam(
    model_name_or_path: str, model_config: Qwen2Config, device: str = "cuda"
):
    checkpoints = sorted(Path(model_name_or_path).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {model_name_or_path}"
    ckpt_path = checkpoints[0]
    state_dict = torch.load(ckpt_path, map_location="cuda")

    # 根据设备选择合适的 dtype
    torch.set_default_dtype(torch.half)

    model = Qwen2Model(model_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    new_sd = model.state_dict()

    return model, new_sd


def compare_model_weights(hf_sd, new_sd, model_config, rtol=1e-5, atol=1e-8):
    """
    比较两个模型权重字典的各个参数是否相等。

    Args:
        hf_sd (dict): Hugging Face 模型的 state_dict。
        new_sd (dict): 自定义模型的 state_dict。
        rtol (float): 允许的相对误差。
        atol (float): 允许的绝对误差。

    Returns:
        bool: 如果权重完全匹配，则返回 True, 否则返回 False。
    """

    all_match = True

    # 检查键是否一致
    hf_keys = set(hf_sd.keys())
    new_keys = set(new_sd.keys())

    if hf_keys != new_keys:
        print("键不一致！")
        print("Hugging Face 多出的键:", hf_keys - new_keys)
        print("自定义模型多出的键:", new_keys - hf_keys)
        # all_match = False

    # 映射嵌入层  # 映射归一化层
    mapping = {
        "model.norm.weight": "norm_weight",
        "model.embed_tokens.weight": "embed_tokens.weight",
        "lm_head.weight": "lm_head_weight",
    }

    # 映射层
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

    # 根据 Transformer 层数量生成映射
    for i in range(model_config.num_layers):
        for hf_key, custom_key in layers.items():
            mapped_key = hf_key.format(i=i)  # hf 权重参数字典 key
            custom_mapped_key = custom_key.format(i=i)  # 自定义模型权重参数字典 key
            mapping[mapped_key] = custom_mapped_key

    # 创建新的状态字典
    for hf_key, tensor in tqdm(hf_sd.items(), desc="Mapping weights"):
        custom_key = mapping.get(hf_key, None)
        hf_param = hf_sd[hf_key]
        new_param = new_sd[custom_key]

        if not torch.allclose(hf_param, new_param, rtol=rtol, atol=atol):
            print(f"hf 参数 {hf_key} 不匹配！")
            print(f"Hugging Face 权重: {hf_param}")
            print(f"自定义模型权重: {new_param}")
            all_match = False

    if all_match:
        print("所有权重完全匹配！")
    else:
        print("权重存在不匹配！")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 定义 Qwen2.5-3B 模型权重路径和配置参数
    original_model_path = "/gemini/pretrain/Qwen2.5-3B"
    my_model_path = "/gemini/code/Qwen2.5-3B-Instruct/"
    json_file_path = os.path.join(original_model_path, "config.json")  # JSON 文件的路径
    model_config = load_config_from_json(json_file_path, device)  # 加载配置

    # 加载原始 hf 模型权重
    original_model, tokenizer, hf_sd = load_original_llama(original_model_path, device)
    # 加载自定义模型权重
    custom_model = Qwen2Model(model_config)
    custom_model, new_sd = load_custom_llam(my_model_path, model_config, device)

    compare_model_weights(hf_sd, new_sd, model_config)

    for name, param in custom_model.named_parameters():
        print(name, param.shape)
