from tqdm.auto import tqdm
import torch, os, shutil, glob
from typing import Dict
from ..models.qwen2 import Qwen2Config

def build_new_weight_dir(checkpoints_dir:str, new_sd):
    # 保存 lite_llama 模型权重并构建新的权重目录
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在的目录
    my_weight_dir = os.path.join(current_dir, "../../my_weight/") # 项目所在根目录
    os.makedirs(my_weight_dir, exist_ok=True) # 创建文件夹（如果不存在）
    torch.save(new_sd, os.path.join(my_weight_dir, "my_llama3.2-1B.pth"))

    # 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(checkpoints_dir, "*.json"))
    for file_path in json_files:
        shutil.copy(file_path, my_weight_dir) # 复制 hf 权重目录的所有 json 文件到新的目录
        print(f"已复制: {file_path} -> {my_weight_dir}")

def convert_qwen2_hf_to_litellama(
    checkpoints_dir: str, 
    hf_sd, 
    model_config: Qwen2Config, 
    print_params: bool = False,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    将 Hugging Face 格式的预训练模型的权重字典转换为自定义模型的权重字典。
    """
    # 映射嵌入层、映射归一化层、映射模型最后的输出线性层
    mapping = {
        "model.norm.weight": "norm_weight", 
        "model.embed_tokens.weight": "embed_tokens.weight",
        "lm_head.weight": "lm_head_weight", # 只支持 hf 格式模型权重
    }

    # 映射层
    layers = {
        'model.layers.{i}.self_attn.q_proj.weight': 'layers.{i}.self_attn.q_proj_weight',
        'model.layers.{i}.self_attn.q_proj.bias': 'layers.{i}.self_attn.q_proj_bias',

        'model.layers.{i}.self_attn.k_proj.weight': 'layers.{i}.self_attn.k_proj_weight',
        'model.layers.{i}.self_attn.k_proj.bias': 'layers.{i}.self_attn.k_proj_bias',

        'model.layers.{i}.self_attn.v_proj.weight': 'layers.{i}.self_attn.v_proj_weight',
        'model.layers.{i}.self_attn.v_proj.bias': 'layers.{i}.self_attn.v_proj_bias',

        'model.layers.{i}.self_attn.o_proj.weight': 'layers.{i}.self_attn.o_proj_weight',

        'model.layers.{i}.mlp.gate_proj.weight': 'layers.{i}.mlp.gate_proj.weight',
        'model.layers.{i}.mlp.up_proj.weight': 'layers.{i}.mlp.up_proj.weight',
        'model.layers.{i}.mlp.down_proj.weight': 'layers.{i}.mlp.down_proj.weight',

        'model.layers.{i}.input_layernorm.weight': 'layers.{i}.input_layernorm_weight',
        'model.layers.{i}.post_attention_layernorm.weight': 'layers.{i}.post_attention_layernorm_weight',
    }

    #  根据 Transformer 层数量生成映射
    for i in range(model_config.num_layers):
        for hf_key, custom_key in layers.items():
            mapped_key = hf_key.format(i=i) # hf 权重参数字典 key
            custom_mapped_key = custom_key.format(i=i) # 自定义模型权重参数字典 key
            mapping[mapped_key] = custom_mapped_key

    # 创建新的状态字典
    new_sd = {}
    for hf_key, tensor in tqdm(hf_sd.items(), desc="Mapping weights"):
        custom_key = mapping.get(hf_key, None)
        if custom_key is not None:
            new_sd[custom_key] = tensor # 浅拷贝
        else:
            print(f"custom_key: {custom_key}, hf_key: {hf_key}")
            # 如果某些权重不需要映射，可以选择忽略或处理
            pass  # 忽略未映射的权重
    
    # 保存转换好的自定义权重
    build_new_weight_dir(checkpoints_dir, new_sd)
    # new_sd["lm_head_weight"] = hf_sd["model.embed_tokens.weight"]
    
    if print_params:
        # 打印预训练模型的参数名称
        print("Pretrained model parameters:")
        for name, parameters in hf_sd.items():
            print(name, parameters.shape)

        # 打印自定义模型的参数名称
        print("Custom model parameters:")
        for name, parameters in new_sd.items():
            print(name, parameters.shape)

    return new_sd

def convert_llama_torch_to_litellama(checkpoints_dir, hf_sd, model_config):
    """
    将 pytorch bin 格式的模型的权重字典转换为自定义模型的权重字典。

    参数:
        checkpoints_dir: Hugging Face 模型的目录
        hf_sd (dict): Hugging Face 模型的状态字典。
        model_config (LlamaConfig): 自定义模型的配置参数。

    返回:
        dict: 转换后的状态字典。
    """
    mapping = {
        "tok_embeddings.weight": "embed_tokens.weight",
        "norm.weight": "norm_weight", 
        "output.weight": "lm_head.weight",
    }

    layers = {
        # key 是原始权重值, value 是自定义模型结构权重参数
        "layers.{i}.attention.wq.weight": "layers.{i}.attention.wq.weight",
        "layers.{i}.attention.wk.weight": "layers.{i}.attention.wk.weight",
        "layers.{i}.attention.wv.weight": "layers.{i}.attention.wv.weight",
        "layers.{i}.attention.wo.weight": "layers.{i}.attention.wo.weight",
        "layers.{i}.feed_forward.w1.weight": "layers.{i}.feed_forward.gate_proj.weight",
        "layers.{i}.feed_forward.w3.weight": "layers.{i}.feed_forward.up_proj.weight",
        "layers.{i}.feed_forward.w2.weight": "layers.{i}.feed_forward.down_proj.weight",

        "layers.{i}.attention_norm.weight": "layers.{i}.attention_norm_weight",
        "layers.{i}.ffn_norm.weight": "layers.{i}.ffn_norm_weight",
    }

    # 根据 Transformer 层数量生成映射
    for i in range(model_config.n_layers):
        for hf_key, custom_key in layers.items():
            # 左边是 hf 权重参数字典 key, 右边是自定义模型权重参数字典 key
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)

    # 创建新的状态字典
    new_sd = {}
    for hf_key, tensor in tqdm(hf_sd.items(), desc="Mapping weights"):
        if hf_key in mapping:
            new_sd[custom_key] = tensor
        else:
            print(f"Warning: Unmapped key {hf_key}")
    
    build_new_weight_dir(checkpoints_dir, new_sd)
    return new_sd

def convert_llavallama_hf_to_litellama(checkpoints_dir, hf_sd, model_config):
    """
    将 Hugging Face 模型的权重字典转换为自定义模型的权重字典。

    参数:
        checkpoints_dir: Hugging Face 模型的目录
        hf_sd (dict): Hugging Face 模型的状态字典。
        model_config (LlamaConfig): 自定义模型的配置参数。

    返回:
        dict: 转换后的状态字典。
    """
    mapping = {
        "language_model.model.embed_tokens.weight": "language_model.embed_tokens.weight",
        "language_model.model.norm.weight": "language_model.norm_weight", 
        "language_model.model.lm_head.weight": "language_model.lm_head.weight",
    }

    layers = {
        # key 是原始权重值, value 是自定义模型结构权重参数
        "language_model.model.layers.{i}.attention.wq.weight": "language_model.layers.{i}.attention.wq.weight",
        "language_model.model.layers.{i}.attention.wk.weight": "language_model.layers.{i}.attention.wk.weight",
        "language_model.model.layers.{i}.attention.wv.weight": "language_model.layers.{i}.attention.wv.weight",
        "language_model.model.layers.{i}.attention.wo.weight": "language_model.layers.{i}.attention.wo.weight",

        "language_model.model.layers.{i}.feed_forward.gate_proj.weight": "language_model.layers.{i}.feed_forward.gate_proj.weight",
        "language_model.model.layers.{i}.feed_forward.up_proj.weight": "language_model.layers.{i}.feed_forward.up_proj.weight",
        "language_model.model.layers.{i}.feed_forward.down_proj.weight": "language_model.layers.{i}.feed_forward.down_proj.weight",

        "language_model.model.layers.{i}.input_layernorm.weight": "language_model.layers.{i}.attention_norm_weight",
        "language_model.model.layers.{i}.post_attention_layernorm.weight": "language_model.layers.{i}.ffn_norm_weight",
    }

    # 根据 Transformer 层数量生成映射
    for i in range(model_config.n_layers):
        for hf_key, custom_key in layers.items():
            # 左边是 hf 权重参数字典 key, 右边是自定义模型权重参数字典 key
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)

    # 创建新的状态字典
    new_sd = {}
    for hf_key, tensor in tqdm(hf_sd.items(), desc="Mapping weights"):
        if hf_key in mapping:
            new_sd[custom_key] = tensor
        else:
            print(f"Warning: Unmapped key {hf_key}")
    
    build_new_weight_dir(checkpoints_dir, new_sd)
    return new_sd