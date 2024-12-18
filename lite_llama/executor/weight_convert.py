from tqdm.auto import tqdm
import torch, os, shutil, glob
from typing import Dict
from ..models.qwen2 import Qwen2Config

def build_new_weight_dir(checkpoints_dir:str, new_sd):
    # 保存 lite_llama 模型权重并构建新的权重目录
    model_id = os.path.basename(os.path.normpath(checkpoints_dir))
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在的目录
    my_weight_dir = os.path.join(current_dir, "../../my_weight/" + model_id) # 项目所在根目录
    os.makedirs(my_weight_dir, exist_ok=True) # 创建文件夹（如果不存在）
    
    # 保存模型的状态字典。
    torch.save(new_sd, os.path.join(my_weight_dir, model_id + ".pth"), _use_new_zipfile_serialization=True)

    # 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(checkpoints_dir, "*.json"))
    for file_path in json_files:
        shutil.copy(file_path, my_weight_dir) # 复制 hf 权重目录的所有 json 文件到新的目录
        print(f"已复制: {file_path} -> {my_weight_dir}")

def convert_qwen2_hf_to_litellama(
    checkpoints_dir: str, 
    hf_sd, 
    num_layers, 
    print_params: bool = True,
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

    # 根据 Transformer 层数量生成映射
    for i in range(num_layers):
        for hf_key, custom_key in layers.items():
            # 左边是 hf 权重参数字典 key, 右边是自定义模型权重参数字典 key
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)

    # 创建新的状态字典
    new_sd = {}
    for hf_key, tensor in tqdm(hf_sd.items(), desc="Mapping weights"):
        bigger = (tensor > 1).any()
        print(f"key {hf_key}, contains bigger {bigger}")
        custom_key = mapping.get(hf_key, None)
        if custom_key is not None:
            new_sd[custom_key] = tensor # 浅拷贝
        else:
            print(f"custom_key: {custom_key}, hf_key: {hf_key}")
            pass  # 忽略未映射的权重
    
    # del hf_sd

    # 进行 kv_proj 合并操作
    for i in range(num_layers):
        k_key = f"layers.{i}.self_attn.k_proj_weight"
        v_key = f"layers.{i}.self_attn.v_proj_weight"
        k_bias_key = f"layers.{i}.self_attn.k_proj_bias"
        v_bias_key = f"layers.{i}.self_attn.v_proj_bias"
        
        if k_key in new_sd and v_key in new_sd and k_bias_key in new_sd and v_bias_key in new_sd:
            # 1. kv weight 权重合并
            k_tensor = new_sd[k_key]
            v_tensor = new_sd[v_key]
            # 按最后一维拼接后成为 [2 * hidden_size, hidden_size]
            kv_tensor = torch.cat([k_tensor, v_tensor], dim=0)
            print(f"{k_key} and {v_key} concat success!")

            # 新增 kv_proj.weight
            kv_key = f"layers.{i}.self_attn.kv_proj_weight"
            new_sd[kv_key] = kv_tensor
            print(f"new {kv_key} key init success!")

            # 2. kv bias 权重合并
            k_bias_tensor = new_sd[k_bias_key]
            v_bias_tensor = new_sd[v_bias_key]
            kv_bias_tensor = torch.cat([k_bias_tensor, v_bias_tensor], dim=0)

            kv_bias_key = f"layers.{i}.self_attn.kv_proj_bias"
            new_sd[kv_bias_key] = kv_bias_tensor

            # 删除原来的 k_proj, v_proj
            del new_sd[k_key]
            del new_sd[v_key]
            del new_sd[k_bias_key]
            del new_sd[v_bias_key]

    # 保存转换好的自定义权重
    build_new_weight_dir(checkpoints_dir, new_sd)
    
    if print_params:
        # 打印预训练模型的参数名称
        print("Pretrained model parameters:")
        for name, parameters in hf_sd.items():
            print(name, parameters.shape)

        # 打印自定义模型的参数名称
        print("Custom model parameters:")
        for name, parameters in new_sd.items():
            print(name, parameters.shape)

    # return new_sd

def convert_llama_torch_to_litellama(checkpoints_dir, hf_sd, num_layers):
    """
    将 pytorch bin 格式的模型的权重字典转换为自定义模型的权重字典。

    参数:
        checkpoints_dir: pytorch 模型的目录
        hf_sd (dict): pytorch 模型的状态字典。

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
        "layers.{i}.attention.wq.weight": "layers.{i}.attention.q_proj.weight",
        "layers.{i}.attention.wk.weight": "layers.{i}.attention.k_proj.weight",
        "layers.{i}.attention.wv.weight": "layers.{i}.attention.v_proj.weight",
        "layers.{i}.attention.wo.weight": "layers.{i}.attention.o_proj.weight",

        "layers.{i}.feed_forward.w1.weight": "layers.{i}.feed_forward.gate_proj.weight",
        "layers.{i}.feed_forward.w3.weight": "layers.{i}.feed_forward.up_proj.weight",
        "layers.{i}.feed_forward.w2.weight": "layers.{i}.feed_forward.down_proj.weight",

        "layers.{i}.attention_norm.weight": "layers.{i}.attention_norm_weight",
        "layers.{i}.ffn_norm.weight": "layers.{i}.ffn_norm_weight",
    }

    # 根据 Transformer 层数量生成映射
    for i in range(num_layers):
        for hf_key, custom_key in layers.items():
            # 左边是 hf 权重参数字典 key, 右边是自定义模型权重参数字典 key
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)

    # 创建新的状态字典
    new_sd = {}
    for hf_key, tensor in tqdm(hf_sd.items(), desc="Mapping weights"):
        custom_key = mapping.get(hf_key, None)
        if custom_key is not None:
            new_sd[custom_key] = tensor
        else:
            print(f"Warning: Unmapped key {hf_key}")
    
    del hf_sd

    build_new_weight_dir(checkpoints_dir, new_sd)
    return new_sd

def convert_llama_hf_to_litellama(checkpoints_dir, hf_sd, num_layers):
    """
    将 hf 格式的模型的权重字典转换为自定义模型的权重字典。

    参数:
        checkpoints_dir: Hugging Face 模型的目录
        hf_sd (dict): Hugging Face 模型的状态字典。

    返回:
        dict: 转换后的状态字典。
    """
    mapping = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm_weight", 
        "lm_head.weight": "lm_head.weight"
    }

    layers = {
        # key 是原始权重值, value 是自定义模型结构权重参数
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

    # 根据 Transformer 层数量生成映射
    for i in range(num_layers):
        for hf_key, custom_key in layers.items():
            # 左边是 hf 权重参数字典 key, 右边是自定义模型权重参数字典 key
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)

    # 创建新的状态字典
    new_sd = {}
    for hf_key, tensor in tqdm(hf_sd.items(), desc="Mapping weights"):
        custom_key = mapping.get(hf_key, None)
        if custom_key is not None:
            new_sd[custom_key] = tensor
        else:
            print(f"Warning: Unmapped key {hf_key}")
    
    # 进行 kv_proj 合并操作
    for i in range(num_layers):
        k_key = f"layers.{i}.self_attn.k_proj.weight"
        v_key = f"layers.{i}.self_attn.v_proj.weight"
        if k_key in new_sd and v_key in new_sd:
            k_tensor = new_sd[k_key]
            v_tensor = new_sd[v_key]
            # 假设 k_proj, v_proj 的 shape 都是 [hidden_size, hidden_size]
            # 按最后一维拼接后成为 [2 * hidden_size, hidden_size]
            kv_tensor = torch.cat([k_tensor, v_tensor], dim=0)
            
            # 新增 kv_proj.weight
            kv_key = f"layers.{i}.self_attn.kv_proj_weight"
            new_sd[kv_key] = kv_tensor
            
            # 删除原来的 k_proj, v_proj
            del new_sd[k_key]
            del new_sd[v_key]

    for name, parameters in new_sd.items():
        print(name, parameters.shape)

    # 将处理后的权重保存到指定目录
    build_new_weight_dir(checkpoints_dir, new_sd)
    
def convert_llavallama_hf_to_litellama(checkpoints_dir, hf_sd, num_layers):
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
        "language_model.lm_head.weight": "language_model.lm_head.weight",
    }

    layers = {
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
    }

    # 根据 Transformer 层数量生成映射
    for i in range(num_layers):
        for hf_key, custom_key in layers.items():
            # 左边是 hf 权重参数字典 key, 右边是自定义模型权重参数字典 key
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)

    # 创建新的状态字典
    new_sd = {}
    for hf_key, tensor in tqdm(hf_sd.items(), desc="Mapping weights"):
        custom_key = mapping.get(hf_key, None)
        if custom_key is not None:
            new_sd[custom_key] = tensor
        else:
            new_sd[hf_key] = tensor
            print(f"Warning: Unmapped key {hf_key}")
    
    # 进行 kv_proj 合并操作
    for i in tqdm(range(num_layers), desc="Mapping kv fusedweights"):
        k_key = f"language_model.layers.{i}.self_attn.k_proj.weight"
        v_key = f"language_model.layers.{i}.self_attn.v_proj.weight"
        if k_key in new_sd and v_key in new_sd:
            k_tensor = new_sd[k_key]
            v_tensor = new_sd[v_key]
            # 假设 k_proj, v_proj 的 shape 都是 [hidden_size, hidden_size]
            # 按最后一维拼接后成为 [2 * hidden_size, hidden_size]
            kv_tensor = torch.cat([k_tensor, v_tensor], dim=0)
            print(f"{k_key} and {k_key} concat success!")

            # 新增 kv_proj.weight
            kv_key = f"language_model.layers.{i}.self_attn.kv_proj_weight"
            new_sd[kv_key] = kv_tensor
            print(f"new {kv_key} key init success!")

            # 删除原来的 k_proj, v_proj
            del new_sd[k_key]
            del new_sd[v_key]

    for name, parameters in new_sd.items():
        print(name, parameters.shape)

    build_new_weight_dir(checkpoints_dir, new_sd)

