from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
from tqdm.auto import tqdm
import json, sys, os
# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lite_llama.lite_llama.llama import Llama, LlamaConfig

def load_original_llama(model_name_or_path: str, device: str = "cuda"):
    # config = LlamaConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,config = config)
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="cuda",
        # config = config
    )
    model.to(device)
    return model, tokenizer

def load_custom_llama(model_config: LlamaConfig, pretrained_model: AutoModelForCausalLM, device: str = "cuda"):
    # 将预训练模型的权重映射到自定义模型
    hf_sd = pretrained_model.state_dict()
    # 映射嵌入层  # 映射归一化层
    mapping = {
        "model.norm.weight": "norm_weight", 
        "model.embed_tokens.weight": "embed_tokens.weight",
        # "model.embed_tokens.weight": "lm_head.weight",
    }

    # 映射层
    layers = {
        'model.layers.{i}.self_attn.q_proj.weight': 'layers.{i}.attention.wq.weight',
        'model.layers.{i}.self_attn.k_proj.weight': 'layers.{i}.attention.wk.weight',
        'model.layers.{i}.self_attn.v_proj.weight': 'layers.{i}.attention.wv.weight',
        'model.layers.{i}.self_attn.o_proj.weight': 'layers.{i}.attention.wo.weight',
        'model.layers.{i}.mlp.gate_proj.weight': 'layers.{i}.feed_forward.gate_proj.weight',
        'model.layers.{i}.mlp.up_proj.weight': 'layers.{i}.feed_forward.up_proj.weight',
        'model.layers.{i}.mlp.down_proj.weight': 'layers.{i}.feed_forward.down_proj.weight',
        'model.layers.{i}.post_attention_layernorm.weight': 'layers.{i}.ffn_norm_weight',
        'model.layers.{i}.input_layernorm.weight': 'layers.{i}.attention_norm_weight'
    }

    #  根据 Transformer 层数量生成映射
    for i in range(model_config.n_layers):
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
            print(f"custom_key: {custom_key}")
            # 如果某些权重不需要映射，可以选择忽略或处理
            pass  # 忽略未映射的权重
    
    new_sd["lm_head.weight"] = hf_sd["model.embed_tokens.weight"]

    # 打印预训练模型的参数名称
    print("Pretrained model parameters:")
    for name in hf_sd.keys():
        print(name)

    # 打印自定义模型的参数名称
    print("Custom model parameters:")
    for name in new_sd.keys():
        print(name)

    torch.save(new_sd, "/gemini/code/Llama-3.2-1B-Instruct/my_weight/my_llama3.2-1B.pth")
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    torch.set_default_dtype(torch.half)
    my_model = Llama(model_args).to(device)
    my_model.load_state_dict(new_sd, strict=True)
    
    return my_model

def compare_models(original_model, custom_model, tokenizer, input_text: str, device: str = "cuda"):
    # 准备输入
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # 原始模型输出
    with torch.no_grad():
        original_outputs = original_model(**inputs, output_hidden_states=True)
    original_logits = original_outputs.logits

    # 自定义模型输出
    tokens = inputs['input_ids']
    with torch.no_grad():
        custom_outputs = custom_model(tokens, start_pos=0)
    custom_logits = custom_outputs

    # 比较输出
    # print(torch.abs(original_model.state_dict()["model.embed_tokens.weight"] - custom_model.state_dict()["lm_head.weight"]))
    difference = torch.abs(original_logits - custom_logits).mean().item()
    print(f"Average difference between models: {difference}")

    # 可以设置阈值，判断是否一致
    if difference < 1e-2:
        print("Models are consistent.")
    else:
        print("Models are not consistent.")

    print(f"custom_model.hidden_states number: {len(custom_model.hidden_states)}, original_outputs.hidden_states number: {len(original_outputs.hidden_states)} ")
    
    # 比较所有 layer 的隐藏层状态输出
    layer_idxs = range(len(custom_model.hidden_states))
    for index in tqdm(layer_idxs):
        custom_layer_output = custom_model.hidden_states[index]
        original_layer_output = original_outputs.hidden_states[index]

        difference = torch.abs(custom_layer_output - original_layer_output).mean().item()
        print(f"Difference at layer {index}: {difference}")

def load_config_from_json(json_file_path: str) -> LlamaConfig:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config = LlamaConfig(config_dict, max_seq_len = 2048)
    return config

if __name__ == "__main__":
    # 定义模型配置参数
    json_file_path = '/gemini/code/Llama-3.2-1B-Instruct/my_weight/config.json' # JSON 文件的路径
    model_args = load_config_from_json(json_file_path) # 加载配置
    # model_args = LlamaConfig(max_batch_size=2) # 旧版 LlamaConfig 不支持新的 rope 参数

    # 加载原始模型
    original_model_path = "/gemini/code/Llama-3.2-1B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_model, tokenizer = load_original_llama(original_model_path, device)
    
    # 加载自定义模型
    custom_model = load_custom_llama(model_args, original_model, device)

    # 测试文本
    test_text = "Once upon a time in a distant land,"

    # 比较模型输出
    compare_models(original_model, custom_model, tokenizer, test_text, device)
