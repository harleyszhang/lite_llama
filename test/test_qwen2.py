from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
import torch
from tqdm.auto import tqdm
import json, sys, os
from pathlib import Path

# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lite_llama.models.qwen2 import Qwen2Model, Qwen2Config
from lite_llama.executor.model_executor import ModelExecutor

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def load_config_from_json(json_file_path: str, device: str="cuda") -> Qwen2Config:
    with open(json_file_path, "r") as f:
        config_dict = json.load(f)
    
    config = Qwen2Config(config_dict, max_seq_len = 2048, device=device)
    return config

def load_original_llama(model_name_or_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.to(device)
    return model, tokenizer

def load_custom_llam(model_name_or_path: str, model_config: Qwen2Config, device: str = "cuda"):
    checkpoints = sorted(Path(model_name_or_path).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {model_name_or_path}"
    ckpt_path = checkpoints[0]
    state_dict = torch.load(ckpt_path, map_location="cuda")

    # 根据设备选择合适的 dtype
    torch.set_default_dtype(torch.half)

    model = Qwen2Model(model_config).to(device)
    model.load_state_dict(state_dict, strict=True)

    return model

def load_and_convert_to_custom_qwen2(model_config: Qwen2Config, pretrained_model: AutoModelForCausalLM, device: str = "cuda"):
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
            print(f"custom_key: {custom_key}")
            # 如果某些权重不需要映射，可以选择忽略或处理
            pass  # 忽略未映射的权重
    
    new_sd["lm_head_weight"] = hf_sd["model.embed_tokens.weight"]

    # 打印预训练模型的参数名称
    print("Pretrained model parameters:")
    for name in hf_sd.keys():
        print(name)

    # 打印自定义模型的参数名称
    print("Custom model parameters:")
    for name in new_sd.keys():
        print(name)

    # 保存转换好的自定义权重
    torch.save(new_sd, "/gemini/code/Qwen2.5-3B-Instruct/my_qwen2.5-3B.pth")

    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    torch.set_default_dtype(torch.half)
    my_model = Qwen2Model(model_config).to(device)
    my_model.load_state_dict(new_sd, strict=True)
    
    return my_model

def decode_stage_compare(original_model, model_executor, tokenizer, input_text: str, device: str = "cuda"):
    """
    在解码阶段逐步比较原始模型和自定义模型的输出。

    Args:
        original_model (Any): 原始模型。
        custom_model (Any): 自定义模型。
        tokenizer (Any): tokenizer。
        input_text (str): 输入文本。
        device (str, optional): 设备类型，默认是 "cuda"。
    """
    # 准备输入
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)

    # 设置生成参数
    max_new_tokens = 10
    original_model.eval()
    custom_model.eval()

    # 初始化生成的 tokens
    original_generated = input_ids
    custom_generated = input_ids

    for step in tqdm(range(max_new_tokens), desc="Decoding steps"):
        # 原始模型生成下一个 token
        with torch.no_grad():
            original_outputs = original_model(original_generated, 
                                              attention_mask=attention_mask, 
                                              output_hidden_states=True,
                                              return_dict = True,
                                              use_cache = True)
            original_logits = original_outputs.logits[:, -1, :]  # 获取最后一个时间步的 logits
            original_next_token = torch.argmax(original_logits, dim=-1, keepdim=True)

        # 自定义模型生成下一个 token
        with torch.no_grad():
            custom_outputs_logits, select_index = model_executor.forward(input_ids, start_pos=original_generated.shape[1] - 1) # 模型执行器的前向推理
            # custom_outputs_logits = custom_model(custom_generated, start_pos=original_generated.shape[1]-1,)
            probs = torch.softmax(original_logits[:, -1] / 0.6, dim=-1) # temperature = 0.6
            custom_next_token = sample_top_p(probs, p = 0.9)

        # 比较所有 layer 的隐藏层状态输出
        # print("original_outputs.hidden_states length is", len(original_outputs.hidden_states)) # 17
        # print("model_executor.model.hidden_states length is", len(model_executor.model.hidden_states))         # 17

        layer_idxs = range(len(model_executor.model.hidden_states))

        print(f"============== Step {step+1}: Layer Compares: ====================")
        for index in tqdm(layer_idxs):
            custom_layer_output = model_executor.model.hidden_states[index]
            original_layer_output = original_outputs.hidden_states[index]

            difference = torch.abs(custom_layer_output - original_layer_output).mean().item()
            print(f"Difference at layer {index}: {difference}")

        # # 比较 logits
        logits_diff = torch.abs(original_logits - custom_outputs_logits).mean().item()
        print(f"=========== Step {step+1}: Logits difference is: {logits_diff} ================")

        # 更新 attention mask if necessary
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)], dim=-1)

    print("Decode stage comparison completed.")

def compare_models(original_model, model_executor, tokenizer, input_text: str, device: str = "cuda"):
    # custom_model = model_executor
    print("\n############################ [Starting Prefill stage comparison] #################################")
    # 准备输入
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # 原始模型输出
    with torch.no_grad():
        original_outputs = original_model(**inputs, output_hidden_states=True)
    original_logits = original_outputs.logits

    # 自定义模型输出
    tokens = inputs['input_ids']
    with torch.no_grad():
        custom_outputs = model_executor(tokens, start_pos=0)
    custom_logits = custom_outputs

    # 比较 logits 输出
    difference = torch.abs(original_logits - custom_logits).mean().item()
    print(f"Average difference between models: {difference}")

    # 可以设置阈值，判断是否一致
    if difference < 1e-2:
        print("Models are consistent.")
    else:
        print("Models are not consistent.")

    print(f"model_executor.model.hidden_states number: {len(model_executor.model.hidden_states)}, original_outputs.hidden_states number: {len(original_outputs.hidden_states)} ")
    
    # 比较所有 layer 的隐藏层状态输出
    layer_idxs = range(len(model_executor.model.hidden_states))
    for index in tqdm(layer_idxs):
        custom_layer_output = model_executor.model.hidden_states[index]
        original_layer_output = original_outputs.hidden_states[index]

        difference = torch.abs(custom_layer_output - original_layer_output).mean().item()
        print(f"Difference at layer {index}: {difference}")

    # 解码阶段比较
    print("\n############################ [Starting Decode stage comparison] #################################")
    decode_stage_compare(original_model, model_executor, tokenizer, input_text, device)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 定义模型配置参数
    original_model_path = "/gemini/pretrain/Qwen2.5-3B"
    my_model_path = "/gemini/code/Qwen2.5-3B-Instruct/"
    json_file_path = os.path.join(original_model_path, 'config.json') # JSON 文件的路径
    model_config = load_config_from_json(json_file_path, device) # 加载配置

    # 加载原始模型
    original_model, tokenizer = load_original_llama(original_model_path, device)
    custom_model = Qwen2Model(model_config)
    
    for name, param in custom_model.named_parameters():
        print(name, param.shape)

    # 加载自定义模型
    # custom_model = load_and_convert_to_custom_qwen2(model_config, original_model, device)
    model_executor = ModelExecutor.build(
            checkpoints_dir = my_model_path,
            tokenizer_path = my_model_path,
            load_model = True,
            max_batch_size = 64,
            max_seq_len = 2048,
            triton_weight = True,
            device = device
        )
    # 测试文本
    test_text = "Once upon a time in a distant land,"

    # 比较模型输出
    compare_models(original_model, model_executor, tokenizer, test_text, device)
