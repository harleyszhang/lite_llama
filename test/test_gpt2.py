from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# huggingface-cli download --resume-download --local-dir-use-symlinks False gpt2 --local-dir gpt2

# 加载模型和分词器
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 模型设置为评估模式
model.eval()

# 初始提示文本
prompt = "The meaning of life is"

# 转换为输入 ID
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]  # 初始输入 ID

# 设置解码参数
max_gen_len = 50  # 最大生成长度
temperature = 0.7  # 控制生成随机性
top_p = 0.9  # 使用 nucleus sampling
eos_token_id = tokenizer.eos_token_id  # 终止标记

# 初始化解码结果
generated_ids = input_ids.clone()

# 多轮解码
for step in range(max_gen_len):
    # 前向传播获取 logits
    outputs = model.forward(input_ids=generated_ids)
    logits = outputs.logits  # 形状: [batch_size, seq_len, vocab_size]

    # 取最后一个 token 的分布
    next_token_logits = logits[:, -1, :]

    # 应用 softmax 获取概率分布
    probs = torch.softmax(next_token_logits / temperature, dim=-1)

    # 使用 nucleus sampling (top-p) 选择下一个 token
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_mask = cumulative_probs > top_p
    sorted_probs[top_p_mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(sorted_probs, num_samples=1)

    # 将新 token 拼接到生成序列
    generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    # 检查是否生成了终止标记
    if next_token.item() == eos_token_id:
        break

# 解码生成的文本
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("生成的文本:", output_text)