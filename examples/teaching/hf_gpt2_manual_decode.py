"""教学示例:用 HuggingFace 的 ``model.forward`` 手写逐 token 解码循环。

展示 lite_llama 引擎内部替你做掉的那些步骤——维护 ``past_key_values``、
每步只喂新 token、top-p 过滤后采样。与 ``lite_llama.engine`` 的实现对照阅读,
可以看出后者省掉了哪些主机侧开销。

用法::

    python examples/teaching/hf_gpt2_manual_decode.py --model-dir /path/to/gpt2
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_text(
    model, tokenizer, prompt, max_length=50, temperature=1.0, top_p=0.9, device="cuda"
):
    """
    使用 model.forward 实现逐步生成文本，并正确设置 attention_mask。

    Args:
        model: 已加载的因果语言模型。
        tokenizer: 对应的 tokenizer。
        prompt: 初始输入文本。
        max_length: 生成的最大 token 数量。
        temperature: 采样时的温度参数。
        top_p: 采样时的 top-p 参数。
        device: 设备类型。

    Returns:
        生成的文本字符串。
    """
    # 编码输入 prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # [1, S]

    # 初始化生成的 Token 列表
    generated_ids = input_ids

    # 初始化 past_key_values 为 None
    past_key_values = None

    for _ in range(max_length):
        # 调用模型的 forward 方法
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)

        # 获取 logits，并仅关注最后一个 token 的 logits
        logits = outputs.logits  # [1, 1, V]
        next_token_logits = logits[:, -1, :] / temperature  # [1, V]

        # 应用 top-p 过滤
        sorted_logits, sorted_indices = torch.sort(next_token_logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # 创建 mask
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the mask to include the first token exceeding p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # 应用 mask
        sorted_logits[sorted_indices_to_remove] = -float("Inf")
        # 应用 softmax
        probs = torch.softmax(sorted_logits, dim=-1)

        # 采样下一个 token
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

        # 反向排序索引以获取原始 token ID
        next_token = sorted_indices.gather(-1, next_token)

        # 将生成的 token 添加到生成的 Token 列表中
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # 更新 input_ids 为新生成的 token
        input_ids = next_token

        # 更新 past_key_values
        past_key_values = outputs.past_key_values

    # 解码生成的 token
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("LITE_LLAMA_MODEL_DIR", "gpt2"),
        help="本地权重目录或 HuggingFace 模型名(默认 'gpt2',直接从 hub 拉取)",
    )
    parser.add_argument("--prompt", default="Once upon a time in a distant land,")
    parser.add_argument("--max-length", type=int, default=100)
    args = parser.parse_args()

    model_name = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 将模型移动到 GPU（如果可用）并设置为评估模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 生成文本
    generated = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_length=args.max_length,
        temperature=1.0,
        top_p=0.9,
        device=device,
    )
    print(generated)
