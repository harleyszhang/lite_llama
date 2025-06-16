import torch
from typing import Optional
from lite_llama.utils.prompt_templates import get_prompter
from lite_llama.generate_stream import GenerateStreamText  # 导入 GenerateText 类

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

checkpoints_dir = "/home/honggao/lite_llama/my_weight/Qwen3-1.7B"

def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gpu_num_blocks=40960,
    max_gen_len: Optional[int] = 1024,
    compiled_model: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if max_seq_len <= 1024:
        short_prompt = True
    else:
        short_prompt = False
    model_prompter = get_prompter("qwen2", checkpoints_dir, short_prompt)

    # 初始化 LLM 文本生成器
    generator = GenerateStreamText(
        checkpoints_dir=checkpoints_dir,
        tokenizer_path=checkpoints_dir,
        max_gpu_num_blocks=max_gpu_num_blocks,
        max_seq_len=max_seq_len,
        compiled_model=compiled_model,
        device=device,
    )

    while True:
        prompt = input("请输入您的提示（输入 'exit' 退出）：\n")  # 提示用户输入
        # NOTE: strip() 是字符串方法，用于移除字符串开头和结尾的指定字符（默认为空格或换行符）。
        if prompt.strip().lower() == "exit":
            print("程序已退出。")
            break

        print("\n生成结果: ", end="", flush=True)

        model_prompter.insert_prompt(prompt)
        prompts = [model_prompter.model_input]

        # 调用生成函数，开始流式生成
        stream = generator.text_completion_stream(
            prompts,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
        )

        completion = ""  # 初始化生成结果
        # NOTE: 创建了一个 generator 后，可以通过 for 循环来迭代它
        for batch_completions in stream:
            new_text = batch_completions[0]["generation"][len(completion) :]
            completion = batch_completions[0]["generation"]
            print(new_text, end="", flush=True)
        print("\n\n==================================\n")


if __name__ == "__main__":
    main()
