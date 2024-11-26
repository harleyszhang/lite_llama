import torch
from typing import Optional
from lite_llama.generate_stream import GenerateText # 导入 GenerateText 类

# checkpoints_dir = '/gemini/code/Llama-3.2-1B-Instruct/my_weight/' # 改成自己的存放模型路径
checkpoints_dir = "/gemini/code/Qwen2.5-3B-Instruct/"

def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 1,  # 每次处理一个 Prompt
    max_gen_len: Optional[int] = 512,
    load_model: bool = True,
    compiled_model: bool = True,
    triton_weight: bool = True
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    generator = GenerateText(
        checkpoints_dir=checkpoints_dir,
        tokenizer_path=checkpoints_dir,
        max_batch_size = max_batch_size,
        max_seq_len = max_seq_len,
        load_model = load_model,
        compiled_model = compiled_model,
        triton_weight = triton_weight,
        device=device,
    )

    while True:
        prompt = input("请输入您的提示（输入 'exit' 退出）：\n") # 提示用户输入
        # NOTE: strip() 是字符串方法，用于移除字符串开头和结尾的指定字符（默认为空格或换行符）。
        if prompt.strip().lower() == 'exit':
            print("程序已退出。")
            break

        print("\n生成结果: ", end='', flush=True)

        # 调用生成函数，开始流式生成
        stream = generator.text_completion_stream(
            [prompt],
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
        )

        completion = '' # 初始化生成结果
        # NOTE: 创建了一个 generator 后，可以通过 for 循环来迭代它
        for batch_completions in stream:
            new_text = batch_completions[0]['generation'][len(completion):]
            completion = batch_completions[0]['generation']
            print(new_text, end='', flush=True)
        print("\n\n==================================\n")

if __name__ == "__main__":
    main()