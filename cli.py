import torch
from typing import List, Optional
from lite_llama.generate import GenerateText # 导入 GenerateText 类

def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,  # 每次处理一个 Prompt
    max_gen_len: Optional[int] = 64,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoints_dir = '/gemini/code/Llama-3.2-1B-Instruct/my_weight/'
    tokenizer_path = '/gemini/code/Llama-3.2-1B-Instruct/'

    generator = GenerateText(
        checkpoints_dir=checkpoints_dir,
        tokenizer_path=tokenizer_path,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        load_model=True,
        compiled_model=True,
        triton_weight=True,
        device=device,
    )

    while True:
        # 提示用户输入
        prompt = input("请输入您的提示（输入 'exit' 退出）：\n")
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

        # 初始化生成结果
        completion = ''
        for batch_completions in stream:
            new_text = batch_completions[0]['generation'][len(completion):]
            completion = batch_completions[0]['generation']
            print(new_text, end='', flush=True)
        print("\n\n==================================\n")

if __name__ == "__main__":
    main()