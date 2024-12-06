import torch
from typing import Optional

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from lite_llama.llava_generate_stream import LlavaGeneratorStream # 导入 GenerateText 类\

checkpoints_dir = "/gemini/code/lite_llama/my_weight/llava-1.5-7b-hf"

def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gpu_num_blocks = None,
    max_gen_len: Optional[int] = 64,
    load_model: bool = True,
    compiled_model: bool = True,
    triton_weight: bool = True
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    generator = LlavaGeneratorStream(
        checkpoints_dir=checkpoints_dir,
        tokenizer_path=checkpoints_dir,
        max_gpu_num_blocks = max_gpu_num_blocks,
        max_seq_len = max_seq_len,
        load_model = load_model,
        compiled_model = compiled_model,
        triton_weight = triton_weight,
        device=device,
    )

    # 调用生成函数，开始流式生成
    prompts = ["USER: <image>\nWhat's the content of the image? ASSISTANT:"]
    image_items = ["https://www.ilankelman.org/stopsigns/australia.jpg"]

    stream = generator.text_completion_stream(
        prompts,
        image_items,
        temperature=temperature,
        top_p=top_p,
        max_gen_len=max_gen_len,
    )

    completion = '' # 初始化生成结果
    # NOTE: 创建了一个 generator 后，可以通过 for 循环来迭代它
    for batch_completions in stream:
        new_text = batch_completions[0]['generation'][len(completion):]
        completion = batch_completions[0]['generation']
        print(new_text, end=' ', flush=True)
    print("\n\n==================================\n")

if __name__ == "__main__":
    main()