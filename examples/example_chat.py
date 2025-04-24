from typing import List, Optional
import torch

import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from lite_llama.generate import GenerateText
from lite_llama.generate_stream import GenerateStreamText
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

checkpoints_dir = "/gemini/code/lite_llama/my_weight/Qwen2.5-3B" # 改成自己的存放模型路径

def cli_generate_stream(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_gpu_num_blocks = None,
    max_gen_len: Optional[int] = 128,
):
    """
    程序的入口点，用于使用预训练模型生成文本。

    参数：
        temperature (float): 控制生成随机性的温度值。
        top_p (float): 控制生成多样性的 top-p 采样参数。
        max_seq_len (int): 输入提示的最大序列长度。
        max_batch_size (int): 生成序列的最大批量大小。
        max_gen_len (int): 生成序列的最大长度。

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generator = GenerateStreamText(
        checkpoints_dir=checkpoints_dir,
        tokenizer_path=checkpoints_dir,
        max_gpu_num_blocks = max_gpu_num_blocks, 
        max_seq_len=max_seq_len,
        load_model=True,
        compiled_model=True,
        triton_weight=True,
        device=device,
    )

    prompts: List[str] = [
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        "Roosevelt was the first president of the United States, he has",

        "Here are some tips and resources to help you get started:"
    ]

    for idx, prompt in enumerate(prompts):
        print(f"Prompt {idx}: {prompt}")
        print("Generated output:", end='', flush=True)

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

def cli_generate(
	temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_gen_len: Optional[int] = 64,
):
	"""
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	generator = GenerateText(
		checkpoints_dir = checkpoints_dir,
		tokenizer_path = checkpoints_dir,
		max_seq_len = max_seq_len,
		load_model = True,
		compiled_model = True,
		triton_weight = True,
		device = device,
	)

	prompts: List[str] = [
		# For these prompts, the expected answer is the natural continuation of the prompt
		"I believe the meaning of life is",
		"Simply put, the theory of relativity states that ",
		"""A brief message congratulating the team on the launch:

		Hi everyone,
		
		I just """,
		"Roosevelt was the first president of the United States, he has",
	]

	results = generator.text_completion(
		prompts,
		temperature=temperature,
		top_p=top_p,
		max_gen_len=max_gen_len,
	)

	for prompt, result in zip(prompts, results):
		print(prompt)
		print(f"> {result['generation']}")
		print("\n==================================\n")

def main(stream_flag = False):
    cli_generate_stream() if stream_flag else cli_generate()

if __name__ == "__main__":
    main()