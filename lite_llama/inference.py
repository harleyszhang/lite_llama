from typing import Optional
import torch

import time

from .utils.prompt_templates import get_prompter
from .generate import GenerateText


class Inference(object):
    def __init__(
        self,
        temperature: float,
        top_p: float,
        max_seq_len: int,
        max_gen_len: Optional[int],
        lite_llama_ckpt_dir: str,
        device: str = "cuda",
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.lite_llama_ckpt_dir = lite_llama_ckpt_dir
        self.device = device

    def load_generator(self, max_gpu_num_blocks=None) -> GenerateText:
        """
        Initializes the lite-llama generator
        """
        generator = GenerateText(
            checkpoints_dir=self.lite_llama_ckpt_dir,
            tokenizer_path=self.lite_llama_ckpt_dir,
            max_seq_len=self.max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            compiled_model=True,
            device=self.device,
        )
        return generator

    def count_tokens(self, texts: list[str], tokenizer) -> int:
        # Optimized segmentation statistics
        total_tokens = 0
        for t in texts:
            ids = tokenizer(t, add_special_tokens=False)["input_ids"]
            total_tokens += len(ids)
        return total_tokens

    def inference(self, generator: GenerateText, prompts: list[str]):
        """
        Inference is performed using lite-llama's GenerateText instance and returns 
        the result with the time taken and the number of tokens output
        """

        # Warm-up step: use a short dummy input to allow the model to 
        # perform a simple inference to load caches/compile optimizations, etc.
        warm_up_prompt = ["Hello World"] * 4
        _ = generator.text_completion(
            warm_up_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_gen_len=5,
        )

        start_time = time.time()
        results = generator.text_completion(
            prompts,
            temperature=self.temperature,
            top_p=self.top_p,
            max_gen_len=self.max_gen_len,
        )
        end_time = time.time()

        total_tokens = self.count_tokens(results, generator.tokenizer)

        return results, end_time - start_time, total_tokens

    def process(self, prompts):
        if "qwen2" in self.lite_llama_ckpt_dir.lower():
            model_type = "qwen2"
        elif "llama" in self.lite_llama_ckpt_dir.lower():
            model_type = "llama"
        elif "llava" in self.lite_llama_ckpt_dir.lower():
            model_type = "llava"
        else:
            print("Error! Unsupported model type!")

        model_prompter = get_prompter(model_type, self.lite_llama_ckpt_dir)
        update_prompts = []
        for prompt in prompts:
            model_prompter.insert_prompt(prompt)
            update_prompts.append(model_prompter.model_input)

        # 1. lite-llama inference
        lite_llama_generator = self.load_generator(max_gpu_num_blocks=40960)
        lite_llama_results, lite_llama_time, lite_llama_tokens = self.inference(
            lite_llama_generator, update_prompts
        )
        del lite_llama_generator
        torch.cuda.empty_cache()  # Release the memory used by lite_llama_generator after use.

        return lite_llama_results
