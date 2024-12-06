from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from lite_llama.generate import GenerateText

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

checkpoints_dir = "/gemini/code/lite_llama/my_weight/Qwen2.5-3B"  # 根据实际情况修改

def load_lite_llama_generator(
    checkpoints_dir: str,
    max_seq_len: int,
    device: str = "cuda"
) -> GenerateText:
    """
    初始化 lite-llama 的生成器
    """
    generator = GenerateText(
        checkpoints_dir=checkpoints_dir,
        tokenizer_path=checkpoints_dir,
        max_seq_len=max_seq_len,
        load_model=True,
        compiled_model=True,
        triton_weight=True,
        device=device,
    )
    return generator

def count_tokens(texts: List[str], tokenizer) -> int:
    """
    使用提供的 tokenizer 对列表中所有文本进行分词，并统计 tokens 总数。
    """
    total_tokens = 0
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        total_tokens += len(ids)
    return total_tokens

def lite_llama_inference(
    generator: GenerateText,
    prompts: List[str],
    temperature: float,
    top_p: float,
    max_gen_len: Optional[int],
    device: str = "cuda"
):
    """
    使用 lite-llama 的 GenerateText 实例执行推理，并返回结果与耗时、输出 tokens 数量
    """
    start_time = time.time()
    results = generator.text_completion(
        prompts,
        temperature=temperature,
        top_p=top_p,
        max_gen_len=max_gen_len,
        device=device,
    )
    end_time = time.time()

    # 使用 generator 内部的 tokenizer 来统计 tokens 数量
    # 假设 generator 有 tokenizer 属性或者可访问，如果没有则需自行创建 tokenizer
    texts = [r['generation'] for r in results]
    total_tokens = count_tokens(texts, generator.tokenizer)

    return results, end_time - start_time, total_tokens

def transformers_inference(
    hf_model_name: str,
    prompts: List[str],
    temperature: float,
    top_p: float,
    max_gen_len: int,
    device: str = "cuda"
):
    """
    使用 Transformers 官方库加载模型并执行推理，返回结果与耗时、输出 tokens 数量。
    """
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    generation_kwargs = {
        "max_new_tokens": max_gen_len,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": True
    }

    start_time = time.time()
    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids,
            **generation_kwargs
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"generation": text})
    end_time = time.time()

    # 统计 transformers 输出 tokens 数量
    texts = [r['generation'] for r in results]
    total_tokens = count_tokens(texts, tokenizer)

    return results, end_time - start_time, total_tokens

def compare_inference_speed(
    prompts: List[str],
    temperature: float,
    top_p: float,
    max_seq_len: int,
    max_gen_len: Optional[int],
    lite_llama_ckpt_dir: str,
    hf_model_name: str,
    device: str = "cuda"
):
    """
    对比 lite-llama 与 transformers 官方模型在相同 prompts 下的推理速度和吞吐量。
    """
    # 1. lite-llama inference
    lite_llama_generator = load_lite_llama_generator(lite_llama_ckpt_dir, max_seq_len, device=device)
    lite_llama_results, lite_llama_time, lite_llama_tokens = lite_llama_inference(
        lite_llama_generator, prompts, temperature, top_p, max_gen_len, device=device
    )

    # 2. transformers inference
    hf_results, hf_time, hf_tokens = transformers_inference(
        hf_model_name, prompts, temperature, top_p, max_gen_len if max_gen_len else 64, device=device
    )

    # 打印时间对比
    print("Lite-LLaMA inference time: {:.4f} s".format(lite_llama_time))
    print("Transformers inference time: {:.4f} s".format(hf_time))

    # 计算吞吐量（tokens/s）
    lite_llama_throughput = lite_llama_tokens / lite_llama_time if lite_llama_time > 0 else float('inf')
    hf_throughput = hf_tokens / hf_time if hf_time > 0 else float('inf')

    print(f"Lite-LLaMA throughput: {lite_llama_throughput:.2f} tokens/s")
    print(f"Transformers throughput: {hf_throughput:.2f} tokens/s")

    # 打印一些示例结果对比
    for i, (prompt, ll_res, hf_res) in enumerate(zip(prompts, lite_llama_results, hf_results)):
        print(f"Prompt {i}:\n{prompt}")
        print("Lite-LLaMA: {}".format(ll_res['generation']))
        print("Transformers: {}".format(hf_res['generation']))
        print("\n" + "="*40 + "\n")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 准备测试用的prompts
    prompts: List[str] = [
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        "Roosevelt was the first president of the United States, he has",
    ]
    
    # 假设 transformers 模型名称为 "Qwen/Qwen-7B" (需根据实际情况更改)
    hf_model_name = "/gemini/code/llm_weights/Qwen/Qwen2.5-3B-Instruct"

    compare_inference_speed(
        prompts=prompts,
        temperature=0.6,
        top_p=0.9,
        max_seq_len=512,
        max_gen_len=64,
        lite_llama_ckpt_dir=checkpoints_dir,
        hf_model_name=hf_model_name,
        device=device
    )

if __name__ == "__main__":
    main()