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
    # 优化后的分词统计
    total_tokens = 0
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
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
    texts = [res['generation'] for res in results]
    # total_tokens = sum(len(output) for output in texts)
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
    使用 Transformers 官方库对一组 prompts 进行批量推理, 返回结果与耗时、输出 tokens 数量。
    """
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    # 加载模型，并移动到 GPU（如果可用）
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(hf_model_name, trust_remote_code=True)
        
    model = load_checkpoint_and_dispatch(model, hf_model_name, device_map="auto", dtype=torch.float16 )

    model.eval()

    generation_kwargs = {
        "max_new_tokens": max_gen_len,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": True  # 此处可根据需求改为 True
    }

    # 对 prompts 批量编码
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device).to(device)
    input_ids = inputs.input_ids

    start_time = time.time()
    # 一次性进行批量推理
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
        # 根据 batch 输出的数量解码所有结果, outputs 的形状一般为 [batch_size, seq_length]. 输出序列批量解码，并且 skip_special_tokens=True 跳过特殊token
        generated_ids = outputs[:, input_ids.size(-1):] 
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    end_time = time.time()
    # 将结果打包为字典列表，与输入 prompts 对齐, 将返回的结果中的输入提示词去除
    results = [{"generation": text} for text in generated_texts]
    texts = [res['generation'] for res in results]
    total_tokens = count_tokens(texts, tokenizer) # 统计 tokens 数量

    return results, (end_time - start_time), total_tokens

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
    print("lite_llama inference time: {:.4f} s".format(lite_llama_time))
    print("Transformers inference time: {:.4f} s".format(hf_time))

    # 计算吞吐量（tokens/s）
    lite_llama_throughput = lite_llama_tokens / lite_llama_time if lite_llama_time > 0 else float('inf')
    hf_throughput = hf_tokens / hf_time if hf_time > 0 else float('inf')

    print(f"lite_llama throughput: {lite_llama_throughput:.2f} tokens/s")
    print(f"Transformers throughput: {hf_throughput:.2f} tokens/s")

    # 打印一些示例结果对比
    for i, (prompt, litellama_res, hf_res) in enumerate(zip(prompts, lite_llama_results, hf_results)):
        print(f"Prompt {i}:\n{prompt}")
        print("lite_llama: {}".format(litellama_res['generation']))
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