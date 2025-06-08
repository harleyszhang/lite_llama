from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import sys, os, time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from lite_llama.generate import GenerateText
from lite_llama.utils.prompt_templates import get_prompter

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")


def load_lite_llama_generator(
    checkpoints_dir: str,
    max_seq_len: int,
    max_gpu_num_blocks=None,
    device: str = "cuda",
) -> GenerateText:
    """
    初始化 lite-llama 的生成器
    """
    generator = GenerateText(
        checkpoints_dir=checkpoints_dir,
        tokenizer_path=checkpoints_dir,
        max_seq_len=max_seq_len,
        max_gpu_num_blocks=max_gpu_num_blocks,
        compiled_model=True,
        device=device,
    )
    return generator


def count_tokens(texts: list[str], tokenizer) -> int:
    # 优化后的分词统计
    total_tokens = 0
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        total_tokens += len(ids)
    return total_tokens


def lite_llama_inference(
    generator: GenerateText,
    prompts: list[str],
    temperature: float,
    top_p: float,
    max_gen_len: Optional[int],
    device: str = "cuda",
):
    """
    使用 lite-llama 的 GenerateText 实例执行推理，并返回结果与耗时、输出 tokens 数量
    """

    # 预热步骤：使用一个简短的假输入，让模型进行一次简单推理，以加载缓存/编译优化等
    warm_up_prompt = ["Hello World"] * 4
    _ = generator.text_completion(
        warm_up_prompt,
        temperature=temperature,
        top_p=top_p,
        max_gen_len=5,
    )

    start_time = time.time()
    results = generator.text_completion(
        prompts,
        temperature=temperature,
        top_p=top_p,
        max_gen_len=max_gen_len,
    )
    end_time = time.time()

    total_tokens = count_tokens(results, generator.tokenizer)

    return results, end_time - start_time, total_tokens


def transformers_inference(
    hf_model_name: str,
    prompts: list[str],
    temperature: float,
    top_p: float,
    max_gen_len: int,
    device: str = "cuda",
):
    """
    使用 Transformers 官方库对一组 prompts 进行批量推理, 返回结果与耗时、输出 tokens 数量。
    """
    # from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    # 确保分词器有 eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # 预热步骤：让模型先对一个非常简单的 prompt 做一次推理
    warm_up_prompt = ["Hello World"]
    warm_up_inputs = tokenizer(
        warm_up_prompt, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    with torch.no_grad():
        _ = model.generate(
            **warm_up_inputs,
            max_new_tokens=10,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    start_time = time.time()
    model_inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    input_ids = model_inputs.input_ids
    generation_kwargs = {
        "max_new_tokens": max_gen_len,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": True,
        "eos_token_id": None,  # 避免过早终止
    }

    # 一次性进行批量推理
    with torch.no_grad():
        outputs = model.generate(**model_inputs, **generation_kwargs)
        generated_ids = outputs[:, input_ids.size(-1) :]
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    end_time = time.time()
    results = [{"generation": text} for text in generated_texts]
    texts = [res["generation"] for res in results]
    total_tokens = count_tokens(texts, tokenizer)
    total_time = end_time - start_time
    prompts_tokens = input_ids.numel()
    per_token_latency = total_time / total_tokens if total_tokens > 0 else float("inf")

    return results, total_time, total_tokens, prompts_tokens, per_token_latency


def compare_inference_speed(
    prompts: list[str],
    temperature: float,
    top_p: float,
    max_seq_len: int,
    max_gen_len: Optional[int],
    lite_llama_ckpt_dir: str,
    hf_model_name: str,
    print_result=False,
    device: str = "cuda",
):
    """
    对比 lite-llama 与 transformers 官方模型在相同 prompts 下的推理速度和吞吐量。
    """
    if "qwen2" in lite_llama_ckpt_dir.lower():
        model_type = "qwen2"
    elif "llama" in lite_llama_ckpt_dir.lower():
        model_type = "llama"
    elif "llava" in lite_llama_ckpt_dir.lower():
        model_type = "llava"
    else:
        print("Error! Unsupported model type!")

    model_prompter = get_prompter(model_type, lite_llama_ckpt_dir)
    update_prompts = []
    for prompt in prompts:
        model_prompter.insert_prompt(prompt)
        update_prompts.append(model_prompter.model_input)

    # 1. lite-llama inference
    lite_llama_generator = load_lite_llama_generator(
        lite_llama_ckpt_dir, max_seq_len, max_gpu_num_blocks=40960, device=device
    )
    lite_llama_results, lite_llama_time, lite_llama_tokens = lite_llama_inference(
        lite_llama_generator,
        update_prompts,
        temperature,
        top_p,
        max_gen_len,
        device=device,
    )
    del lite_llama_generator
    torch.cuda.empty_cache()  # 使用完成后释放 lite_llama_generator 占用的显存

    # 2. transformers inference
    hf_results, hf_time, hf_tokens, prompts_tokens, hf_pt_latency = (
        transformers_inference(
            hf_model_name,
            update_prompts,
            temperature,
            top_p,
            max_gen_len,
            device=device,
        )
    )

    lite_llama_pt_latency = lite_llama_time / (lite_llama_tokens)

    # 打印时间对比
    print("lite_llama inference time: {:.4f} s".format(lite_llama_time))
    print("Transformers inference time: {:.4f} s".format(hf_time))

    print("lite_llama inference output tokens number: {:2d}".format(lite_llama_tokens))
    print("Transformers inference output tokens number: {:2d}".format(hf_tokens))

    # 吞吐量计算
    lite_llama_throughput = (
        (lite_llama_tokens) / lite_llama_time if lite_llama_time > 0 else float("inf")
    )
    print(f"lite_llama throughput: {lite_llama_throughput:.2f} tokens/s")

    hf_throughput = hf_tokens / hf_time if hf_time > 0 else float("inf")
    print(f"Transformers throughput: {hf_throughput:.2f} tokens/s")

    # 打印 per token latency
    print(f"lite_llama per token latency: {lite_llama_pt_latency * 1000:.6f} ms/token")
    print(f"Transformers per token latency: {hf_pt_latency * 1000:.6f} ms/token")

    # 打印部分推理结果对比
    if print_result:
        for i, (prompt, litellama_res, hf_res) in enumerate(
            zip(prompts, lite_llama_results, hf_results)
        ):
            # print(f"\n[Prompt {i}]:\n{prompt}")
            if i == 0:  # 省略部分打印
                print("\n[lite_llama]: {}".format(litellama_res))
                print("\n[Transformers]: {}".format(hf_res["generation"]))
                print("\n" + "=" * 40 + "\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts: list[str] = [
        "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
        "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
        "Can you introduce the History of the American Civil War. ",
        "who is the first president of the United States and what's his life story?",
        "How to learn c++, give me some code example.",
        "How to learn python, give me some code examples.",
        "How to learn llm, please introduce transformer architecture ",
        "How to learn cnn, please introduce resnet architecture and give code ",
        "How to learn cuda programming, give me some code example.",
        "How to learn rust, give me some code examples.",
        "How to learn java, give me some code example.",
        "How to learn linux c, give me some code examples.",
        "A Complete Introduction to the History of the American Civil War",
        "Python is a good programming language, how tolearn it?",
        "Please introduce llama model architecture and give implement cuda code.",
        "Please introduce Qwen2.5 model structure and give cuda implement code.",
    ]

    # prompts: list[str] = [
    #     "How to learn cnn, please introduce resnet architecture and give code ",
    #     "How to learn cuda programming, give me some code example.",
    # ]

    # prompts: list[str] = [
    #     "How to learn cnn, please introduce resnet architecture and give code.",
    #     "How to learn cuda programming, give me some code example.",
    #     "How to learn rust, give me some code examples.",
    #     "How to learn c++, give me some code examples.",
    # ]

    # prompts: list[str] = [
    #     "I believe the meaning of life is to find happiness in the simple things. This is a very subjective and personal perspective, and it may vary from person to person. However, I believe that the simple things can bring a sense of joy and fulfillment to our lives.",
    #     "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    #     "A Complete Introduction to the History of the American Civil War",
    #     "Roosevelt was the first president of the United States, he has a lot of information on the early history of the United States. He was born in 1883,",
    #     "How to learn c++, give me some code example.",
    #     "How to learn python, give me some code examples.",
    #     "How to learn llm, please introduce transformer architecture ",
    #     "How to learn cnn, please introduce resnet architecture and give code ",
    # ]

    # prompts: list[str] = [
    #     "I believe the meaning of life is to find happiness in the simple things. This is a very subjective and personal perspective, and it may vary from person ",
    #     "Simply put, the theory of relativity states that 3D space is not fixed, but is relative to the observer's frame of reference. Time is also relative, and it appears to ",
    #     """A brief message congratulating the team on the launch:

    #     Hi everyone,

    #     I just heard about the launch of the new product and I wanted to take a moment to express my """,
    #     "Roosevelt was the 26th president of the United States, he has a lot of information on the early history of the ,",
    # ]

    # prompts: list[str] = [
    #     "I believe the meaning of life is",
    #     "Simply put, the theory of relativity states that 3D space",
    #     """A brief message congratulating the team on the launch:

    #     Hi everyone,

    #     I just heard""",
    #     "Roosevelt was the 26th president of the United States,",
    # ]

    hf_model_name = "/gemini/code/my_weight/Llama-3.2-1B-Instruct-hf"
    custom_checkpoints_dir = (
        "/gemini/code/my_weight/Llama-3.2-1B-Instruct"  # 根据实际情况修改
    )
    # hf_model_name = "/gemini/code/my_weight/Qwen-hf/Qwen2.5-1.5B-Instruct"
    # custom_checkpoints_dir = "/gemini/code/my_weight/Qwen2.5-1.5B-Instruct"
    compare_inference_speed(
        prompts=prompts,
        temperature=0.7,
        top_p=0.8,
        max_seq_len=2048,
        max_gen_len=1900,
        lite_llama_ckpt_dir=custom_checkpoints_dir,
        hf_model_name=hf_model_name,
        print_result=True,
        device=device,
    )


if __name__ == "__main__":
    main()
