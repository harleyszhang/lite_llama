"""速度回归:lite_llama(eager / cuda graph) vs HF transformers,同机对比。

用法: .venv/bin/python scripts/bench_speed.py
"""

import sys
import time

import torch

sys.path.insert(0, "examples")
sys.path.insert(0, ".")

from benchmark import count_tokens, transformers_inference

from lite_llama import SamplingParams, TextGenerator

CKPT = "my_weight/Qwen2.5-0.5B"
HF_NAME = "Qwen/Qwen2.5-0.5B"
MAX_GEN_LEN = 512

PROMPTS = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
]


def run_lite(use_cuda_graph: bool):
    gen = TextGenerator(
        checkpoints_dir=CKPT,
        max_seq_len=2048,
        max_gpu_num_blocks=40960,
        use_cuda_graph=use_cuda_graph,
    )
    params = SamplingParams(temperature=0.7, top_p=0.8, max_gen_len=MAX_GEN_LEN)
    _ = gen.generate(["Hello World"], SamplingParams(temperature=0.7, max_gen_len=5))
    t0 = time.time()
    outs = gen.generate(PROMPTS, params)
    dt = time.time() - t0
    n = count_tokens(outs, gen.tokenizer)
    del gen
    torch.cuda.empty_cache()
    return n, dt


def main():
    n_e, t_e = run_lite(False)
    print(f"lite eager : {n_e} tokens in {t_e:.2f}s -> {n_e / t_e:.1f} tok/s")

    n_g, t_g = run_lite(True)
    print(f"lite graph : {n_g} tokens in {t_g:.2f}s -> {n_g / t_g:.1f} tok/s")

    _, t_hf, n_hf, _, _ = transformers_inference(
        HF_NAME, PROMPTS, temperature=0.7, top_p=0.8, max_gen_len=MAX_GEN_LEN
    )
    print(f"HF         : {n_hf} tokens in {t_hf:.2f}s -> {n_hf / t_hf:.1f} tok/s")


if __name__ == "__main__":
    main()
