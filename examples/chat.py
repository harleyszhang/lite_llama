"""Batch text generation with :class:`~rapid_llm.engine.generator.TextGenerator`.

A minimal CLI demo: build a :class:`TextGenerator` over the checkpoint
named at the top, then generate for a fixed prompt batch (streaming or
blocking) and print the texts.

Usage:
    python examples/chat.py   # set ``checkpoints_dir`` first
"""

from __future__ import annotations

import torch

from rapid_llm.engine import SamplingParams, TextGenerator

checkpoints_dir = "my_weight/Qwen2.5-0.5B"  # 改成自己的存放模型路径

PROMPTS: list[str] = [
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    """A brief message congratulating the team on the launch:

    Hi everyone,

    I just """,
    "Roosevelt was the first president of the United States, he has",
    "Here are some tips and resources to help you get started:",
]


def cli_generate_stream(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 128,
) -> None:
    """Stream each prompt's completion token group by token group."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = TextGenerator(
        checkpoints_dir=checkpoints_dir,
        max_seq_len=max_seq_len,
        device=device,
    )
    params = SamplingParams(temperature=temperature, top_p=top_p, max_gen_len=max_gen_len)

    for idx, prompt in enumerate(PROMPTS):
        print(f"Prompt {idx}: {prompt}")
        print("Generated output:", end="", flush=True)
        for step in generator.stream([prompt], params):
            print(step[0], end="", flush=True)
        print("\n\n==================================\n")


def cli_generate(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 128,
) -> None:
    """Generate completions for the whole prompt batch in one blocking call."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = TextGenerator(
        checkpoints_dir=checkpoints_dir,
        max_seq_len=max_seq_len,
        device=device,
    )
    params = SamplingParams(temperature=temperature, top_p=top_p, max_gen_len=max_gen_len)

    completions = generator.generate(PROMPTS, params)
    for prompt, completion in zip(PROMPTS, completions):
        print(f"Prompt: {prompt}")
        print(f"Generated: {completion}\n\n==================================\n")


if __name__ == "__main__":
    cli_generate_stream()
