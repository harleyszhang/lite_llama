"""Single-image chat with :class:`~rapid_llm.engine.generator.VisionGenerator`.

Builds a :class:`VisionGenerator` over an LLaVA checkpoint, runs one
image + question turn, and prints the answer — the multimodal smoke
test.

Usage:
    python examples/example_llava.py   # set paths at the top first
"""

from __future__ import annotations

import torch
from PIL import Image

from rapid_llm.engine import SamplingParams, VisionGenerator

checkpoints_dir = "my_weight/llava-1.5-7b-hf"  # 改成自己的存放模型路径
image_path = "/path/to/your/image.jpg"  # 改成自己的图片路径


def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 256,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = VisionGenerator(
        checkpoints_dir=checkpoints_dir,
        max_seq_len=max_seq_len,
        device=device,
    )
    params = SamplingParams(temperature=temperature, top_p=top_p, max_gen_len=max_gen_len)

    image = Image.open(image_path).convert("RGB")
    # LLaVA expects the <image> marker where the visual tokens should be
    # inserted; Qwen3-VL takes a plain prompt instead.
    prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"

    for delta in generator.stream(prompt, [image], params):
        print(delta, end="", flush=True)
    print("\n\n==================================\n")


if __name__ == "__main__":
    main()
