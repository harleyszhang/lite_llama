"""Basic batch inference for both text and vision-language models.

This example mirrors the entry-level style of ``vllm/examples/basic.py``:
it builds a :class:`TextGenerator` for batch text completion and a
:class:`VisionGenerator` for batch image-conditioned completion, then prints
every result so the script can be run directly without extra setup.

Run from the repository root:

    python examples/basic.py

By default both the text and vision examples run; use ``--text-only`` or
``--vision-only`` to run just one of them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from lite_llama.engine import SamplingParams, TextGenerator, VisionGenerator

TEXT_MODEL = "my_weight/Qwen2.5-0.5B"
VISION_MODEL = "my_weight/Qwen3-VL-4B-Instruct"
IMAGE = "images/llava_test/dog.jpeg"

TEXT_PROMPTS: list[str] = [
    "The future of artificial intelligence is",
    "In three sentences, explain quantum computing:",
    "Once upon a time, in a quiet village,",
]

VISION_PROMPTS: list[str] = [
    "Describe this image in one sentence.",
    "What animal is in the picture?",
    "What colors are dominant in this image?",
]


def run_text_batch(model_dir: str, max_gen_len: int = 64) -> None:
    """Batch text generation with :class:`TextGenerator`."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = TextGenerator(
        checkpoints_dir=model_dir,
        max_seq_len=2048,
        device=device,
    )
    params = SamplingParams(temperature=0.0, top_p=1.0, max_gen_len=max_gen_len)

    print("=" * 60)
    print("Text model batch inference")
    print("=" * 60)
    completions = generator.generate(TEXT_PROMPTS, params)
    for prompt, completion in zip(TEXT_PROMPTS, completions, strict=True):
        print(f"Prompt: {prompt}")
        print(f"Output: {completion}")
        print("-" * 60)


def run_vision_batch(model_dir: str, image_path: str, max_gen_len: int = 64) -> None:
    """Batch vision-language generation with :class:`VisionGenerator`.

    The public VisionGenerator currently accepts one prompt at a time, so this
    function runs the image-conditioned prompts sequentially while reusing the
    same engine for each request.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = VisionGenerator(
        checkpoints_dir=model_dir,
        max_seq_len=2048,
        device=device,
    )
    params = SamplingParams(temperature=0.0, top_p=1.0, max_gen_len=max_gen_len)

    image = Image.open(image_path).convert("RGB")

    print("=" * 60)
    print("Vision-language model batch inference")
    print("=" * 60)
    for prompt in VISION_PROMPTS:
        print(f"Prompt: {prompt}")
        print("Output: ", end="", flush=True)
        for delta in generator.stream(prompt, [image], params):
            print(delta, end="", flush=True)
        print("\n" + "-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lite-LLaMA basic batch inference example.")
    parser.add_argument("--text-only", action="store_true", help="Only run the text model example.")
    parser.add_argument(
        "--vision-only", action="store_true", help="Only run the vision model example."
    )
    parser.add_argument(
        "--text-model", default=TEXT_MODEL, help="Path to a lite_llama text checkpoint."
    )
    parser.add_argument(
        "--vision-model", default=VISION_MODEL, help="Path to a lite_llama vision checkpoint."
    )
    parser.add_argument("--image", default=IMAGE, help="Image path used by the vision example.")
    parser.add_argument("--max-gen-len", type=int, default=64, help="Maximum number of new tokens.")
    args = parser.parse_args()

    if not args.vision_only:
        if not Path(args.text_model).exists():
            print(f"Text checkpoint not found: {args.text_model}; skipping text example.")
        else:
            run_text_batch(args.text_model, args.max_gen_len)

    if not args.text_only:
        if not Path(args.vision_model).exists():
            print(f"Vision checkpoint not found: {args.vision_model}; skipping vision example.")
        elif not Path(args.image).exists():
            print(f"Image not found: {args.image}; skipping vision example.")
        else:
            run_vision_batch(args.vision_model, args.image, args.max_gen_len)


if __name__ == "__main__":
    main()
