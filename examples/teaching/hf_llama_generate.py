"""教学示例:HuggingFace ``generate()`` 与 ``pipeline()`` 两种基线用法。

作为 lite_llama 的性能/精度对照基线保留:同样的 prompt 与采样参数下,
``benchmarks/bench_hf_baseline.py`` 用的就是这套调用方式。

用法::

    python examples/teaching/hf_llama_generate.py --model-dir my_weight/Llama-3.2-1B-Instruct
"""

import argparse
import os

import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast, pipeline


def load_llama_model(model_path: str, device: str = "cuda"):
    """
    Load the LLaMA model and tokenizer.

    Args:
        model_path (str): Path to the directory containing the model files.
        tokenizer_path (str): Path to the directory containing the tokenizer files.
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        model: The loaded LLaMA model.
        tokenizer: The loaded tokenizer.
    """
    tokenizer = LlamaTokenizerFast.from_pretrained(model_path, legacy=False)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use float16 for faster inference if supported
        low_cpu_mem_usage=True,
    )
    model.to(device)
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_length: int = 50, device: str = "cuda"):
    """
    Generate text using the LLaMA model.

    Args:
        model: The loaded LLaMA model.
        tokenizer: The loaded tokenizer.
        prompt (str): The input text prompt.
        max_length (int): The maximum length of the generated text.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        str: The generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,  # Enable sampling to introduce randomness
            temperature=0.7,  # Adjust temperature for creativity
            top_p=0.9,  # Use top-p (nucleus) sampling
            repetition_penalty=1.2,  # Penalize repetitions
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def pipline_text(model_id):
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("LITE_LLAMA_MODEL_DIR"),
        required=os.environ.get("LITE_LLAMA_MODEL_DIR") is None,
        help="HF 格式的 Llama 权重目录(或设置 LITE_LLAMA_MODEL_DIR)",
    )
    parser.add_argument("--prompt", default="I believe the meaning of life is,")
    parser.add_argument("--max-length", type=int, default=100)
    args = parser.parse_args()
    model_path = args.model_dir

    # Load the model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_llama_model(model_path, device)

    # Test the model with a sample prompt
    prompt = args.prompt
    generated_text = generate_text(
        model, tokenizer, prompt, max_length=args.max_length, device=device
    )

    print("Prompt:")
    print(prompt)
    print("\nGenerated Text:")
    print(generated_text)

    pipline_text(model_path)
