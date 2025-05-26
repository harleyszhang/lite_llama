import torch
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
from lite_llama.utils.common import get_gpu_memory, detect_device, count_tokens, get_model_type
from lite_llama.utils.prompt_templates import get_prompter
from lite_llama.generate_stream import GenerateStreamText  # Original import
from lite_llama.quantization.gptq.gptq_executor import GPTQGenerateStreamText  # GPTQ import

import sys, os, time
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import psutil

process = psutil.Process(os.getpid())


def is_gptq_model(checkpoint_path: str) -> bool:
    """Check if the model is GPTQ quantized"""
    quantize_config_path = Path(checkpoint_path) / "quantization_config.json"
    return quantize_config_path.exists()


def report_resource_usage(ram_before, vram_before, gpu_type) -> None:
    end_time = time.time()
    ram_after = process.memory_info().rss
    vram_after = get_gpu_memory(gpu_type)

    ram_used = (ram_after - ram_before) / (1024**3)  # Bytes to GB

    if vram_before is not None and vram_after is not None:
        vram_used = vram_after - vram_before
        vram_text = f"{vram_used:.2f} GB"
    else:
        vram_text = "Unavailable"

    print(f"CPU RAM Used: {ram_used:.2f} GB")
    print(f"GPU VRAM Used: {vram_text}")


def main(
    prompt: str = "Hello, my name is",
    *,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gpu_num_blocks=40960,
    max_gen_len: Optional[int] = 1024,
    load_model: bool = True,
    compiled_model: bool = False,
    triton_weight: bool = True,
    gpu_type: str = "nvidia",
    checkpoint_path: Path = Path("checkpoints/lit-llama/7B/"),
    quantize: Optional[str] = None,
    use_gptq: Optional[bool] = None,  # New parameter for explicit GPTQ control
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert checkpoint_path.is_dir(), checkpoint_path
    checkpoint_path = str(checkpoint_path)

    if max_seq_len <= 1024:
        short_prompt = True
    else:
        short_prompt = False

    # Get model type and prompter
    model_prompter = get_prompter(
        get_model_type(checkpoint_path), checkpoint_path, short_prompt
    )

    # Start resource tracking
    ram_before = process.memory_info().rss
    gpu_type = detect_device()
    vram_before = get_gpu_memory(gpu_type)

    # Init LLM generator

    # Auto-detect GPTQ if not explicitly specified
    if use_gptq is None:
        use_gptq = is_gptq_model(checkpoint_path)
        if use_gptq:
            print(f"GPTQ quantized model detected in {checkpoint_path}")

    # Choose appropriate generator class
    if use_gptq:
        print("Using GPTQ-enabled generator")
        generator = GPTQGenerateStreamText(
            checkpoints_dir=checkpoint_path,
            tokenizer_path=checkpoint_path,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            load_model=load_model,
            compiled_model=compiled_model,
            triton_weight=triton_weight,
            device=device,
            use_gptq=True,  # Explicitly set GPTQ mode
        )
    else:
        print("Using standard FP16 generator")
        generator = GenerateStreamText(
            checkpoints_dir=checkpoint_path,
            tokenizer_path=checkpoint_path,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            load_model=load_model,
            compiled_model=compiled_model,
            triton_weight=triton_weight,
            device=device,
        )

    model_prompter.insert_prompt(prompt)
    prompts = [model_prompter.model_input]

    # Call the generation function and start the stream generation
    stream = generator.text_completion_stream(
        prompts,
        temperature=temperature,
        top_p=top_p,
        max_gen_len=max_gen_len,
    )

    completion = ""  # Initialize to generate the result
    # NOTE: After creating a generator, it can be iterated through a for loop
    text_msg = ""
    start = time.perf_counter()

    for batch_completions in stream:
        new_text = batch_completions[0]["generation"][len(completion) :]
        completion = batch_completions[0]["generation"]
        print(new_text, end="", flush=True)
        text_msg += new_text
    end = time.perf_counter()

    print("\n\n==================================\n")
    print(
        f"Time for inference: {(end - start):.2f} sec, {count_tokens(text_msg, generator.tokenizer) / (end - start):.2f} tokens/sec"
    )

    # Report resource usage
    report_resource_usage(ram_before, vram_before, gpu_type)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")

    # Create a wrapper function that adds the use_gptq parameter
    def main_with_gptq_option(
        prompt: str = "Hello, my name is",
        *,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 2048,
        max_gpu_num_blocks=40960,
        max_gen_len: Optional[int] = 1024,
        load_model: bool = True,
        compiled_model: bool = False,
        triton_weight: bool = True,
        gpu_type: str = "nvidia",
        checkpoint_path: Path = Path("checkpoints/lit-llama/7B/"),
        quantize: Optional[str] = None,
        force_gptq: bool = False,
        force_fp16: bool = False,
    ):
        """
        Generate text using lite_llama with automatic GPTQ detection

        Args:
            prompt: Input prompt text
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            max_seq_len: Maximum sequence length
            max_gpu_num_blocks: Maximum GPU memory blocks
            max_gen_len: Maximum generation length
            load_model: Whether to load model weights
            compiled_model: Whether to use compiled model
            triton_weight: Whether to use Triton kernels
            gpu_type: GPU type (nvidia/amd/cpu)
            checkpoint_path: Path to model checkpoint directory
            quantize: Quantization method (deprecated, kept for compatibility)
            force_gptq: Force GPTQ mode even if no quantization_config.json
            force_fp16: Force FP16 mode even if quantization_config.json exists
        """
        # Determine use_gptq based on force flags
        use_gptq = None
        if force_gptq and force_fp16:
            raise ValueError("Cannot force both GPTQ and FP16 modes simultaneously")
        elif force_gptq:
            use_gptq = True
        elif force_fp16:
            use_gptq = False

        return main(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_gen_len=max_gen_len,
            load_model=load_model,
            compiled_model=compiled_model,
            triton_weight=triton_weight,
            gpu_type=gpu_type,
            checkpoint_path=checkpoint_path,
            quantize=quantize,
            use_gptq=use_gptq,
        )

    CLI(main_with_gptq_option)