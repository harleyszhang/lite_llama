import torch
from typing import Optional
from lite_llama.utils.prompt_templates import get_prompter, get_image_token
from lite_llama.generate_stream import GenerateStreamText  # import GenerateText
from lite_llama.utils.image_process import vis_images

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
from lite_llama.utils.common import get_gpu_memory, detect_device, count_tokens, get_model_type, quantization
from lite_llama.llava_generate_stream import LlavaGeneratorStream

import sys, os, time
from pathlib import Path
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import psutil
from lite_llama.utils.logger import log

process = psutil.Process(os.getpid())

def report_resource_usage(ram_before, vram_before) -> None:
    end_time = time.time()
    ram_after = process.memory_info().rss
    vram_after = get_gpu_memory(detect_device())

    ram_used = (ram_after - ram_before) / (1024 ** 3)  # Bytes to GB

    if vram_before is not None and vram_after is not None:
        vram_used = vram_after - vram_before
        vram_text = f"{vram_used:.2f} GB"
    else:
        vram_text = "Unavailable"

    log.info(f"CPU RAM Used: {ram_used:.2f} GB")
    log.info(f"GPU VRAM Used: {vram_text}")


def generate_llama(
        prompt: str = "Hello, my name is",
        quantize: Optional[str] = None,
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
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert checkpoint_path.is_dir(), checkpoint_path
    checkpoint_path = str(checkpoint_path)
    if max_seq_len <= 1024:
        short_prompt = True
    else:
        short_prompt = False
    model_prompter = get_prompter(get_model_type(checkpoint_path), checkpoint_path, short_prompt)

    # Start resource tracking
    ram_before = process.memory_info().rss

    vram_before = get_gpu_memory(gpu_type)

    # Init LLM generator
    with quantization(quantize):

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

    completion = ''  # Initialize to generate the result
    # NOTE: After creating a generator, it can be iterated through a for loop
    text_msg = ""
    start = time.perf_counter()
    for batch_completions in stream:
        new_text = batch_completions[0]['generation'][len(completion):]
        completion = batch_completions[0]['generation']
        print(new_text, end='', flush=True)
        text_msg +=new_text
    end = time.perf_counter()

    print("\n\n==================================\n")
    log.info(f"Time for inference: {(end - start):.2f} sec, {count_tokens(text_msg, generator.tokenizer)/(end - start):.2f} tokens/sec")

    # Report resource usage
    report_resource_usage(ram_before, vram_before)


def generate_llava(
        prompt: str = "Hello, my name is",
        checkpoint_path: Path = Path("checkpoints/lit-llama/7B/"),
        figure_path: Path = Path("figures/lit-llama/"),
        gpu_type: str = "nvidia",
        quantize: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 2048,
        max_gpu_num_blocks=None,
        max_gen_len: Optional[int] = 512,
        load_model: bool = True,
        compiled_model: bool = False,
        triton_weight: bool = True
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if max_seq_len <= 1024:
        short_prompt = True
    else:
        short_prompt = False

    if not os.path.isfile(figure_path):
        log.error(f"'{figure_path}' Not a valid file path！")
    else:
        image_input = str(figure_path).strip()
    image_items = [image_input]  # Prepare the image_items list
    image_num = len(image_items)  # Calculate the number of input images
    vis_images(image_items)  # Displaying images in the terminal
    assert checkpoint_path.is_dir(), checkpoint_path
    checkpoint_path = str(checkpoint_path)
    model_prompter = get_prompter("llama", checkpoint_path, short_prompt)

    # Start resource tracking
    ram_before = process.memory_info().rss

    vram_before = get_gpu_memory(gpu_type)

    # Initializing the Multimodal Model Text Generator
    with quantization(quantize):

        try:
            generator = LlavaGeneratorStream(
                checkpoints_dir=checkpoint_path,
                tokenizer_path=checkpoint_path,
                max_gpu_num_blocks=max_gpu_num_blocks,
                max_seq_len=max_seq_len,
                load_model=load_model,
                compiled_model=compiled_model,
                triton_weight=triton_weight,
                device=device,
            )
        except Exception as e:
            log.error(f"Model loading failure: {e}")
            sys.exit(1)

    image_token = get_image_token()
    model_prompter.insert_prompt(image_token * image_num + prompt)
    prompts = [model_prompter.model_input]

    try:
        stream = generator.text_completion_stream(
            prompts,
            image_items,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
        )
    except Exception as e:
        log.error(f"Text Generation Failure: {e}")

    completion = ''  # Initialization generates results
    text_msg = ""
    start = time.perf_counter()

    for batch_completions in stream:
        next_text = batch_completions[0]['generation'][len(completion):]
        completion = batch_completions[0]['generation']
        print(f"\033[91m{next_text}\033[0m", end='', flush=True)  # 红色文本
        text_msg += next_text
    end = time.perf_counter()

    print("\n\n==================================\n")
    log.info(f"Time for inference: {(end - start):.2f} sec, {count_tokens(text_msg, generator.tokenizer)/(end - start):.2f} tokens/sec")
    # Report resource usage
    report_resource_usage(ram_before, vram_before)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")

    # Create a wrapper function that adds the use_gptq parameter
    def main(
        prompt: str = "Hello, my name is",
        checkpoint_path: Path = Path("checkpoints/lite-llama/7B/"),
        figure_path: Optional[Path] = None,
        quant: str = "gpt.int4"
    ):
        """
        Generate text using lite_llama with automatic GPTQ detection

        Args:
            prompt: Input prompt text
            checkpoint_path: Path to model checkpoint directory
            figure_path: Path to Image file for LLaVA generation, optional
            quant: GPTQ quantization mode
        """
        # Determine use_gptq based on force flags
        gpu_type = detect_device()
        model_path = os.path.abspath(checkpoint_path)
        if figure_path:
            generate_llava(prompt=prompt, checkpoint_path=Path(model_path), figure_path=Path(figure_path),
                           gpu_type=gpu_type, quantization=quant)
        else:
            generate_llama(prompt=prompt, checkpoint_path=Path(model_path), gpu_type=gpu_type, quantization=quant)

    CLI(main)