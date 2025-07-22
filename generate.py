# 对原有的generate.py进行修改，添加量化支持

import torch
from typing import Optional, List
from lite_llama.utils.prompt_templates import get_prompter, get_image_token
from lite_llama.generate_stream import GenerateStreamText
from lite_llama.utils.image_process import vis_images

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
from lite_llama.utils.common import get_gpu_memory, detect_device, count_tokens, get_model_type
from lite_llama.llava_generate_stream import LlavaGeneratorStream

import sys, os, time
from pathlib import Path
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import psutil
from lite_llama.utils.logger import log

# 新增导入
from lite_llama.quantization.quant_manager import quantization_manager, QuantizationType

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
        quantization: Optional[str] = None,  # 新增参数
        *,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 2048,
        max_gpu_num_blocks=40960,
        max_gen_len: Optional[int] = 1024,
        compiled_model: bool = False,
        gpu_type: str = "nvidia",
        checkpoint_path: Path = Path("checkpoints/lit-llama/7B/"),
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert checkpoint_path.is_dir(), checkpoint_path
    checkpoint_path = str(checkpoint_path)

    # 检测量化类型
    if quantization is None:
        quantization = quantization_manager.detect_quantization_type(checkpoint_path)
        if quantization != QuantizationType.NONE:
            log.info(f"自动检测到量化类型: {quantization}")

    if max_seq_len <= 1024:
        short_prompt = True
    else:
        short_prompt = False
    model_prompter = get_prompter(get_model_type(checkpoint_path), checkpoint_path, short_prompt)

    # Start resource tracking
    ram_before = process.memory_info().rss
    vram_before = get_gpu_memory(gpu_type)

    # 创建生成器，传入量化参数
    generator = GenerateStreamText(
        checkpoints_dir=checkpoint_path,
        tokenizer_path=checkpoint_path,
        max_gpu_num_blocks=max_gpu_num_blocks,
        max_seq_len=max_seq_len,
        compiled_model=compiled_model,
        device=device,
        quantization=quantization,  # 新增参数
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

    completion = ''
    text_msg = ""
    start = time.perf_counter()
    for batch_completions in stream:
        new_text = batch_completions[0]['generation'][len(completion):]
        completion = batch_completions[0]['generation']
        print(new_text, end='', flush=True)
        text_msg += new_text
    end = time.perf_counter()

    print("\n\n==================================\n")
    log.info(
        f"Time for inference: {(end - start):.2f} sec, {count_tokens(text_msg, generator.tokenizer) / (end - start):.2f} tokens/sec")

    # Report resource usage
    report_resource_usage(ram_before, vram_before)


def generate_llava(
        prompt: str = "Hello, my name is",
        checkpoint_path: Path = Path("checkpoints/lit-llama/7B/"),
        figure_path: Path = Path("figures/lit-llama/"),
        gpu_type: str = "nvidia",
        quantization: Optional[str] = None,  # 新增参数
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 2048,
        max_gpu_num_blocks=None,
        max_gen_len: Optional[int] = 512,
        compiled_model: bool = False,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 检测量化类型
    if quantization is None:
        quantization = quantization_manager.detect_quantization_type(str(checkpoint_path))
        if quantization != QuantizationType.NONE:
            log.info(f"自动检测到量化类型: {quantization}")
    
    if max_seq_len <= 1024:
        short_prompt = True
    else:
        short_prompt = False

    if not os.path.isfile(figure_path):
        log.error(f"'{figure_path}' Not a valid file path！")
    else:
        image_input = str(figure_path).strip()
    image_items = [image_input]
    image_num = len(image_items)
    vis_images(image_items)
    assert checkpoint_path.is_dir(), checkpoint_path
    checkpoint_path = str(checkpoint_path)
    model_prompter = get_prompter("llama", checkpoint_path, short_prompt)

    # Start resource tracking
    ram_before = process.memory_info().rss
    vram_before = get_gpu_memory(gpu_type)

    try:
        generator = LlavaGeneratorStream(
            checkpoints_dir=checkpoint_path,
            tokenizer_path=checkpoint_path,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            compiled_model=compiled_model,
            device=device,
            quantization=quantization,  # 新增参数
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

    completion = ''
    text_msg = ""
    start = time.perf_counter()

    for batch_completions in stream:
        next_text = batch_completions[0]['generation'][len(completion):]
        completion = batch_completions[0]['generation']
        print(f"\033[91m{next_text}\033[0m", end='', flush=True)
        text_msg += next_text
    end = time.perf_counter()

    print("\n\n==================================\n")
    log.info(
        f"Time for inference: {(end - start):.2f} sec, {count_tokens(text_msg, generator.tokenizer) / (end - start):.2f} tokens/sec")

    # Report resource usage
    report_resource_usage(ram_before, vram_before)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")


    def main(
            prompt: str = "Hello, my name is",
            checkpoint_path: Path = Path("checkpoints/lite-llama/7B/"),
            figure_path: Optional[Path] = None,
            quantization: Optional[str] = None,  # 新增参数
    ):
        """
        Generate text using lite_llama with optional quantization support

        Args:
            prompt: Input prompt text
            checkpoint_path: Path to model checkpoint directory
            figure_path: Path to Image file for LLaVA generation, optional
            quantization: Quantization method ('gptq', 'awq', 'smoothquant', or None for auto-detection)
        """
        gpu_type = detect_device()
        model_path = os.path.abspath(checkpoint_path)

        # 验证量化参数
        if quantization and quantization not in ['gptq', 'awq', 'smoothquant']:
            log.error(f"不支持的量化方法: {quantization}")
            log.info("支持的量化方法: gptq, awq, smoothquant")
            return

        if figure_path:
            generate_llava(
                prompt=prompt,
                checkpoint_path=Path(model_path),
                figure_path=Path(figure_path),
                gpu_type=gpu_type,
                quantization=quantization,
            )
        else:
            generate_llama(
                prompt=prompt,
                checkpoint_path=Path(model_path),
                gpu_type=gpu_type,
                quantization=quantization
            )


    CLI(main)


# 新增量化推理的便捷函数
def run_quantized_inference(
    model_path: str,
    prompt: str,
    quantization_method: Optional[str] = None,
    **kwargs
):
    """
    运行量化推理的便捷函数
    
    Args:
        model_path: 模型路径
        prompt: 输入提示
        quantization_method: 量化方法，None为自动检测
        **kwargs: 其他推理参数
    """
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    # 获取模型类型
    model_type = get_model_type(model_path)
    
    # 设置默认参数
    default_params = {
        'temperature': 0.6,
        'top_p': 0.9,
        'max_seq_len': 2048,
        'max_gen_len': 1024,
        'compiled_model': False,
    }
    default_params.update(kwargs)
    
    if model_type == 'llava':
        # LLaVA模型需要图像输入
        figure_path = kwargs.get('figure_path')
        if not figure_path:
            log.warning("LLaVA模型需要图像输入，将使用默认图像")
            # 这里可以设置一个默认图像路径
        
        generate_llava(
            prompt=prompt,
            checkpoint_path=Path(model_path),
            figure_path=Path(figure_path) if figure_path else None,
            quantization=quantization_method,
            **default_params
        )
    else:
        generate_llama(
            prompt=prompt,
            checkpoint_path=Path(model_path),
            quantization=quantization_method,
            **default_params
        )


# 量化性能测试函数
def benchmark_quantized_model(
    model_path: str,
    quantization_methods: Optional[List[str]] = None,
    test_prompts: Optional[List[str]] = None,
    num_runs: int = 3
):
    """
    对量化模型进行性能基准测试
    
    Args:
        model_path: 模型路径
        quantization_methods: 要测试的量化方法列表
        test_prompts: 测试提示列表
        num_runs: 每个配置的运行次数
    """
    
    if quantization_methods is None:
        quantization_methods = ['gptq', 'awq', 'smoothquant', None]  # None代表无量化
    
    if test_prompts is None:
        test_prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot."
        ]
    
    results = {}
    
    for method in quantization_methods:
        method_name = method or "no_quantization"
        log.info(f"测试量化方法: {method_name}")
        
        method_results = []
        
        for prompt in test_prompts:
            prompt_results = []
            
            for run in range(num_runs):
                log.info(f"运行 {run + 1}/{num_runs}: {prompt[:50]}...")
                
                start_time = time.time()
                try:
                    run_quantized_inference(
                        model_path=model_path,
                        prompt=prompt,
                        quantization_method=method,
                        max_gen_len=256  # 限制生成长度以便快速测试
                    )
                    end_time = time.time()
                    prompt_results.append(end_time - start_time)
                    
                except Exception as e:
                    log.error(f"测试失败 ({method_name}, run {run + 1}): {e}")
                    prompt_results.append(float('inf'))
            
            method_results.append(prompt_results)
        
        results[method_name] = method_results
    
    # 打印结果摘要
    log.info("=" * 60)
    log.info("基准测试结果摘要")
    log.info("=" * 60)
    
    for method_name, method_results in results.items():
        avg_times = []
        for prompt_results in method_results:
            valid_times = [t for t in prompt_results if t != float('inf')]
            if valid_times:
                avg_times.append(sum(valid_times) / len(valid_times))
        
        if avg_times:
            overall_avg = sum(avg_times) / len(avg_times)
            log.info(f"{method_name:15}: {overall_avg:.2f}s 平均响应时间")
        else:
            log.info(f"{method_name:15}: 测试失败")
    
    return results