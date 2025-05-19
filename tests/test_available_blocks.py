import torch, gc
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import logging, json, os, sys

# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lite_llama.models.model_config import LlamaConfig

logger = logging.getLogger(__name__)


def load_config_from_json(json_file_path: str, device: str = "cuda") -> LlamaConfig:
    with open(json_file_path, "r") as f:
        config_dict = json.load(f)
    config = LlamaConfig(config_dict, max_seq_len=2048, device=device)
    return config


def _get_cache_block_size(model_config, block_size: int = 1) -> int:
    head_size = model_config.head_dim
    num_heads = model_config.num_kv_heads
    num_attention_layers = model_config.num_layers

    key_cache_block = block_size * num_heads * head_size
    value_cache_block = key_cache_block
    total = num_attention_layers * (key_cache_block + value_cache_block)
    dtype_size = 2  # torch.float16

    return dtype_size * total


@torch.inference_mode()
def determine_num_available_blocks(
    model_config, gpu_memory_utilization=0.9
) -> Tuple[int, int]:
    """
    评估模型的峰值内存使用情况，以确定在不发生内存溢出的情况下可以分配的 KV（键值）缓存块的数量。

    该方法首先清理 CUDA 缓存，然后使用虚拟输入执行一次前向传播，以评估模型的内存使用情况。
    接着，计算在剩余可用内存下，最多可以分配的 GPU 和 CPU 缓存块数量。

    提示：
        可以通过调整 `gpu_memory_utilization` 参数来限制 GPU 内存的使用。
    """
    # 清理 CUDA 缓存，以确保获取准确的内存使用信息
    torch.cuda.empty_cache()

    # 使用虚拟输入执行一次前向传播，以评估模型的内存使用情况

    # 同步 CUDA 操作，确保内存信息准确
    torch.cuda.synchronize()
    # 获取当前 GPU 的空闲内存和总内存（单位：字节）
    free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()
    # 计算模型加载后的峰值内存使用量
    # Get the peak memory allocation recorded by torch
    peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

    # 清理未使用的缓存，计算非Torch分配的内存
    torch.cuda.empty_cache()
    torch_allocated_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]

    total_allocated_bytes = torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]
    non_torch_allocations = total_allocated_bytes - torch_allocated_bytes

    if non_torch_allocations > 0:
        peak_memory += non_torch_allocations

    available_kv_cache_memory = total_gpu_memory * gpu_memory_utilization - peak_memory

    # 计算每个缓存块的大小
    cache_block_size = _get_cache_block_size(model_config)
    # 计算在剩余可用内存下，最多可以分配的 GPU 缓存块数量
    num_gpu_blocks = int(
        (total_gpu_memory * gpu_memory_utilization - peak_memory) // cache_block_size
    )
    # 确保缓存块数量不为负数
    num_gpu_blocks = max(num_gpu_blocks, 0)

    logger.info(
        "Memory profiling results: total_gpu_memory=%.2fGiB \n"
        " initial_memory_usage=%.2fGiB peak_torch_memory=%.2fGiB \n"
        " memory_usage_post_profile=%.2fGib \n"
        " non_torch_memory=%.2fGiB kv_cache_size=%.2fGiB \n"
        " gpu_memory_utilization=%.2f",
        total_gpu_memory / (1024**3),
        (total_gpu_memory - free_memory_pre_profile) / (1024**3),
        (peak_memory - non_torch_allocations) / (1024**3),
        total_allocated_bytes / (1024**3),
        non_torch_allocations / (1024**3),
        available_kv_cache_memory / (1024**3),
        gpu_memory_utilization,
    )

    # 进行垃圾回收，释放未使用的内存
    gc.collect()
    # 再次清理 CUDA 缓存
    torch.cuda.empty_cache()
    # 返回可分配的 GPU 和 CPU 缓存块数量（此处 CPU 块数量为 0）

    return num_gpu_blocks, 0


def load_original_llama(model_name_or_path: str, device: str = "cuda"):
    # config = LlamaConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.to(device)
    return model, tokenizer


if __name__ == "__main__":
    # 定义模型权重路径及配置参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_model_path = "/gemini/code/Llama-3.2-1B-Instruct"
    # 加载原始模型
    original_model, tokenizer = load_original_llama(original_model_path, device)
    # 定义模型配置参数
    json_file_path = (
        "/gemini/code/Llama-3.2-1B-Instruct/my_weight/config.json"  # JSON 文件的路径
    )
    model_config = load_config_from_json(json_file_path, device)  # 加载配置
    determine_num_available_blocks(model_config)
