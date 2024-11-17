import torch
import logging 
from config import LlamaConfig
from typing import List

logger = logging.getLogger(__name__)

def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    return torch.tensor([], dtype=dtype).element_size()

class ComputeMaxAvailableBlocks:
    """A class that can execute a forward pass with dummy inputs to profile the memory usage of the model.
    and  calculate the maximum possible number of GPU blocks that can be allocated with the remaining free memory.
    if not execute dummy forward run, it should be run after cuda graph!
    """
    def __init__(self, model_config: LlamaConfig, gpu_memory_utilization=0.9, block_size=1):
        self.model_config = model_config
        self.gpu_memory_utilization = gpu_memory_utilization
        self.block_size = block_size # 一个 block 表示多少个 tokens
        self.dtype = self.model_config.torch_dtype
        
        if self.dtype in ["float16", "bfloat16", "fp16", "bfp16"]:
            self.dtype_size = 2
        elif self.dtype in ["int8", "fp18"]:
            self.dtype_size = 1 # byte
        else:
            print(f"Unsupported dtype: {self.dtype_size}!")
        
    def compute_cache_block_size_bytes(self,):
        """Get the size of the KV cache block size in bytes.
        """
        if self.head_dim is None:
            head_size = self.model_config.hidden_size // self.model_config.num_heads
        else:
            head_size = self.model_config.head_dim
        
        num_layers = self.model_config.num_layers
        num_kv_heads = self.model_config.num_kv_heads
        # num_heads * head_size = hidden_size
        kv_cache_token_bytes_per_layer = (num_kv_heads * head_size) * 2 * self.dtype_size
        transformer_kv_cache_token_bytes = kv_cache_token_bytes_per_layer * num_layers

        transformer_kv_cache_blocks_bytes = transformer_kv_cache_token_bytes * self.block_size

        return transformer_kv_cache_blocks_bytes

    def compute_num_available_blocks(self, model_path=None, dummy_input = None, model_byes=None):
        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()
        # 获取峰值内存分配
        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # 清理未使用的缓存，计算非Torch分配的内存
        torch.cuda.empty_cache()
        torch_allocated_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        total_allocated_bytes = torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes

        # 如果有非Torch分配的内存，增加到峰值内存中
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations

        # 计算可用于KV缓存的内存, 单位 字节
        available_kv_cache_memory = (
            total_gpu_memory * self.gpu_memory_utilization -
            peak_memory
        )

        cache_block_size = self.compute_cache_block_size_bytes() # 单位字节

        if cache_block_size == 0:
            max_gpu_num_blocks = 0
        else:
            max_gpu_num_blocks = int(available_kv_cache_memory // cache_block_size)
        
        logger.info(
            "Memory profiling result: total_gpu_memory=%.2f GB"
            " peak_torch_memory = %.2f GB non_torch_memory=%.2f GiB",
            " kv_cache_size = %.2fGiB"
            " available_kv_cache_memory=%.2f GiB max_gpu_num_blocks=%d",
            total_gpu_memory / (1024**3),
            (peak_memory - non_torch_allocations) / (1024**3),
            non_torch_allocations / (1024**3),
            available_kv_cache_memory / (1024**3),
            max_gpu_num_blocks,
        )

        return max_gpu_num_blocks
    

class KVCacheMemoryManager:
    def __init__(self, head_dim, num_kv_heads, num_layers, num_gpu_blocks, dtype, device="cuda"):
        
        # Initialize the gpu_kv_buffer
        self._allocate_kv_cache(
            head_dim, num_kv_heads, num_layers, 
            num_gpu_blocks,  
            dtype, device)

    def _allocate_kv_cache(self, 
        head_dim, num_kv_heads, num_layers,
        max_num_blocks: int,
        dtype,
        device: str="cuda"
    )-> List[torch.Tensor]:
        # kv cache shape: config.max_batch_size, config.max_seq_len, self.num_kv_heads, self.head_dim

        max_num_tokens = max_num_blocks * self.block_size
        # TODO 修改 kv buffer 形状支持 PagedAttention
        self.gpu_kv_buffer = [
            torch.empty((max_num_tokens, 2 * num_kv_heads, head_dim), dtype=dtype) for _ in range(num_layers)
        ]
    
    # 释放键值缓存缓冲区
    def _free_buffers(self):
        self.kv_buffer = None