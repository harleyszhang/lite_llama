import torch
import logging 
from typing import List
from ..models.model_config import LlamaConfig

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
    def __init__(self, head_dim, num_kv_heads, num_layers, gpu_num_blocks, max_num_tokens, block_size=1, dtype=torch.float16, device="cuda"):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.gpu_num_blocks = gpu_num_blocks
        self.block_size = block_size
        self.max_num_tokens = gpu_num_blocks * block_size
        self.dtype = dtype
        self.device = device

        # 定义 kv 内存位置索引和内存使用状态变量
        self.kv_mem_pos_indexs = torch.arange(0, self.max_num_tokens, dtype=torch.long, device="cuda")
        self.kv_mem_use_state = torch.zeros(self.max_num_tokens, dtype = torch.int32, device="cuda")
        self.can_use_mem_size = gpu_num_blocks # 可用的 kv cache tokens 数量

        # Initialize the gpu_kv_buffer
        self._allocate_kv_cache(
            head_dim, num_kv_heads, num_layers, 
            gpu_num_blocks,  
            dtype, device)

    def _allocate_kv_cache(self, 
        max_num_tokens,
        head_dim, num_kv_heads, num_layers,
        dtype,
        device: str="cuda"
    )-> List[torch.Tensor]:
        # kv cache shape: config.max_batch_size, config.max_seq_len, self.num_kv_heads, self.head_dim

        # max_num_tokens = max_num_blocks * self.block_size
        # TODO 修改 kv buffer 形状支持 PagedAttention
        self.gpu_kv_buffer = [
            torch.empty((max_num_tokens, 2 * num_kv_heads, head_dim), dtype=dtype, device=device) for _ in range(num_layers)
        ]
    
    @torch.no_grad()
    def alloc_kv_cache(self, need_size):
        if need_size > self.can_use_mem_size:
            logger.warn(f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}")
            return None
        
        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        select_index = can_use_pos_index[0:need_size]
        self.add_refs(select_index)
        
        return select_index
    
    @torch.no_grad()
    def alloc_contiguous_kvcache(self, need_size):
        if self.can_use_mem_size < need_size:
            logger.info(f"warn no enough contiguous cache need_size {need_size} left_size {self.can_use_mem_size}")
            return None
        
        # batch 大小是动态变化的
        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        
        # 可用块索引中排除了最后 need_size - 1 个索引，因为这些索引作为起始点时，没有足够的块来满足连续的 need_size 个块的需求。
        start_indexs = can_use_pos_index[:-need_size + 1]
        # 对应的排除前面的 need_size - 1 个索引，如果以这些索引作为终点时，不满足需求
        end_indexs = can_use_pos_index[need_size - 1]
        # 计算每对 start 和 end 之间的差值
        diff = end_indexs - start_indexs
        
        contiguous_blocks = (diff == need_size - 1)
        # 找到那些差值等于 need_size - 1 的位置，意味着 start 和 end 之间有连续的 need_size 个块
        contiguous_blocks = (diff == (need_size - 1)).nonzero(as_tuple=True)[0]
        
        if contiguous_blocks.numel() > 0: # numel 返回张量种元素数量
            start_index = start_indexs[contiguous_blocks[0]].item() # item 方法将张量转换成 Python 数值
            end_index = start_index + need_size
            select_index = self.kv_mem_pos_indexs[start_index:end_index]
            self.add_refs(select_index)
            return select_index, start_index, end_index

        return None
    
    @torch.no_grad()
    def add_refs(self, token_index: torch.Tensor):
        state = self.kv_mem_use_state[token_index]
        has_used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size -= all_tokens - has_used_tokens
  
        self.kv_mem_use_state[token_index] += 1
        return
    
    @torch.no_grad()
    def decrease_refs(self, token_index: torch.Tensor):
        # 使用 unique 方法获取 token_index 中唯一的 token 索引，并返回每个唯一索引在原始张量中出现的次数。
        token_index, counts = token_index.unique(return_counts=True)
        # 当引用计数减少到零时，意味着该缓存块可以被释放或重新分配。
        self.kv_mem_use_state[token_index] -= counts
        state = self.kv_mem_use_state[token_index]
        used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size += all_tokens - used_tokens
        return
    
    # 释放键值缓存缓冲区
    def _free_buffers(self):
        self.gpu_kv_buffer = None
    
    @torch.no_grad()
    def free_all(self,):
        self.can_use_mem_size = len(self.kv_mem_use_state)
        self.kv_mem_use_state[:] = 0
