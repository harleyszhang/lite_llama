import torch
import json, gc
from pathlib import Path

from ..utils.dummy_data import DummyInputGenerator
from .executor_struct import AttentionInfo, CONFIG_CLASS_MAP
from ..utils.logger import get_logger

logger = get_logger(__name__)


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    return torch.tensor([], dtype=dtype).element_size()

class ComputeMaxAvailableBlocks:
    """A class that can execute a forward pass with dummy inputs to profile the memory usage of the model.
    and  calculate the maximum possible number of GPU blocks that can be allocated with the remaining free memory.
    if not execute dummy forward run, it should be run after cuda graph!
    """
    def __init__(
        self, 
        num_layers, 
        hidden_size, 
        num_heads, 
        num_kv_heads, 
        head_dim, 
        gpu_memory_utilization=0.9, 
        block_size=1, 
        dtype=torch.float16,
        device="cuda"
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.head_dim = head_dim

        self.gpu_memory_utilization = gpu_memory_utilization
        self.block_size = block_size # 一个 block 表示多少个 tokens
        self.dtype = dtype
        self.device = device
        self.dtype_size = get_dtype_size(dtype)
        
    def compute_cache_block_size_bytes(self):
        """Get the size of the KV cache block size in bytes.
        """

        kv_cache_token_bytes_per_layer = (self.num_kv_heads * self.head_dim) * 2 * self.dtype_size
        transformer_kv_cache_token_bytes = kv_cache_token_bytes_per_layer * self.num_layers

        transformer_kv_cache_blocks_bytes = transformer_kv_cache_token_bytes * self.block_size

        return transformer_kv_cache_blocks_bytes

    def compute_num_available_blocks(self, model, model_path=None):
        """
        评估模型的峰值内存使用情况，以确定在不发生内存溢出的情况下可以分配的 KV（键值）缓存块的数量。

        该方法首先清理 CUDA 缓存，然后使用虚拟输入执行一次前向传播，以评估模型的内存使用情况。
        接着，计算在剩余可用内存下，最多可以分配的 GPU 和 CPU 缓存块数量。

        提示：
            可以通过调整 `gpu_memory_utilization` 参数来限制 GPU 内存的使用。
        """
        # 清理 CUDA 缓存，以确保获取准确的内存使用信息
        # NOTE: torch.cuda.empty_cache() 用于释放 GPU 上由缓存分配器持有的未占用内存。
        # NOTE: torch.cuda.reset_peak_memory_stats() 用于重置 CUDA 内存分配器所跟踪的“峰值”统计数据。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 获取当前 GPU 的空闲内存和总内存（单位：字节）# free_memory_pre_profile=9178578944
        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()
        # 使用虚拟输入执行一次前向传播，以评估模型的内存使用情况        
        params_path = Path(model_path) / "config.json"
        if params_path.exists():
            with open(params_path, "r") as f:
                params = json.load(f)
                model_config = CONFIG_CLASS_MAP.get(params["model_type"].lower())
                    
                # 创建虚拟输入 
                batch_size = 1
                seq_len = 32  # 使用较小的序列长度进行内存评估
                dummy_generator = DummyInputGenerator(device="cuda")
                dummy_input, dummy_position_ids = dummy_generator.generate_dummy_input(model_config, batch_size, seq_len)
                    
                # 创建虚拟的 atten_info 对象
                dummy_atten_info = AttentionInfo()
                
                dummy_atten_info.kv_buffer = [
                    torch.empty((seq_len, 2 * self.num_kv_heads, self.head_dim), dtype=self.dtype, device=self.device) for _ in range(self.num_layers)
                ]
                
                dummy_atten_info.cur_select_index = torch.arange(seq_len, dtype=torch.int32, device="cuda")
                dummy_atten_info.b_start_loc = torch.tensor([0], dtype=torch.int32, device="cuda")
                dummy_atten_info.b_seq_len = torch.tensor([1], device="cuda")
                dummy_atten_info.max_actual_seq_len=seq_len
                # 执行前向传播
                with torch.no_grad():
                    _ = model(dummy_input, dummy_position_ids, dummy_atten_info)

        logger.info(f"模型加载后可用内存: {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")
        # 同步 CUDA 操作，确保内存信息准确
        torch.cuda.synchronize()
        # 计算模型加载后的峰值内存使用量. Get the peak memory allocation recorded by torch
        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # 清理未使用的缓存，计算非 Torch 分配的内存. 检查是否有任何剩余内存可能已在“torch”之外的 gpu 上分配。例如，NCCL 操作在前向传递期间可能会使用几 GB
        torch.cuda.empty_cache()
        torch_allocated_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        total_allocated_bytes = torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
        
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations

        available_kv_cache_memory = (
            total_gpu_memory * self.gpu_memory_utilization -
            peak_memory)
        
        # 计算每个缓存块的大小
        cache_block_size = self.compute_cache_block_size_bytes()
        # 计算在剩余可用内存下，最多可以分配的 GPU 缓存块数量
        num_gpu_blocks = int(
            (total_gpu_memory * self.gpu_memory_utilization -
            peak_memory) // cache_block_size
        )
        
        num_gpu_blocks = max(num_gpu_blocks, 0) # 确保缓存块数量不为负数

        logger.info(
            f" Memory profiling results: total_gpu_memory = {total_gpu_memory / (1024**3):.2f} GB \n"
            f"    initial_memory_usage = {(total_gpu_memory - free_memory_pre_profile) / (1024**3):.2f} GB "
            f"peak_torch_memory = {(peak_memory - non_torch_allocations) / (1024**3):.2f} GB \n"
            f"    memory_usage_post_profile = {total_allocated_bytes / (1024**3):.2f} GB \n"
            f"    non_torch_memory = {non_torch_allocations / (1024**3):.2f} GB, "
            f"kv_cache_size = {available_kv_cache_memory / (1024**3):.2f} GB \n"
            f"    gpu_memory_utilization = {self.gpu_memory_utilization:.2f}"
        )

        gc.collect() # 进行垃圾回收，释放未使用的内存
        torch.cuda.empty_cache() # 再次清理 CUDA 缓存
        
        return num_gpu_blocks # 返回可分配的 GPU 和 CPU 缓存块数量（此处 CPU 块数量为 0）
    

class KVCacheMemoryManager:
    """
        param:
        num_layers: int, 模型的 Transformer 层数
        num_kv_heads: int, 每层的 KV 头数
        head_dim: int, 每个头的维度
        gpu_num_blocks: int, 用户自行设置的最大可用 blocks(tokens), 如果设置该值， kv cache 内存管理器的最大可用内存-tokens 由该值决定。
        block_size: int, 每个 block 的大小，默认为 1
    """
    def __init__(self, num_layers, num_kv_heads, head_dim, gpu_num_blocks, block_size=1, dtype=torch.float16, device="cuda"):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gpu_num_blocks = gpu_num_blocks # 手动设定的给kv cache 内存管理分配的可用 blocks 数目:gpu_num_blocks
        self.block_size = block_size
        self.max_num_tokens = gpu_num_blocks * block_size

        self.dtype = dtype
        self.device = device
        self.can_use_mem_size = gpu_num_blocks # 可用的 kv cache tokens 数量

        # 定义 kv 内存位置索引和内存使用状态变量
        self.kv_mem_pos_indexs = torch.arange(0, self.max_num_tokens, dtype=torch.long, device="cuda")
        self.kv_mem_use_state = torch.zeros(self.max_num_tokens, dtype = torch.int32, device="cuda")

        # Initialize the gpu_kv_buffer
        self.init_kv_buffers(
            self.max_num_tokens,
            head_dim, num_kv_heads, num_layers, 
            dtype, device
        )

    def init_kv_buffers(self,    # 为每一层预先分配KV缓存的GPU内存， shape = [max_num_tokens, 2 * num_kv_heads, head_dim]  2 表示 kv 两个缓存d的拼接
        max_num_tokens,
        head_dim, num_kv_heads, num_layers,
        dtype,
        device: str="cuda"
    )-> list[torch.Tensor]:
        # kv cache shape: config.max_batch_size, config.max_seq_len, self.num_kv_heads, self.head_dim
        # max_num_tokens = max_num_blocks * self.block_size
        # TODO 修改 kv buffer 形状支持 PagedAttention
        self.gpu_kv_buffer = [
            torch.empty((max_num_tokens, 2 * num_kv_heads, head_dim), dtype=dtype, device=device) for _ in range(num_layers)
        ]
        logger.debug(f"gpu_kv_buffer per layer shape: {self.gpu_kv_buffer[0].shape}")
        
        
    #=========================判断是否可以分配连续的或者不连续的kv cache=================================
    @torch.no_grad()
    def alloc_kvcache(self, need_size):
        if need_size > self.can_use_mem_size:
            logger.warning(f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}")
            return None
        
        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        select_index = can_use_pos_index[0:need_size]
        self.add_ref(select_index)
        
        return select_index
    
    @torch.no_grad()
    def alloc_contiguous_kvcache(self, need_size):
        if need_size > self.can_use_mem_size:
            logger.warning(f"warn no enough contiguous cache need_size {need_size} left_size {self.can_use_mem_size}")
            return None

        # 获取未使用的内存块索引
        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        N = can_use_pos_index.numel()
        if N >= need_size:
            # 正确地计算 start_indexs 和 end_indexs. 
            # NOTE: 起始索引不能大于 N - need_size, 又因为 [: index] 切片操作是不包含 index 的, 所以需要将 N - need_size 加 1
            start_indexs = can_use_pos_index[:N - need_size + 1]
            # NOTE: can_use_pos_index[3:], 将获取索引为 3 到 9 的元素。
            end_indexs = can_use_pos_index[need_size - 1:]
            diff = end_indexs - start_indexs

            # 寻找连续的块，差值应为 need_size - 1
            contiguous_blocks = (diff == need_size - 1).nonzero(as_tuple=True)[0]

            if contiguous_blocks.numel() > 0:
                # 取出第一个连续块的起始索引
                # NOTE: contiguous_blocks[0] 是第一个连续块的索引
                # NOTE: start_indexs[contiguous_blocks[0]] 获取第一个连续块
                # 的起始索引
                # NOTE: end_indexs[contiguous_blocks[0]] 获取第一个连续块
                # 的结束索引
                # NOTE: start_indexs[contiguous_blocks[0]] 是连续块的起
                start_index = start_indexs[contiguous_blocks[0]].item()
                end_index = start_index + need_size
                select_index = self.kv_mem_pos_indexs[start_index:end_index]
                self.add_ref(select_index)
                return select_index, start_index, end_index

        return None
    
    @torch.no_grad()
    def alloc_kvcache_index(self, need_size):
        alloc_mem = self.alloc_contiguous_kvcache(need_size)
        if alloc_mem is not None:
            select_index, start_index, end_index = alloc_mem
            kv_cache = None
        else:
            select_index = self.alloc_kvcache(need_size)
            kv_cache = torch.empty(
                (need_size, self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        
        return select_index.to(torch.int32), kv_cache
    
    # 增加引用计数
    @torch.no_grad()
    def add_ref(self, token_index: torch.Tensor):
        state = self.kv_mem_use_state[token_index]
        has_used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size -= all_tokens - has_used_tokens
  
        self.kv_mem_use_state[token_index] += 1
        return
    
    # 减少引用计数
    @torch.no_grad()
    def release_ref(self, token_index: torch.Tensor):
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
    
    # 释放指定的kv cache 内存块索引
    @torch.no_grad()
    def free(self, free_index):
        free_index = free_index.long()
        self.release_ref(free_index)
        if self.can_use_mem_size == len(self.mem_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")
        return
    
    # 释放所有内存
    @torch.no_grad()
    def free_all(self,):
        self.can_use_mem_size = len(self.kv_mem_use_state)
        self.kv_mem_use_state[:] = 0

def indexs_convert(indexs: torch.tensor, batch_size: int):
    """
    prefill 阶段分配的kv cache 索引和 decode 阶段分配的索引合并在一起需要做变换
    TODO: 支持连续批处理开发时用上.
    """
    pass