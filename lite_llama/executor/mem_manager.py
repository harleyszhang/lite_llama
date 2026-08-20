import gc

import torch

from ..utils.logger import get_logger
from .executor_struct import AttentionInfo

logger = get_logger(__name__)


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return the size of one element of ``dtype`` in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


class MemoryProfiler:
    """Estimates how many KV-cache tokens fit in the remaining GPU memory.

    A short dummy forward pass measures the model's peak activation memory, and the
    leftover budget (``total * utilization - peak``) is divided by the per-token KV
    cache size. Replaces the old ``ComputeMaxAvailableBlocks`` + ``DummyInputGenerator``
    pair, which needed a separate config lookup table to build its dummy inputs; here
    the already-constructed model supplies every dimension.

    Args:
        num_layers: Decoder layer count.
        num_kv_heads: Key/value heads per layer.
        head_dim: Size of one attention head.
        gpu_memory_utilization: Fraction of total GPU memory the cache may occupy.
        dtype: KV-cache dtype.
        device: Torch device string.
        reserved_bytes: Extra budget to withhold from the cache (e.g. CUDA graph
            capture workspace), so a later allocation does not OOM.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        gpu_memory_utilization: float = 0.9,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        reserved_bytes: int = 0,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.device = device
        self.reserved_bytes = reserved_bytes

    def _kv_bytes_per_token(self) -> int:
        # Both K and V for every layer: factor of 2 for K+V.
        return self.num_kv_heads * self.head_dim * 2 * self.num_layers * get_dtype_size(self.dtype)

    def _run_dummy_forward(self, model, vocab_size: int, seq_len: int = 32) -> None:
        """Drive one prefill pass so peak activation memory is recorded."""
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=self.device)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        dummy = AttentionInfo()
        dummy.kv_buffer = [
            torch.empty(
                (seq_len, 2 * self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.num_layers)
        ]
        dummy.cur_select_index = torch.arange(seq_len, dtype=torch.int32, device=self.device)
        dummy.b_req_tokens_table = torch.arange(
            seq_len, dtype=torch.int32, device=self.device
        ).view(1, seq_len)
        dummy.b_start_loc = torch.tensor([0], dtype=torch.int32, device=self.device)
        dummy.b_req_idx = torch.tensor([0], dtype=torch.int32, device=self.device)
        dummy.b_seq_len = torch.tensor([seq_len], dtype=torch.int32, device=self.device)
        dummy.max_actual_seq_len = seq_len

        with torch.no_grad():
            model(input_ids, position_ids, dummy)

    def available_kv_blocks(self, model, vocab_size: int) -> int:
        """Return the number of KV-cache tokens that fit in free GPU memory.

        Falls back to a small fixed budget on CPU (where memory profiling APIs do
        not apply), which keeps unit tests runnable without a GPU.
        """
        if not torch.cuda.is_available() or self.device == "cpu":
            logger.warning("CUDA unavailable; using a minimal KV cache for CPU execution")
            return 4096

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _, total_gpu_memory = torch.cuda.mem_get_info()

        self._run_dummy_forward(model, vocab_size)
        torch.cuda.synchronize()

        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        torch.cuda.empty_cache()
        # Account for memory allocated outside the torch caching allocator.
        torch_current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        free_now, _ = torch.cuda.mem_get_info()
        non_torch = (total_gpu_memory - free_now) - torch_current
        if non_torch > 0:
            peak_memory += non_torch

        budget = total_gpu_memory * self.gpu_memory_utilization - peak_memory - self.reserved_bytes
        num_blocks = max(int(budget // self._kv_bytes_per_token()), 0)

        logger.info(
            "KV-cache profiling: total=%.2f GB peak=%.2f GB -> %d cache tokens (util=%.2f)",
            total_gpu_memory / 1024**3,
            peak_memory / 1024**3,
            num_blocks,
            self.gpu_memory_utilization,
        )

        gc.collect()
        torch.cuda.empty_cache()
        return num_blocks


class KVCacheMemoryManager:
    """
    param:
    num_layers: int, 模型的 Transformer 层数
    num_kv_heads: int, 每层的 KV 头数
    head_dim: int, 每个头的维度
    gpu_num_blocks: int, 用户自行设置的最大可用 blocks(tokens), 如果设置该值， kv cache 内存管理器的最大可用内存-tokens 由该值决定。
    block_size: int, 每个 block 的大小，默认为 1
    """

    def __init__(
        self,
        num_layers,
        num_kv_heads,
        head_dim,
        gpu_num_blocks,
        block_size=1,
        dtype=torch.float16,
        device="cuda",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gpu_num_blocks = (
            gpu_num_blocks  # 手动设定的给kv cache 内存管理分配的可用 blocks 数目:gpu_num_blocks
        )
        self.block_size = block_size
        self.max_num_tokens = gpu_num_blocks * block_size

        self.dtype = dtype
        self.device = device
        self.can_use_mem_size = gpu_num_blocks  # 可用的 kv cache tokens 数量

        # 定义 kv 内存位置索引和内存使用状态变量
        self.kv_mem_pos_indexs = torch.arange(
            0, self.max_num_tokens, dtype=torch.long, device=self.device
        )
        self.kv_mem_use_state = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=self.device
        )

        # Initialize the gpu_kv_buffer
        self.init_kv_buffers(self.max_num_tokens, head_dim, num_kv_heads, num_layers, dtype, device)

    def init_kv_buffers(
        self,  # 为每一层预先分配KV缓存的GPU内存， shape = [max_num_tokens, 2 * num_kv_heads, head_dim]  2 表示 kv 两个缓存d的拼接
        max_num_tokens,
        head_dim,
        num_kv_heads,
        num_layers,
        dtype,
        device: str = "cuda",
    ) -> None:
        # kv cache shape: config.max_batch_size, config.max_seq_len, self.num_kv_heads, self.head_dim
        # max_num_tokens = max_num_blocks * self.block_size
        # TODO 修改 kv buffer 形状支持 PagedAttention
        self.gpu_kv_buffer = [
            torch.empty((max_num_tokens, 2 * num_kv_heads, head_dim), dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        logger.debug(f"gpu_kv_buffer per layer shape: {self.gpu_kv_buffer[0].shape}")

    # =========================判断是否可以分配连续的或者不连续的kv cache=================================
    @torch.no_grad()
    def alloc_kvcache(self, need_size):
        if need_size > self.can_use_mem_size:
            logger.warning(
                f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}"
            )
            return None

        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        select_index = can_use_pos_index[0:need_size]
        self.add_ref(select_index)

        return select_index

    @torch.no_grad()
    def alloc_contiguous_kvcache(self, need_size):
        if need_size > self.can_use_mem_size:
            logger.warning(
                f"warn no enough contiguous cache need_size {need_size} left_size {self.can_use_mem_size}"
            )
            return None

        # 获取未使用的内存块索引
        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        N = can_use_pos_index.numel()
        if need_size <= N:
            # 正确地计算 start_indexs 和 end_indexs.
            # NOTE: 起始索引不能大于 N - need_size, 又因为 [: index] 切片操作是不包含 index 的, 所以需要将 N - need_size 加 1
            start_indexs = can_use_pos_index[: N - need_size + 1]
            # NOTE: can_use_pos_index[3:], 将获取索引为 3 到 9 的元素。
            end_indexs = can_use_pos_index[need_size - 1 :]
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
        """Reserve ``need_size`` cache rows, preferring a contiguous run."""
        alloc_mem = self.alloc_contiguous_kvcache(need_size)
        if alloc_mem is not None:
            select_index, _start_index, _end_index = alloc_mem
        else:
            select_index = self.alloc_kvcache(need_size)
        return select_index.to(torch.int32)

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

    # 释放指定的kv cache 内存块索引
    @torch.no_grad()
    def free(self, free_index):
        free_index = free_index.long()
        self.release_ref(free_index)
        if self.can_use_mem_size == len(self.kv_mem_use_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")
        return

    # 释放所有内存
    @torch.no_grad()
    def free_all(
        self,
    ):
        self.can_use_mem_size = len(self.kv_mem_use_state)
        self.kv_mem_use_state[:] = 0
