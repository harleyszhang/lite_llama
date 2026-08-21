"""Paged KV cache: block allocation with reference counting, plus sizing.

:class:`MemoryProfiler` measures free device memory to decide how many cache blocks
fit; :class:`KVCacheManager` hands out and ref-counts block indices so prefill and
decode reserve rows and release them when a sequence ends.

Usage:
    idx = kv.alloc_kvcache_index(need_size); kv.release_ref(idx)
"""

import gc

import torch

from ..utils.logger import get_logger
from .attention_metadata import AttentionMetadata

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

        dummy = AttentionMetadata()
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


class KVCacheManager:
    """Owns the paged KV buffers and hands out cache rows by reference count.

    One row per token (``block_size=1``), so "block" and "token" are the same unit
    here. ``kv_mem_use_state[i]`` is the reference count of row ``i``: a row is free
    at zero, and :attr:`can_use_mem_size` tracks how many are.

    :meth:`alloc_kvcache_index` is the entry point and picks between three
    strategies, cheapest first: a bump cursor while the cache is append-only, then a
    contiguous-run search, then any scattered free rows.

    Args:
        num_layers: Decoder layer count.
        num_kv_heads: Key/value heads per layer.
        head_dim: Size of one attention head.
        gpu_num_blocks: Cache capacity in blocks, either profiled by
            :class:`MemoryProfiler` or set by the caller.
        block_size: Tokens per block; only 1 is implemented.
        dtype: KV-cache dtype.
        device: Torch device string.
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
        self.gpu_num_blocks = gpu_num_blocks
        self.block_size = block_size
        self.max_num_tokens = gpu_num_blocks * block_size

        self.dtype = dtype
        self.device = device
        self.can_use_mem_size = gpu_num_blocks  # rows currently free

        # Row indices to hand out, and the per-row reference count.
        self.kv_mem_pos_indexs = torch.arange(
            0, self.max_num_tokens, dtype=torch.long, device=self.device
        )
        self.kv_mem_use_state = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=self.device
        )
        # Callers need int32 row indices; keeping a pre-cast copy lets the
        # bump-allocator fast path return a view instead of casting per step.
        self.kv_mem_pos_indexs_int32 = self.kv_mem_pos_indexs.to(torch.int32)
        # Cursor for the append-only fast path, and whether it is still exact
        # (invalidated by any partial free, restored by ``free_all``).
        self._bump_cursor = 0
        self._bump_is_exact = True

        # Initialize the gpu_kv_buffer
        self.init_kv_buffers(self.max_num_tokens, head_dim, num_kv_heads, num_layers, dtype, device)

    def init_kv_buffers(
        self,
        max_num_tokens,
        head_dim,
        num_kv_heads,
        num_layers,
        dtype,
        device: str = "cuda",
    ) -> None:
        """Pre-allocate one KV tensor per layer, ``[max_num_tokens, 2 * kv_heads, head_dim]``.

        K and V share the tensor along dim 1 — K heads first, then V heads — so a
        decode step writes both with one kernel launch.
        """
        # TODO: reshape into [blocks, block_size, ...] to support PagedAttention.
        self.gpu_kv_buffer = [
            torch.empty((max_num_tokens, 2 * num_kv_heads, head_dim), dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        logger.debug(f"gpu_kv_buffer per layer shape: {self.gpu_kv_buffer[0].shape}")

    @torch.no_grad()
    def alloc_kvcache(self, need_size):
        """Reserve ``need_size`` free rows, wherever they are. Returns ``None`` if short."""
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
        """Reserve ``need_size`` *consecutive* free rows, or ``None`` if there is no such run.

        Returns:
            ``(select_index, start_index, end_index)``, or ``None``.
        """
        if need_size > self.can_use_mem_size:
            logger.warning(
                f"warn no enough contiguous cache need_size {need_size} left_size {self.can_use_mem_size}"
            )
            return None

        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        N = can_use_pos_index.numel()
        if need_size <= N:
            # Two views of the free list offset by need_size - 1, so start_indexs[j]
            # and end_indexs[j] are the ends of a candidate window. The last valid
            # start is at N - need_size, and slicing excludes the stop, hence the + 1.
            start_indexs = can_use_pos_index[: N - need_size + 1]
            end_indexs = can_use_pos_index[need_size - 1 :]
            # A window holds consecutive rows exactly when its two ends differ by
            # need_size - 1; anything larger means a used row sits in between.
            contiguous_blocks = (end_indexs - start_indexs == need_size - 1).nonzero(as_tuple=True)[
                0
            ]

            if contiguous_blocks.numel() > 0:
                start_index = start_indexs[contiguous_blocks[0]].item()  # first run wins
                end_index = start_index + need_size
                select_index = self.kv_mem_pos_indexs[start_index:end_index]
                self.add_ref(select_index)
                return select_index, start_index, end_index

        return None

    @torch.no_grad()
    def alloc_kvcache_index(self, need_size):
        """Reserve ``need_size`` cache rows, preferring a contiguous run.

        A decode step reserves one row per sequence, so this runs on the hot
        path. The search in :meth:`alloc_contiguous_kvcache` costs a
        ``nonzero`` over the whole cache plus two ``.item()`` reads, i.e. three
        device synchronisations per decode step, which stalls the launch
        pipeline far longer than the allocation itself.

        While the cache is only ever appended to — the state every
        ``generate()`` call starts from, because it opens with
        :meth:`free_all` — the answer that search returns is exactly the next
        ``need_size`` rows after the previous allocation. The bump cursor below
        returns those rows directly, with no device reads at all, and any
        partial free falls back to the general search.
        """
        if self._bump_is_exact and self._bump_cursor + need_size <= self.max_num_tokens:
            start = self._bump_cursor
            select_index = self.kv_mem_pos_indexs_int32[start : start + need_size]
            self.kv_mem_use_state[start : start + need_size] += 1
            self._bump_cursor += need_size
            self.can_use_mem_size -= need_size
            return select_index

        alloc_mem = self.alloc_contiguous_kvcache(need_size)
        if alloc_mem is not None:
            select_index, _start_index, _end_index = alloc_mem
        else:
            select_index = self.alloc_kvcache(need_size)
        return select_index.to(torch.int32)

    @torch.no_grad()
    def add_ref(self, token_index: torch.Tensor):
        """Increment the reference count of the given rows.

        Only rows that were free reduce :attr:`can_use_mem_size`; taking a second
        reference on an already-held row costs no capacity.
        """
        state = self.kv_mem_use_state[token_index]
        has_used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size -= all_tokens - has_used_tokens

        self.kv_mem_use_state[token_index] += 1
        return

    @torch.no_grad()
    def release_ref(self, token_index: torch.Tensor):
        """Decrement the reference count of the given rows, freeing those that reach zero.

        ``token_index`` may name a row more than once — the engine releases prefill
        and every decode step in one concatenated tensor — so counts are collapsed
        with ``unique`` first and subtracted in one go.
        """
        # Freeing rows leaves holes, so the append-only cursor no longer
        # describes the free list; fall back to searching until ``free_all``.
        self._bump_is_exact = False
        token_index, counts = token_index.unique(return_counts=True)
        self.kv_mem_use_state[token_index] -= counts
        state = self.kv_mem_use_state[token_index]
        used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size += all_tokens - used_tokens
        return

    @torch.no_grad()
    def free(self, free_index):
        """Release rows by index, logging when that empties the cache."""
        free_index = free_index.long()
        self.release_ref(free_index)
        if self.can_use_mem_size == len(self.kv_mem_use_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")
        return

    @torch.no_grad()
    def free_all(
        self,
    ):
        """Drop every reference at once, as each ``generate()`` call starts by doing."""
        self.can_use_mem_size = len(self.kv_mem_use_state)
        self.kv_mem_use_state[:] = 0
        # The cache is empty again, so appending from row 0 is exact once more.
        self._bump_cursor = 0
        self._bump_is_exact = True
