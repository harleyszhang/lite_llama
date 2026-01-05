import torch
import json
import gc
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Union

from ..utils.dummy_data import DummyInputGenerator
from .executor_struct import AttentionInfo, CONFIG_CLASS_MAP
from ..utils.logger import get_logger

logger = get_logger(__name__)

def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    return torch.tensor([], dtype=dtype).element_size()

class ComputeMaxAvailableBlocks:
    """
    Executes a dummy forward pass to profile memory usage and calculates 
    maximum possible KV blocks.
    """
    def __init__(
        self, 
        num_layers: int, 
        hidden_size: int, 
        num_heads: int, 
        num_kv_heads: int, 
        head_dim: int, 
        gpu_memory_utilization: float = 0.9, 
        block_size: int = 1, 
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.gpu_memory_utilization = gpu_memory_utilization
        self.block_size = block_size 
        self.dtype = dtype
        self.device = device
        self.dtype_size = get_dtype_size(dtype)
        
    def compute_cache_block_size_bytes(self) -> int:
        """Calculate bytes required for one KV block across all layers."""
        # KV Cache shape per layer: [2, num_kv_heads, head_dim] * dtype_size
        # The '2' stands for K and V.
        kv_cache_token_bytes_per_layer = (self.num_kv_heads * self.head_dim) * 2 * self.dtype_size
        transformer_kv_cache_token_bytes = kv_cache_token_bytes_per_layer * self.num_layers
        transformer_kv_cache_blocks_bytes = transformer_kv_cache_token_bytes * self.block_size
        return transformer_kv_cache_blocks_bytes

    def _infer_vocab_size(self, model_path: str | None, model_type: str | None, 
                         model_config, llm_config) -> int:
        """Helper to infer vocab size safely."""
        # 1. Try llm_config / model_config objects
        for cfg in [llm_config, model_config]:
            if cfg is None: continue
            if hasattr(cfg, "vocab_size"): return int(cfg.vocab_size)
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "vocab_size"):
                return int(cfg.text_config.vocab_size)

        # 2. Try loading from config.json manually
        if model_path:
            params_path = Path(model_path) / "config.json"
            if params_path.exists():
                try:
                    with open(params_path, "r") as f:
                        params = json.load(f)
                    # Check direct key
                    if "vocab_size" in params:
                        return int(params["vocab_size"])
                    
                    # Try via Config Class
                    m_type = (model_type or params.get("model_type", "")).lower()
                    cfg_cls = CONFIG_CLASS_MAP.get(m_type)
                    if cfg_cls:
                        cfg_obj = cfg_cls.from_dict(params)
                        if hasattr(cfg_obj, "vocab_size"): return int(cfg_obj.vocab_size)
                        if hasattr(cfg_obj, "text_config"): return int(cfg_obj.text_config.vocab_size)
                except Exception as e:
                    logger.warning(f"Failed to infer vocab_size from json: {e}")
        
        # 3. Fallback
        logger.warning("Could not infer vocab_size, defaulting to 32000 (Llama default)")
        return 32000

    def compute_num_available_blocks(
        self,
        model,
        model_path: str | None = None,
        model_type: str | None = None,
        model_config=None,
        llm_config=None,
    ) -> int:
        
        # Cleanup before profiling
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()

        # 1. Infer Vocab Size
        vocab_size = self._infer_vocab_size(model_path, model_type, model_config, llm_config)

        # 2. Prepare Dummy Inputs
        batch_size = 1
        seq_len = 32 # Short sequence enough to trigger lazy loading
        
        dummy_input = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=self.device, dtype=torch.long
        )
        dummy_position_ids = (
            torch.arange(0, seq_len, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        dummy_atten_info = AttentionInfo()
        # Allocate small dummy buffer for the forward pass
        dummy_atten_info.kv_buffer = [
            torch.empty(
                (seq_len, 2 * self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.num_layers)
        ]
        dummy_atten_info.cur_select_index = torch.arange(seq_len, dtype=torch.int32, device=self.device)
        dummy_atten_info.b_start_loc = torch.tensor([0], dtype=torch.int32, device=self.device)
        dummy_atten_info.b_seq_len = torch.tensor([seq_len], dtype=torch.int32, device=self.device)
        dummy_atten_info.max_actual_seq_len = seq_len

        # 3. Execute Dummy Forward
        try:
            # Use inference_mode for better performance simulation than no_grad
            with torch.inference_mode():
                if (model_type or "").lower() == "llava" and hasattr(model, "language_model"):
                    model.language_model(dummy_input, dummy_position_ids, dummy_atten_info)
                else:
                    model(dummy_input, dummy_position_ids, dummy_atten_info)
        except RuntimeError as e:
            if "sentencepiece" in str(e).lower():
                logger.error("🚨 Error: 'sentencepiece' library is missing. Please run `pip install sentencepiece`.")
                # We can't proceed accurately if the model crashed, but we can try to estimate purely based on weights
                # For now, re-raise to stop execution or handle gracefully.
                raise e
            else:
                logger.warning(f"Dummy forward pass failed: {e}. Memory estimation might be inaccurate.")
        except Exception as e:
             logger.warning(f"Dummy forward pass failed with unknown error: {e}")

        torch.cuda.synchronize()
        
        # 4. Calculate Memory
        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        
        # Clean up dummy tensors
        del dummy_input, dummy_position_ids, dummy_atten_info
        torch.cuda.empty_cache()
        
        # Check for non-torch allocations (fragmentation or drivers)
        current_allocated = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        info_free, info_total = torch.cuda.mem_get_info()
        real_used = info_total - info_free
        non_torch_allocations = real_used - current_allocated
        
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations

        available_bytes = (total_gpu_memory * self.gpu_memory_utilization) - peak_memory
        cache_block_size = self.compute_cache_block_size_bytes()
        
        if available_bytes < 0:
            logger.error("❌ Not enough GPU memory to load model weights!")
            return 0

        num_gpu_blocks = int(available_bytes // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)

        logger.info(
            f"Memory Profiling:\n"
            f"  Total GPU Mem: {total_gpu_memory / (1024**3):.2f} GB\n"
            f"  Peak Torch Mem: {peak_memory / (1024**3):.2f} GB\n"
            f"  Available for KV: {available_bytes / (1024**3):.2f} GB\n"
            f"  Block Size: {cache_block_size} bytes\n"
            f"  Max Blocks: {num_gpu_blocks}"
        )

        gc.collect()
        torch.cuda.empty_cache()
        
        return num_gpu_blocks


class KVCacheMemoryManager:
    """
    Manages the allocation of KV Cache blocks.
    Optimized with a stack-based allocator for O(1) non-contiguous allocation.
    """
    def __init__(self, num_layers, num_kv_heads, head_dim, gpu_num_blocks, block_size=1, dtype=torch.float16, device="cuda"):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gpu_num_blocks = gpu_num_blocks
        self.block_size = block_size
        self.max_num_tokens = gpu_num_blocks * block_size

        self.dtype = dtype
        self.device = device
        
        # Tracking usage
        self.can_use_mem_size = gpu_num_blocks 
        
        # Reference counting state: 0 = free, >0 = used (shared count)
        self.kv_mem_use_state = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=device)
        
        # === Optimization: Free Block Stack ===
        # Instead of searching for zeros, we keep a stack of free indices.
        # This makes allocation O(1).
        # Initialize with all indices [0, 1, ..., max-1] reversed (so we pop from 0 upwards roughly)
        self.free_indices_stack = torch.arange(self.max_num_tokens - 1, -1, -1, dtype=torch.long, device=device)
        self.free_stack_top = self.max_num_tokens # Pointer to the top of the stack (size)

        self.gpu_kv_buffer: List[torch.Tensor] | None = None
        self.init_kv_buffers(self.max_num_tokens, head_dim, num_kv_heads, num_layers, dtype, device)

    def init_kv_buffers(self, max_num_tokens, head_dim, num_kv_heads, num_layers, dtype, device):
        # Shape: [max_num_tokens, 2 * num_kv_heads, head_dim]
        # Using 2 * num_kv_heads allows storing K and V contiguously in the last dimension or head dim
        try:
            self.gpu_kv_buffer = [
                torch.empty((max_num_tokens, 2 * num_kv_heads, head_dim), dtype=dtype, device=device) 
                for _ in range(num_layers)
            ]
            logger.info(f"Initialized KV Cache: {max_num_tokens} tokens, Shape: {self.gpu_kv_buffer[0].shape}")
        except torch.cuda.OutOfMemoryError:
            logger.error("OOM when allocating KV Buffer. Try reducing gpu_memory_utilization.")
            raise

    @torch.inference_mode()
    def alloc_kvcache(self, need_size: int) -> Optional[torch.Tensor]:
        """
        Allocate non-contiguous memory blocks.
        Time Complexity: O(1) (Amortized) using stack pop.
        """
        if need_size > self.can_use_mem_size:
            logger.warning(f"KV Cache OOM: Need {need_size}, Left {self.can_use_mem_size}")
            return None
        
        # Pop from the free stack
        # self.free_indices_stack is pre-allocated. We slice the top `need_size`.
        current_top = self.free_stack_top
        new_top = current_top - need_size
        
        if new_top < 0:
            # Should not happen if can_use_mem_size check passes, but for safety
            logger.error("Inconsistency in KV Memory Manager state.")
            return None

        # Get indices
        select_index = self.free_indices_stack[new_top:current_top]
        self.free_stack_top = new_top
        
        # Update Ref Count & Available Size
        self.add_ref(select_index, update_stack=False) # Don't update stack inside add_ref, we handled it
        
        return select_index

    @torch.inference_mode()
    def alloc_contiguous_kvcache(self, need_size: int) -> Optional[Tuple[torch.Tensor, int, int]]:
        """
        Allocate contiguous memory.
        Note: This is expensive (O(N)) because we must scan for holes.
        Use alloc_kvcache (PagedAttention style) whenever possible.
        """
        if need_size > self.can_use_mem_size:
            return None

        # Fallback to scanning kv_mem_use_state because the stack is random order
        # Finding contiguous zeros
        zero_indices = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        
        if zero_indices.numel() < need_size:
            return None

        # Vectorized check for contiguous blocks
        # We look for a sequence where index[i+need-1] - index[i] == need-1
        # Optimization: Only scan if strictly needed
        
        # Prepare start and end candidates
        start_candidates = zero_indices[: -need_size + 1]
        end_candidates = zero_indices[need_size - 1 :]
        
        diff = end_candidates - start_candidates
        
        # Check where diff is exactly (need_size - 1)
        valid_starts = (diff == (need_size - 1)).nonzero(as_tuple=True)[0]
        
        if valid_starts.numel() > 0:
            # Pick the first valid block
            idx_in_zeros = valid_starts[0].item()
            start_index = start_candidates[idx_in_zeros].item()
            end_index = start_index + need_size
            
            select_index = self.kv_mem_pos_indexs[start_index:end_index] # Assuming this attr exists or create it
            
            # Critical: We must remove these indices from the free_stack to keep sync
            # This is slow, hence contiguous alloc is slow. 
            # We reconstruct the stack mask-based or lazily. 
            # For now, simplistic approach:
            self.add_ref(select_index, update_stack=True) 
            
            return select_index, start_index, end_index

        return None
    
    @property
    def kv_mem_pos_indexs(self):
        # Lazy creation or stored
        if not hasattr(self, "_kv_pos_idx"):
            self._kv_pos_idx = torch.arange(0, self.max_num_tokens, dtype=torch.long, device=self.device)
        return self._kv_pos_idx

    @torch.inference_mode()
    def alloc_kvcache_index(self, need_size: int):
        """Wrapper to try contiguous first, then paged."""
        # need_size==1 时 contiguous 扫描没有意义，且切片逻辑会产生空张量导致 shape mismatch
        if need_size <= 1:
            select_index = self.alloc_kvcache(need_size)
            if select_index is None:
                raise RuntimeError("KV Cache OOM")
            kv_cache = torch.empty(
                (need_size, self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            return select_index.to(torch.int32), kv_cache

        alloc_mem = self.alloc_contiguous_kvcache(need_size)
        if alloc_mem is not None:
            select_index, _, _ = alloc_mem
            kv_cache = None # Not returning tensor buffer, just indices
        else:
            select_index = self.alloc_kvcache(need_size)
            if select_index is None:
                raise RuntimeError("KV Cache OOM")
            # Legacy return format support
            kv_cache = torch.empty(
                (need_size, self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        
        return select_index.to(torch.int32), kv_cache
    
    @torch.inference_mode()
    def add_ref(self, token_index: torch.Tensor, update_stack: bool = False):
        """
        Increase reference count. 
        If update_stack is True, it means we manually picked indices (like contiguous) 
        and need to remove them from the free stack.
        """
        # 1. Update Reference Count
        self.kv_mem_use_state[token_index] += 1
        
        # 2. Update Available Size Calculation
        # Only decrease available size if it was previously 0 (free)
        # Note: In PagedAttention with sharing, adding ref to already used block doesn't consume *new* physical memory
        # But here logic implies 'can_use_mem_size' tracks raw free slots.
        # Actually, standard logic: can_use_mem_size is just free slots.
        # If token_index was already used, it's not in free slots.
        
        # Assuming token_index passed here are strictly newly allocated from free pool
        # logic in alloc_kvcache guarantees they were free.
        self.can_use_mem_size -= len(token_index)
        
        if update_stack:
            # Expensive: Remove specific values from stack
            # Usually only happens for contiguous alloc
            mask = torch.isin(self.free_indices_stack[:self.free_stack_top], token_index, invert=True)
            remaining = self.free_indices_stack[:self.free_stack_top][mask]
            self.free_indices_stack[:len(remaining)] = remaining
            self.free_stack_top = len(remaining)

    @torch.inference_mode()
    def release_ref(self, token_index: torch.Tensor):
        """
        Decrease reference count. If count hits 0, return to free stack.
        """
        token_index, counts = token_index.unique(return_counts=True)
        
        # Decrease counts
        self.kv_mem_use_state[token_index] -= counts.int()
        
        # Find which ones became free (count == 0)
        # Note: We must clamp to 0 to avoid negative (bug safety)
        self.kv_mem_use_state[token_index] = torch.clamp(self.kv_mem_use_state[token_index], min=0)
        
        freed_mask = (self.kv_mem_use_state[token_index] == 0)
        freed_indices = token_index[freed_mask]
        
        num_freed = freed_indices.numel()
        
        if num_freed > 0:
            # Push back to stack
            current_top = self.free_stack_top
            new_top = current_top + num_freed
            
            if new_top > self.max_num_tokens:
                logger.error("Memory Manager Error: Free stack overflow.")
                return

            self.free_indices_stack[current_top:new_top] = freed_indices
            self.free_stack_top = new_top
            self.can_use_mem_size += num_freed

    def free_all(self):
        self.can_use_mem_size = self.max_num_tokens
        self.kv_mem_use_state.zero_()
        # Reset Stack
        self.free_indices_stack = torch.arange(self.max_num_tokens - 1, -1, -1, dtype=torch.long, device=self.device)
        self.free_stack_top = self.max_num_tokens