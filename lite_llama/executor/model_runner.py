"""Model runner: builds the model, sizes the KV cache, and runs each forward step.

Responsibilities:

* :meth:`ModelRunner.build` — parse ``config.json`` through
  :class:`~lite_llama.models.config.ModelConfig`, resolve the architecture via the
  :class:`~lite_llama.models.registry.ModelRegistry`, and hand both to a
  :class:`~lite_llama.executor.loader.ModelLoader`.
* :meth:`prefill_alloc_kv_cache` / :meth:`decode_alloc_kv_cache` — reserve cache rows.
* :meth:`forward` — dispatch to the model, passing multimodal inputs only when the
  resolved :class:`~lite_llama.models.registry.ModelSpec` says the model wants them.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..kernels import update_kv_index
from ..models.config import ModelConfig
from ..models.registry import ModelRegistry, ModelSpec
from ..utils.logger import get_logger
from .attention_metadata import AttentionMetadata
from .cuda_graph import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_SEQ_LEN_BUCKETS,
    CUDAGraphManager,
    estimate_capture_workspace,
)
from .kv_cache_manager import KVCacheManager, MemoryProfiler
from .loader import DefaultModelLoader, ModelLoader

logger = get_logger(__name__)


class ModelRunner:
    """Owns the model, the KV-cache memory manager, and the per-step attention state."""

    def __init__(
        self,
        checkpoints_dir: str,
        config: ModelConfig,
        spec: ModelSpec,
        model: nn.Module,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool = False,
    ) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.config = config
        self.spec = spec
        self.model = model
        self.device = device

        # ModelConfig already normalises the geometry across HF field names and
        # unwraps the nested text config of a vision-language checkpoint.
        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len

        if max_gpu_num_blocks is None:
            # When decode graphs will be captured later, withhold their workspace
            # from the KV budget — capture OOMs once the cache fills the card.
            reserved = (
                estimate_capture_workspace(self.max_seq_len)
                if use_cuda_graph and not spec.is_multimodal
                else 0
            )
            profiler = MemoryProfiler(
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                device=device,
                reserved_bytes=reserved,
            )
            max_gpu_num_blocks = profiler.available_kv_blocks(model, self.vocab_size)

        self.kv_cache_manager = KVCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            gpu_num_blocks=max_gpu_num_blocks,
            device=device,
        )
        self.max_request_num = max(1, max_gpu_num_blocks // self.max_seq_len)
        # Request -> KV-cache-row mapping; row i holds the cache rows of the
        # request with ``b_req_idx == i``. Written by ``_init_req_tokens_table``
        # at prefill and extended by ``update_kv_index`` at every decode step.
        self.b_req_tokens_table = torch.zeros(
            (self.max_request_num, self.max_seq_len), dtype=torch.int32, device=device
        )

        self.atten_info = AttentionMetadata()
        self.atten_info.kv_buffer = self.kv_cache_manager.gpu_kv_buffer
        self.atten_info.b_req_tokens_table = self.b_req_tokens_table

        # Populated by :meth:`enable_cuda_graph`; when non-None, :meth:`forward`
        # dispatches eligible decode steps to a captured graph.
        self._graph_manager: CUDAGraphManager | None = None

    # ------------------------------------------------------------------ build #
    @classmethod
    def build(
        cls,
        checkpoints_dir: str,
        max_seq_len: int,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool = False,
        loader: ModelLoader | None = None,
    ) -> ModelRunner:
        """Load config + weights and return a ready-to-run runner.

        Args:
            checkpoints_dir: HuggingFace checkpoint directory — ``config.json`` plus
                ``*.safetensors``, exactly as downloaded.
            max_seq_len: Upper bound on sequence length; also bounds the KV cache.
            max_gpu_num_blocks: Manual KV-cache size in tokens; profiled when ``None``.
            device: Torch device string.
            use_cuda_graph: Reserve capture workspace when profiling the KV cache,
                so a later :meth:`enable_cuda_graph` does not OOM.
            loader: Weight-loading strategy; defaults to
                :class:`~lite_llama.executor.loader.DefaultModelLoader`. Inject a
                fake in tests to build a runner without real weights.
        """
        config = ModelConfig.from_pretrained(checkpoints_dir, max_seq_len)
        spec = ModelRegistry.resolve(config.model_type)
        model = (loader or DefaultModelLoader()).load_model(
            config, spec.load_class(), checkpoints_dir, device
        )
        return cls(checkpoints_dir, config, spec, model, max_gpu_num_blocks, device, use_cuda_graph)

    # --------------------------------------------------------- kv allocation #
    def _init_req_tokens_table(
        self, b_req_idx, b_seq_len, alloc_index, max_prompt_len
    ) -> torch.Tensor:
        """Record which cache rows each prefill sequence occupies.

        The model flattens the padded ``[batch, max_prompt_len]`` token grid
        row-major, so sequence ``i``'s ``j``-th token lives at flattened index
        ``i * max_prompt_len + j`` and its K/V land in ``alloc_index`` at the
        same offset. The table must use that same padded layout — a packed
        (sum-of-lengths) layout would point sequence ``i`` at rows written by
        the tail of sequence ``i - 1``, silently corrupting every sequence
        after the first in a mixed-length batch.

        Returns:
            ``b_start_loc``: start offset of each sequence in the flattened batch.
        """
        b_seq_len_list = b_seq_len.cpu().tolist()
        b_req_idx_list = b_req_idx.cpu().tolist()
        b_start_loc = torch.zeros(len(b_seq_len_list), dtype=torch.int32, device=self.device)

        for i, seq_len in enumerate(b_seq_len_list):
            start = i * max_prompt_len
            b_start_loc[i] = start
            self.atten_info.b_req_tokens_table[b_req_idx_list[i], :seq_len] = alloc_index[
                start : start + seq_len
            ]
        return b_start_loc

    def prefill_alloc_kv_cache(self, max_prompt_len, actual_prompt_lens, b_req_idx) -> torch.Tensor:
        """Reserve cache rows for the whole prompt batch.

        Vision tokens need no special handling: the prompt already contains one
        token per vision patch (the processor expanded the ``<image>`` marker), so
        the reservation below covers them.
        """
        batch_size = len(actual_prompt_lens)
        self.atten_info.b_req_idx = b_req_idx
        self.atten_info.cur_select_index = self.kv_cache_manager.alloc_kvcache_index(
            max_prompt_len * batch_size
        )
        self.atten_info.b_seq_len = actual_prompt_lens
        self.atten_info.max_actual_seq_len = max_prompt_len
        self.atten_info.b_start_loc = self._init_req_tokens_table(
            b_req_idx, actual_prompt_lens, self.atten_info.cur_select_index, max_prompt_len
        )
        return self.atten_info.cur_select_index

    def decode_alloc_kv_cache(self, batch_size) -> torch.Tensor:
        """Reserve one cache row per sequence for the next decode step.

        ``update_kv_index`` writes at position ``b_seq_len - 1``, so ``b_seq_len``
        must be incremented *before* the kernel is launched. The legacy code did
        the opposite, which overwrote the mapping of the last prompt token and
        silently produced non-deterministic completions once a second request was
        served on the same runner.
        """
        self.atten_info.cur_select_index = self.kv_cache_manager.alloc_kvcache_index(batch_size)
        self.atten_info.b_seq_len += 1
        self.atten_info.max_actual_seq_len += 1
        update_kv_index(
            self.atten_info.b_req_tokens_table,
            self.atten_info.b_req_idx,
            self.atten_info.b_seq_len,
            self.atten_info.cur_select_index,
        )
        return self.atten_info.cur_select_index

    # ------------------------------------------------------------- inference #
    def enable_cuda_graph(
        self,
        batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
        seq_len_buckets: tuple[int, ...] = DEFAULT_SEQ_LEN_BUCKETS,
    ) -> None:
        """Capture decode graphs for the given ``(batch, seq_len_bucket)`` grid.

        Only the multi-modal-free text models are supported: the vision tower and
        DeepStack hook mutate control flow at every prefill step, and prefill is
        not graph-captured anyway.
        """
        if self.spec.is_multimodal:
            logger.warning("CUDA Graph is not supported for multi-modal models; running eager.")
            return
        if self._graph_manager is not None:
            return  # idempotent

        seq_len_buckets = tuple(b for b in seq_len_buckets if b <= self.max_seq_len)
        if not seq_len_buckets:
            logger.warning(
                "max_seq_len=%d is smaller than every requested bucket; skipping capture",
                self.max_seq_len,
            )
            return

        # b_req_tokens_table only has max_request_num rows; capturing a larger
        # batch would index past the table and corrupt the CUDA context.
        batch_sizes = tuple(b for b in batch_sizes if b <= self.max_request_num)
        if not batch_sizes:
            logger.warning(
                "max_request_num=%d is smaller than every requested batch size; skipping capture",
                self.max_request_num,
            )
            return

        logger.info(
            "Capturing CUDA graphs for batch_sizes=%s seq_len_buckets=%s",
            batch_sizes,
            seq_len_buckets,
        )
        manager = CUDAGraphManager(
            self.model,
            kv_buffer=self.kv_cache_manager.gpu_kv_buffer,
            b_req_tokens_table=self.b_req_tokens_table,
            batch_sizes=batch_sizes,
            seq_len_buckets=seq_len_buckets,
            device=self.device,
        )
        try:
            manager.capture_all()
        except torch.cuda.OutOfMemoryError:
            # A failed capture may leave a half-open graph; dropping the manager
            # is safe because replay state is only installed on success.
            logger.warning("CUDA graph capture ran out of memory; falling back to eager decode")
            return
        self._graph_manager = manager

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        multi_modal_inputs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run one model step.

        Dispatches to a captured CUDA graph when the current step is a decode
        (``seq_len == 1``) whose ``(batch_size, max_actual_seq_len)`` matches one
        of the captured buckets; otherwise runs eager.
        """
        if self.spec.is_multimodal:
            return self.model(input_ids, position_ids, self.atten_info, multi_modal_inputs)

        if self._graph_manager is not None:
            replayed = self._graph_manager.try_replay(input_ids, position_ids, self.atten_info)
            if replayed is not None:
                return replayed

        return self.model(input_ids, position_ids, self.atten_info)
