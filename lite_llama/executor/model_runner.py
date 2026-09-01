"""Model runner: builds the model, sizes the KV cache, runs each forward step.

``ModelRunner.build`` is the one-call constructor (config, loader, KV
blocks); the instance then owns the KV buffers, the request-to-token
table and ``forward`` for both phases.

Usage:
    runner = ModelRunner.build(checkpoints_dir, max_seq_len)
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from ..distributed.parallel_state import (
    all_ranks_agree,
    all_reduce_min,
    divide,
    get_tp_world_size,
)
from ..kernels import update_kv_index
from ..models.config import ModelConfig
from ..models.registry import ModelRegistry, ModelSpec
from ..utils.logger import get_logger
from .attention_metadata import AttentionMetadata
from .cuda_graph import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_SEQ_LEN_BUCKETS,
    TP_GRAPH_PARITY_ATOL,
    CUDAGraphManager,
    estimate_capture_workspace,
)
from .kv_cache_manager import KVCacheManager, MemoryProfiler
from .loader import DefaultModelLoader, ModelLoader
from .slot_batch import SlotBatch

logger = get_logger(__name__)

#: Set to ``0`` to keep the pre-TP-graph behaviour: eager decode whenever
#: ``tp_size > 1``. A kill-switch rather than a config field because the failure it
#: guards against is a hang, and somebody meeting one needs a way out that does not
#: involve editing code.
_TP_GRAPH_ENV = "LITE_LLAMA_TP_CUDA_GRAPH"


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
        cuda_graph_lazy: bool = False,
    ) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.config = config
        self.spec = spec
        self.model = model
        self.device = device

        # ModelConfig already normalises the geometry across HF field names and
        # unwraps the nested text config of a vision-language checkpoint.
        self.num_layers = config.num_layers
        if config.is_mla:
            # MLA caches one latent vector per token: there is no head axis to
            # shard, so every TP rank holds the full row (as vLLM does) — and the
            # occupancy report below is honest about it.
            self.kv_row = (1, config.kv_lora_rank + config.qk_rope_head_dim)
        else:
            # Attention heads are dealt out across tensor-parallel ranks, so this
            # rank caches only the K/V of the heads it owns.
            kv_heads = divide(config.num_kv_heads, get_tp_world_size(), "key/value heads")
            self.kv_row = (2 * kv_heads, config.head_dim)
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        # Element type of the paged KV cache: fp16 verbatim, or uint8 holding
        # e4m3 bytes when the deployment asked for an fp8 cache.
        kv_dtype = config.kv_cache_torch_dtype

        if max_gpu_num_blocks is None:
            # When decode graphs will be captured later, withhold their workspace
            # from the KV budget — capture OOMs once the cache fills the card.
            # Lazy capture (O13) withholds only the seed pair; shapes captured
            # on demand take their workspace from what is free at that moment.
            reserved = (
                estimate_capture_workspace(self.max_seq_len, lazy=cuda_graph_lazy)
                if use_cuda_graph
                else 0
            )
            profiler = MemoryProfiler(
                num_layers=self.num_layers,
                kv_row=self.kv_row,
                dtype=kv_dtype,
                device=device,
                reserved_bytes=reserved,
            )
            # Every rank must reach the same answer: the cache row a rank hands
            # out is derived from its own capacity, and two ranks with different
            # capacities would write the same token to different rows.
            max_gpu_num_blocks = all_reduce_min(
                profiler.available_kv_blocks(model, self.vocab_size)
            )

        self.kv_cache_manager = KVCacheManager(
            num_layers=self.num_layers,
            kv_row=self.kv_row,
            gpu_num_blocks=max_gpu_num_blocks,
            dtype=kv_dtype,
            device=device,
        )
        # Rows in the block table, i.e. the concurrency ceiling. Paging decoupled
        # this from the cache size: a slot's rows are pages it holds, not a
        # reserved ``max_seq_len`` stripe, so the table is sized by how many
        # requests may be in flight. One row above the largest captured batch
        # covers every graph plus the filler slot, and it has to be decided here
        # -- a capture bakes this tensor's pointer, so the table cannot be
        # reallocated later.
        self.max_request_num = max(DEFAULT_BATCH_SIZES) + 1
        # Request -> KV-cache-row mapping; row i holds the cache rows of the
        # request with ``b_req_idx == i``. Written by ``_init_req_tokens_table``
        # at prefill and extended by ``update_kv_index`` at every decode step;
        # under continuous batching it is a block table the scheduler fills in.
        self.b_req_tokens_table = torch.zeros(
            (self.max_request_num, self.max_seq_len), dtype=torch.int32, device=device
        )

        self.atten_info = AttentionMetadata()
        self.atten_info.kv_buffer = self.kv_cache_manager.gpu_kv_buffer
        self.atten_info.b_req_tokens_table = self.b_req_tokens_table

        # Populated by :meth:`enable_cuda_graph`; when non-None, :meth:`forward`
        # dispatches eligible decode steps to a captured graph.
        self._graph_manager: CUDAGraphManager | None = None
        # Populated by :meth:`enable_slot_kv_cache` for the continuous-batching
        # path; the one-shot batch path never builds it.
        self._slot_batch: SlotBatch | None = None

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
        quantization: str | None = None,
        kv_cache_dtype: str = "auto",
        cuda_graph_lazy: bool = False,
        hf_overrides: dict[str, object] | None = None,
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
            quantization: Weight quantisation to apply to an fp16 checkpoint
                (``"int8"``); fp8 checkpoints carry their own and ignore this.
            kv_cache_dtype: KV-cache element type — ``"auto"`` (fp16) or an fp8
                spelling (``"fp8"`` / ``"fp8_e4m3"``), which halves the cache
                footprint so twice as many tokens fit the same budget.
            cuda_graph_lazy: Withhold only the lazy seed pair's workspace (O13)
                instead of the whole grid's — pair with
                :meth:`enable_cuda_graph`'s ``lazy`` flag or on-demand captures
                fight the cache for workspace that was never withheld.
            hf_overrides: Fields applied over the checkpoint's ``config.json``
                (vLLM ``--hf-overrides`` semantics), e.g.
                ``{"num_hidden_layers": 1}`` to trim the stack while still
                loading and running through every production path.
        """
        config = ModelConfig.from_pretrained(
            checkpoints_dir, max_seq_len, kv_cache_dtype, hf_overrides=hf_overrides
        )
        spec = ModelRegistry.resolve(config.model_type)
        model = (loader or DefaultModelLoader()).load_model(
            config, spec.load_class(), checkpoints_dir, device, quantization
        )
        return cls(
            checkpoints_dir, config, spec, model, max_gpu_num_blocks, device, use_cuda_graph,
            cuda_graph_lazy=cuda_graph_lazy,
        )

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

        Written with one masked gather + one ``index_put_`` rather than a Python
        loop over sequences: the loop cost a host round-trip per sequence
        (``.cpu().tolist()`` on both index tensors) plus a kernel launch per row.

        Returns:
            ``b_start_loc``: start offset of each sequence in the flattened batch.
        """
        n = b_seq_len.shape[0]
        rows = torch.arange(n, device=self.device)
        cols = torch.arange(max_prompt_len, device=self.device)
        # [n, max_prompt_len] — True where the padded column is a real token.
        valid = cols.unsqueeze(0) < b_seq_len.view(-1, 1)
        # Flattened source offsets into the padded allocation, row-major.
        src = alloc_index[(rows * max_prompt_len).unsqueeze(1) + cols.unsqueeze(0)]

        table = self.atten_info.b_req_tokens_table
        table.index_put_(
            (
                b_req_idx.view(-1, 1).expand(n, max_prompt_len)[valid],
                cols.unsqueeze(0).expand(n, max_prompt_len)[valid],
            ),
            src[valid],
        )
        return (rows * max_prompt_len).to(torch.int32)

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
        self.atten_info.is_prefill = True
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
        self.atten_info.is_prefill = False
        update_kv_index(
            self.atten_info.b_req_tokens_table,
            self.atten_info.b_req_idx,
            self.atten_info.b_seq_len,
            self.atten_info.cur_select_index,
        )
        return self.atten_info.cur_select_index

    # ------------------------------------------------------------- inference #
    def enable_slot_kv_cache(self) -> SlotBatch:
        """Switch the KV cache to the paged block-table layout continuous batching needs.

        Idempotent, and mutually exclusive with the one-shot batch path: the
        returned :class:`~lite_llama.executor.slot_batch.SlotBatch` reads its rows
        out of the block table the scheduler fills, while
        ``prefill_alloc_kv_cache`` and ``decode_alloc_kv_cache`` allocate rows
        themselves and would hand out pages the scheduler already owns.

        Call it *after* :meth:`enable_cuda_graph` so batch padding can see the
        captured grid.
        """
        if self._slot_batch is None:
            self._slot_batch = SlotBatch(self)
        return self._slot_batch

    def graph_batch_size(self, batch_size: int) -> int:
        """Batch size to submit so a decode step lands on a captured graph.

        Returns ``batch_size`` unchanged when graphs are off or the batch is
        bigger than anything captured, in which case the step runs eager.
        """
        if self._graph_manager is None:
            return batch_size
        return self._graph_manager.pad_to(batch_size) or batch_size

    def enable_cuda_graph(
        self,
        batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
        seq_len_buckets: tuple[int, ...] = DEFAULT_SEQ_LEN_BUCKETS,
        *,
        lazy: bool = False,
    ) -> None:
        """Capture decode graphs for the given ``(batch, seq_len_bucket)`` grid.

        Multimodal models are supported: a capture only ever replays a decode
        step, and by then the vision tokens are ordinary KV-cache rows — the
        vision tower and DeepStack hooks run during prefill, which stays eager.

        ``lazy`` (O13) captures only a seed pair now and lets
        :meth:`CUDAGraphManager.try_replay` capture the remaining shapes the
        first time a step needs them — the cold start stops paying for shapes
        the workload may never produce. The KV profiler must have reserved
        with ``cuda_graph_lazy`` too, or on-demand captures fight the cache for
        the workspace that eager startup would have withheld.

        Under tensor parallelism the captured region contains the blocks'
        all-reduce, so the graphs are only installed once they pass the checks in
        :meth:`_tp_graphs_are_safe`. Everything below the kill-switch runs on every
        rank, and the collectives are reached from decisions every rank computes
        identically — the grid is derived from ``max_seq_len`` and from a
        ``max_request_num`` that came out of an all-reduce, and the capture result
        is folded into the fingerprint rather than being allowed to return early.
        A rank that took the early exit its peers did not would leave them waiting
        in a consensus collective, which is the exact failure these gates exist to
        prevent.
        """
        if self._graph_manager is not None:
            return  # idempotent

        tp_size = get_tp_world_size()
        if tp_size > 1 and os.environ.get(_TP_GRAPH_ENV, "1") == "0":
            logger.warning(
                "%s=0: CUDA Graph disabled under tensor parallelism; running eager.",
                _TP_GRAPH_ENV,
            )
            return

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
            lazy=lazy,
        )
        captured = True
        try:
            if lazy:
                manager.capture_seed()
            else:
                manager.capture_all()
        except torch.cuda.OutOfMemoryError:
            # A failed capture may leave a half-open graph; dropping the manager
            # is safe because replay state is only installed on success. Under TP
            # this cannot simply return: the peers are on their way to a
            # collective, so the failure has to be *reported* into it instead.
            logger.warning("CUDA graph capture ran out of memory; falling back to eager decode")
            captured = False

        if tp_size > 1 and not self._tp_graphs_are_safe(manager, captured):
            manager.discard()
            return
        if not captured:
            return
        self._graph_manager = manager

    def _tp_graphs_are_safe(self, manager: CUDAGraphManager, captured: bool) -> bool:
        """Whether this rank's captured graphs may serve traffic. Same answer everywhere.

        Two checks, in this order because the first is what makes the second
        well-defined:

        1. **Grid agreement.** Every rank contributes a fingerprint of what it
           captured, or ``0`` if it captured nothing. Ranks that disagree all
           return ``False`` together — an asymmetric grid means one rank replays
           where another runs eager, and the replayed all-reduce then waits
           forever. A grid of ``0`` fails even when unanimous, so passing this
           establishes that every rank *has* graphs and the parity check below is
           entered by all of them or none.
        2. **Numerical parity.** Each rank compares every captured graph against
           an eager step on the same synthetic input and keeps the graphs only if
           all ranks are within :data:`TP_GRAPH_PARITY_ATOL`. Reduced with a
           minimum so one rank's failure retires the graphs everywhere; a group
           where half the ranks replay is the hang again.

        Both results are functions of a collective's output, which is why every
        rank branches the same way without a second round of agreement.
        """
        fingerprint = manager.grid_fingerprint() if captured else 0
        if not all_ranks_agree(fingerprint) or fingerprint == 0:
            logger.warning(
                "tensor-parallel ranks captured different CUDA graph grids "
                "(this rank: %d); dropping graphs on every rank and decoding eager",
                fingerprint,
            )
            return False

        error = manager.max_parity_error(self.vocab_size)
        # Phrased as ``error <= tol`` rather than ``not error > tol`` so that a NaN
        # difference — the signature of a graph reading freed memory — fails the
        # gate instead of slipping through a negated comparison.
        local_ok = error <= TP_GRAPH_PARITY_ATOL
        if all_reduce_min(int(local_ok)) != 1:
            logger.warning(
                "CUDA graph replay disagrees with eager decode by %.3e (tolerance %.1e) "
                "on at least one rank; dropping graphs and decoding eager",
                error,
                TP_GRAPH_PARITY_ATOL,
            )
            return False

        logger.info(
            "TP CUDA graphs verified: worst graph-vs-eager logit difference %.3e (tolerance %.1e)",
            error,
            TP_GRAPH_PARITY_ATOL,
        )
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        multi_modal_inputs: dict[str, Any] | None = None,
        logits_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one model step.

        Dispatches to a captured CUDA graph when the current step is a decode
        (``seq_len == 1``) whose ``(batch_size, max_actual_seq_len)`` matches one
        of the captured buckets; otherwise runs eager.

        Args:
            input_ids: ``[batch, seq_len]`` token ids.
            position_ids: Absolute positions for this step.
            multi_modal_inputs: Processor outputs for a multimodal prefill.
            logits_positions: Optional ``[batch]`` per-sequence position whose
                logits the caller wants. Given (a prefill), the model gathers
                those hidden states *before* the lm_head GEMM and returns
                ``[batch, vocab]`` instead of ``[batch, seq_len, vocab]`` —
                for a long prompt that skips seq_len-1 of the vocabulary
                projections. ``None`` (decode, graph replay) returns full logits.
        """
        if self._graph_manager is not None and multi_modal_inputs is None:
            # A decode step (``seq_len == 1``) with no vision payload: eligible
            # for graph replay on text and multimodal models alike — the vision
            # tokens of the latter already sit in the KV cache.
            replayed = self._graph_manager.try_replay(input_ids, position_ids, self.atten_info)
            if replayed is not None:
                return replayed

        if self.spec.is_multimodal:
            return self.model(
                input_ids, position_ids, self.atten_info, multi_modal_inputs, logits_positions
            )

        return self.model(
            input_ids, position_ids, self.atten_info, logits_positions=logits_positions
        )
