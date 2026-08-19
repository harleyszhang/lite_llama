"""Model executor: builds the model, sizes the KV cache, and runs each forward step.

Responsibilities:

* :meth:`ModelExecutor.build` — resolve the architecture from ``config.json`` via the
  :class:`~lite_llama.models.registry.ModelRegistry`, instantiate it on the meta
  device, and stream the checkpoint in.
* :meth:`prefill_alloc_kv_cache` / :meth:`decode_alloc_kv_cache` — reserve cache rows.
* :meth:`forward` — dispatch to the model, passing multimodal inputs only when the
  resolved :class:`~lite_llama.models.registry.ModelSpec` says the model wants them.

Weight loading uses ``torch.device("meta")`` for the empty skeleton (no ``accelerate``
dependency) and relies on ``load_state_dict(assign=True)`` to replace the meta
parameters with the real tensors mmap-loaded from disk.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..kernels import update_kv_index
from ..models.registry import ModelRegistry, ModelSpec
from ..utils.logger import get_logger
from .cuda_graph import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_SEQ_LEN_BUCKETS,
    CUDAGraphManager,
)
from .executor_struct import AttentionInfo
from .mem_manager import KVCacheMemoryManager, MemoryProfiler
from .req_tokens_manager import ReqTokensManager

logger = get_logger(__name__)


@contextlib.contextmanager
def _init_empty_parameters():
    """Skeleton context: parameters allocate on the meta device, buffers do not.

    Mirrors ``accelerate.init_empty_weights(include_buffers=False)``. Buffers must
    keep real storage because non-persistent buffers such as
    :attr:`~lite_llama.models.rotary_embedding.RotaryEmbedding.inv_freq` are absent
    from checkpoints and therefore cannot be materialised by ``load_state_dict``.
    """
    original = nn.Module.register_parameter

    def register_meta_parameter(module: nn.Module, name: str, param) -> None:
        original(module, name, param)
        if module._parameters.get(name) is None:
            return
        # Preserve the Parameter subclass and its attributes (e.g. `requires_grad`).
        existing = module._parameters[name]
        kwargs = existing.__dict__
        module._parameters[name] = type(existing)(existing.to(torch.device("meta")), **kwargs)

    try:
        nn.Module.register_parameter = register_meta_parameter
        yield
    finally:
        nn.Module.register_parameter = original


def _text_config(config: Any) -> Any:
    """Return the text/language sub-config, unwrapping multimodal wrappers."""
    return getattr(config, "text_config", config)


def _text_field(config: Any, *names: str) -> Any:
    """Return the first attribute from ``names`` that exists on ``config``.

    lite_llama configs use short names (``num_layers``, ``num_kv_heads``) whereas
    HuggingFace configs use ``num_hidden_layers`` / ``num_key_value_heads`` — this
    helper hides the mismatch when the executor unwraps a nested text config from a
    multimodal wrapper.
    """
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    raise AttributeError(f"{type(config).__name__} has none of {names}")


class ModelExecutor:
    """Owns the model, the KV-cache memory manager, and the per-step attention state."""

    def __init__(
        self,
        checkpoints_dir: str,
        config: Any,
        spec: ModelSpec,
        model: nn.Module,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
    ) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.config = config
        self.spec = spec
        self.model = model
        self.device = device

        text_config = _text_config(config)
        self.num_layers = _text_field(text_config, "num_layers", "num_hidden_layers")
        num_heads = _text_field(text_config, "num_heads", "num_attention_heads")
        self.num_kv_heads = getattr(text_config, "num_kv_heads", None) or getattr(
            text_config, "num_key_value_heads", num_heads
        )
        hidden_size = _text_field(text_config, "hidden_size")
        self.head_dim = getattr(text_config, "head_dim", None) or (hidden_size // num_heads)
        vocab_size = _text_field(text_config, "vocab_size")
        self.max_seq_len = getattr(text_config, "max_seq_len", None) or getattr(
            config, "max_seq_len", 2048
        )

        if max_gpu_num_blocks is None:
            profiler = MemoryProfiler(
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                device=device,
            )
            max_gpu_num_blocks = profiler.available_kv_blocks(model, vocab_size)

        self.kv_mem_manager = KVCacheMemoryManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            gpu_num_blocks=max_gpu_num_blocks,
            device=device,
        )
        self.max_request_num = max(1, max_gpu_num_blocks // self.max_seq_len)
        self.req_tokens_manager = ReqTokensManager(self.max_request_num, self.max_seq_len)

        self.atten_info = AttentionInfo()
        self.atten_info.kv_buffer = self.kv_mem_manager.gpu_kv_buffer
        self.atten_info.b_req_tokens_table = self.req_tokens_manager.b_req_tokens_table

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
    ) -> ModelExecutor:
        """Load config + weights and return a ready-to-run executor.

        Args:
            checkpoints_dir: Directory holding ``config.json`` and a ``*.pth`` checkpoint.
            max_seq_len: Upper bound on sequence length; also bounds the KV cache.
            max_gpu_num_blocks: Manual KV-cache size in tokens; profiled when ``None``.
            device: Torch device string.
        """
        config, spec = ModelRegistry.load_config(checkpoints_dir, max_seq_len)
        model = cls._load_weights(config, spec, checkpoints_dir, device)
        return cls(checkpoints_dir, config, spec, model, max_gpu_num_blocks, device)

    @staticmethod
    def _load_weights(config: Any, spec: ModelSpec, checkpoints_dir: str, device: str) -> nn.Module:
        """Instantiate on meta, then assign real fp16 weights from the checkpoint."""
        start = time.time()

        if device.startswith("cuda") and not torch.cuda.is_available():
            # ``torch.load(..., map_location="cuda")`` would otherwise fail deep
            # inside pickle with a message that says nothing about drivers.
            raise RuntimeError(
                "device='cuda' was requested but torch.cuda.is_available() is False. "
                "This usually means the installed torch build targets a newer CUDA "
                f"than the NVIDIA driver on this machine. Installed: torch=={torch.__version__} "
                f"(cuda={torch.version.cuda}). Fix by installing a torch build that matches "
                "the local driver, e.g. `uv pip install torch --index-url "
                "https://download.pytorch.org/whl/cu124`."
            )

        # Build the skeleton without allocating parameter storage.
        logger.info(
            "Initializing model of type '%s' and moving it to device '%s'...",
            spec.model_type,
            device,
        )
        with _init_empty_parameters():
            model = ModelRegistry.build_model(config, spec)
        logger.info("The model has been initialized and moved to the device. '%s'", device)

        checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
        if not checkpoints:
            raise FileNotFoundError(
                f"no *.pth checkpoint found in {checkpoints_dir}; run "
                "`lite-llama-convert` on the HuggingFace weights first"
            )
        logger.info('Loading checkpoint "%s"', checkpoints[0])
        state_dict = torch.load(checkpoints[0], mmap=True, weights_only=True, map_location=device)
        # Models whose submodule layout depends on the installed transformers version
        # (e.g. LLaVA's CLIP vision tower) normalise the checkpoint keys here.
        remap = getattr(model, "remap_checkpoint_keys", None)
        if callable(remap):
            state_dict = remap(state_dict)
        # assign=True swaps the meta params for the loaded tensors instead of copying.
        model.load_state_dict(state_dict, strict=True, assign=True)

        model.eval().to(device)
        for name, param in model.named_parameters():
            if param.is_meta:
                raise RuntimeError(f"parameter {name!r} was not materialised from the checkpoint")
        logger.info("Loaded state dict in %.2fs", time.time() - start)

        # The converter stores fp16 weights; half() is a no-op that verifies it.
        model.half()
        for param in model.parameters():
            if param.dtype != torch.float16:
                raise RuntimeError(
                    f"expected fp16 parameters after half(), got {param.dtype}"
                )
        logger.info("Converted model to half precision (FP16)")
        return model

    # --------------------------------------------------------- kv allocation #
    def _init_req_tokens_table(self, b_req_idx, b_seq_len, alloc_index) -> torch.Tensor:
        """Record which cache rows each prefill sequence occupies.

        Returns:
            ``b_start_loc``: start offset of each sequence in the flattened batch.
        """
        b_seq_len_list = b_seq_len.cpu().tolist()
        b_req_idx_list = b_req_idx.cpu().tolist()
        b_start_loc = torch.zeros(len(b_seq_len_list), dtype=torch.int32, device=self.device)

        start = 0
        for i, seq_len in enumerate(b_seq_len_list):
            b_start_loc[i] = start
            self.atten_info.b_req_tokens_table[b_req_idx_list[i], :seq_len] = alloc_index[
                start : start + seq_len
            ]
            start += seq_len
        return b_start_loc

    def prefill_alloc_kv_cache(self, max_prompt_len, actual_prompt_lens, b_req_idx) -> torch.Tensor:
        """Reserve cache rows for the whole prompt batch.

        Vision tokens need no special handling: the prompt already contains one
        token per vision patch (the processor expanded the ``<image>`` marker), so
        the reservation below covers them.
        """
        batch_size = len(actual_prompt_lens)
        self.atten_info.b_req_idx = b_req_idx
        self.atten_info.cur_select_index, _ = self.kv_mem_manager.alloc_kvcache_index(
            max_prompt_len * batch_size
        )
        self.atten_info.b_seq_len = actual_prompt_lens
        self.atten_info.max_actual_seq_len = max_prompt_len
        self.atten_info.b_start_loc = self._init_req_tokens_table(
            b_req_idx, actual_prompt_lens, self.atten_info.cur_select_index
        )
        return self.atten_info.cur_select_index

    def decode_alloc_kv_cache(self, batch_size) -> torch.Tensor:
        """Reserve one cache row per sequence for the next decode step.

        ``update_kv_index`` writes at position ``b_seq_len - 1``, so ``b_seq_len``
        must be incremented *before* the kernel is launched. The legacy code did
        the opposite, which overwrote the mapping of the last prompt token and
        silently produced non-deterministic completions once a second request was
        served on the same executor.
        """
        self.atten_info.cur_select_index, _ = self.kv_mem_manager.alloc_kvcache_index(batch_size)
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

        logger.info(
            "Capturing CUDA graphs for batch_sizes=%s seq_len_buckets=%s",
            batch_sizes,
            seq_len_buckets,
        )
        manager = CUDAGraphManager(
            self.model,
            kv_buffer=self.kv_mem_manager.gpu_kv_buffer,
            b_req_tokens_table=self.req_tokens_manager.b_req_tokens_table,
            batch_sizes=batch_sizes,
            seq_len_buckets=seq_len_buckets,
            device=self.device,
        )
        manager.capture_all()
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
