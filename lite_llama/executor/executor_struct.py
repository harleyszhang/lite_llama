from dataclasses import dataclass, field
import torch
from typing import Optional, Type
from ..models.model_config import LlamaConfig, Qwen2Config, Qwen3Config
from transformers import LlavaConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

CONFIG_CLASS_MAP: dict[str, Type] = {
    "llama": LlamaConfig,
    "qwen2": Qwen2Config,
    "qwen3": Qwen3Config,
    "qwen3_vl": Qwen3VLConfig,
    "llava": LlavaConfig,
}


@dataclass
class ModelRunnerConfig:
    block_size: int = 1
    checkpoints_dir: str = "/gemini/code/Llama-3.2-1B-Instruct"
    max_batch_size: int = 16
    gpu_memory_utilization: float = 0.9


@dataclass
class AttentionInfo:
    kv_buffer: list = field(default_factory=list)
    cur_select_index: Optional[torch.Tensor] = None
    b_req_tokens_table: Optional[torch.Tensor] = None
    b_start_loc: Optional[torch.Tensor] = None
    b_req_idx: Optional[torch.Tensor] = None
    b_seq_len: Optional[torch.Tensor] = None
    max_actual_seq_len: int = 0
