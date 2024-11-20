from dataclasses import dataclass
import torch

@dataclass
class ModelRunnerConfig:
    block_size = 1
    checkpoints_dir = "/gemini/code/Llama-3.2-1B-Instruct"
    max_batch_size = 8
    gpu_memory_utilization=0.9

@dataclass
class AttentionInfo:
    select_index = None
    new_select_index = torch.Tensor
    kv_buffer = torch.Tensor