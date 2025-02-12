from dataclasses import dataclass
import torch
from typing import List

@dataclass
class ModelRunnerConfig:
    block_size = 1
    checkpoints_dir = "/gemini/code/Llama-3.2-1B-Instruct"
    max_batch_size = 16
    gpu_memory_utilization=0.9

@dataclass
class AttentionInfo:
    # kv_cache = None # prefill 阶段的 context kv cache
    kv_buffer = List[torch.tensor([])]
    cur_select_index = torch.empty((0,),dtype=torch.int32)
    b_req_tokens_table = None
    b_start_loc = None
    b_req_idx = None