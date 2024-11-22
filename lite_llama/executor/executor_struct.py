from dataclasses import dataclass
import torch
from typing import List

@dataclass
class ModelRunnerConfig:
    block_size = 1
    checkpoints_dir = "/gemini/code/Llama-3.2-1B-Instruct"
    max_batch_size = 8
    gpu_memory_utilization=0.9

@dataclass
class AttentionInfo:
    select_index = torch.tensor([])
    kv_buffer = List[torch.tensor([])]
    decode_index = torch.tensor([])
    
    # def __init__(self,batch_size):
    #     decode_index = torch.empty([batch_size])