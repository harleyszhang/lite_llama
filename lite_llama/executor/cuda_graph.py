import torch
from copy import deepcopy
from typing import Dict
from .executor_struct import AttentionInfo
from .mem_manager import KVCacheMemoryManager
from ..models.utils import weak_ref_tensor

_BATCH_SIZE_ALIGNMENT = 8
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [_BATCH_SIZE_ALIGNMENT * i for i in range(1, 1025)]

class CUDAGraphRunner:
    def __init__(self, model):
        self.model = model
        self._cuda_graph = None
        self._graph_inputs: Dict[str, torch.Tensor] = {}
        self._graph_output = None
    
    def capture(
        self, 
        input_ids: torch.Tensor, 
        position_ids: torch.Tensor, 
        atten_info: AttentionInfo
    ):
        assert self._cuda_graph is None, "Already compiled the model"
        # 用于捕获的占位符输入
        self._graph_inputs = [input_ids, position_ids, atten_info]
        
        # Warm up
        graph_capture_stream = torch.cuda.Stream()
        graph_capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(graph_capture_stream):
            _ = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                atten_info=atten_info,
            )
        torch.cuda.current_stream().wait_stream(graph_capture_stream)

        # Capture the graph
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._graph_output = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                atten_info=atten_info,
            )
        
        # Save the input and output buffers.
        self._graph_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "kv_buffer": atten_info.kv_buffer,
            "cur_select_index": atten_info.cur_select_index,
            "b_req_tokens_table": atten_info.b_req_tokens_table,
            "b_req_idx": atten_info.b_req_idx,
        }

    def forward(
        self, 
        input_ids: torch.Tensor, 
        position_ids: torch.Tensor, 
        atten_info: AttentionInfo,
    ):
        del atten_info.kv_buffer # kv_buffer are fixed tensors, so we don't need to copy them.
        del atten_info.b_req_tokens_table
        # 更新输入缓冲区
        self._graph_inputs["input_ids"].copy_(input_ids) # 据填充 graph 的输入内存
        self._graph_inputs["position_ids"].copy_(position_ids)

        self._graph_inputs["cur_select_index"].copy_(atten_info.cur_select_index)
        self._graph_inputs["b_req_idx"].copy_(atten_info.b_req_idx)

        self._cuda_graph.replay()

        return self._graph_output
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
class ModelRunner:
    def __init__(self, model, model_config, 
                max_gpu_num_blocks:int,
                kv_mem_manager: KVCacheMemoryManager,
                req_tokens_manager, 
                seq_len: int=1, start_pos = 8
    ):
        self.model = model
        self.model_config = model_config
        self.max_gpu_num_blocks = max_gpu_num_blocks
        self.kv_mem_manager = kv_mem_manager
        self.req_tokens_manager = req_tokens_manager

        self.vocab_size = self.model_config.vocab_size
        self.graph_max_batch_size=self.model_config.max_batch_size
        self.max_seq_len = model_config.max_seq_len

        # 随机参数定义
        self.seq_len = seq_len
        self.start_pos = start_pos

        self.graph_runners = {}

    def build_atten_info(self, batch_size, atten_info, device="cuda"):
        """针对 decode 阶段, 构建 attention 输入信息结构体"""
        atten_info.kv_buffer = self.kv_mem_manager.gpu_kv_buffer # torch.Tensor
        atten_info.b_req_tokens_table = self.req_tokens.manager.b_req_tokens_table # torch.Tensor

        atten_info.b_req_idx = torch.arange(batch_size, device = device) # torch.Tensor
        atten_info.b_seq_len = torch.ones(batch_size, dtype=torch.int32, device="cuda") # torch.Tensor
        atten_info.cur_select_index, = self.kv_mem_manager.alloc_kvcache_index(batch_size) # torch.Tensor

        return atten_info
    
    def capture_decode_graph(self, ):
        """
        针对 decode 阶段捕获 CUDA 图
        """
        # 获取要捕获的批量大小列表，确保批量大小不超过最大批量大小
        batch_size_capture_list = [bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= self.graph_max_batch_size]
        atten_info = AttentionInfo
        print("cuda graph support batch list", batch_size_capture_list)
        
        # NOTE: Capturing the largest batch size first may help reduce the memory usage of CUDA graph.
        for batch_size in reversed(batch_size_capture_list):
            # 构造输入 tokens id 张量
            input_ids = torch.randint(0, self.vocab_size, (batch_size, 1)).cuda()
            position_ids = (
                torch.arange(self.start_pos, self.start_pos + 1, device=input_ids.device)
                .unsqueeze(0)            # shape: [1, seq_len]
                .expand(batch_size, -1)  # shape: [batch_size, seq_len], 不分配额外内存
            )
            atten_info = self.build_atten_info(batch_size, atten_info)
            print("apply cuda grpah atten_info.decode_index shape ", atten_info.decode_index.shape)
            
            graph_intput = (input_ids, position_ids, atten_info)
            graph_runner = CUDAGraphRunner(self.model)
            
            # graph 图捕捉输入
            graph_runner.capture(*graph_intput)
            self.graph_runners[batch_size] = graph_runner
            
            self.kv_mem_manager.free_all()

    def decode(
        self, 
        input_ids: torch.Tensor, 
        position_ids: torch.Tensor, 
        atten_info: AttentionInfo
    ):
        batch_size = input_ids.shape[0]
        if batch_size in self.graph_runners:
            model_executable = self.graph_runners[batch_size]
        else:
            print("Warning: CUDA graph not captured for this batch size, falling back to original model.")
            model_executable = self.model
        
        logits = model_executable(input_ids, position_ids, atten_info)
        return logits