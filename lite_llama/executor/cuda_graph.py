import torch
from copy import deepcopy
from .executor_struct import AttentionInfo
from .mem_manager import ComputeMaxAvailableBlocks, KVCacheMemoryManager

_BATCH_SIZE_ALIGNMENT = 8
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [_BATCH_SIZE_ALIGNMENT * i for i in range(1, 1025)]

class CUDAGraphRunner:
    def __init__(self, model):
        self.model = model
        self._cuda_graph = None
        self._graph_inputs = None
        # self.input_buffers: Dict[str, torch.Tensor] = {}
        self._graph_output = None
    
    def capture(self, input_ids: torch.Tensor, start_pos, atten_info: AttentionInfo):
        assert self._cuda_graph is None, "Already compiled the model"
        # 用于捕获的占位符输入
        self._graph_inputs = [input_ids, start_pos, atten_info]
        # self._graph_inputs["input_ids"] = input_ids
        # self._graph_inputs["start_pos"] = start_pos
        # self._graph_inputs["atten_info"] = atten_info
        
        # Warm up
        graph_capture_stream = torch.cuda.Stream()
        graph_capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(graph_capture_stream):
            _ = self.model.forward(*self._graph_inputs)
        torch.cuda.current_stream().wait_stream(graph_capture_stream)

        # Capture the graph
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._graph_output  = self.model.forward(*self._graph_inputs)

        # self._graph_output['output'] = out 

    def forward(self, input_ids: torch.Tensor, start_pos, atten_info: AttentionInfo):

        # self._graph_inputs["input_ids"].copy_(input_ids)
        # self._graph_inputs["start_pos"].fill_(start_pos)  # 使用 fill_ 原地更新张量的值
        # self._graph_inputs["start_pos"] = deepcopy(start_pos)

        # 更新 AttentionInfo 的相关缓冲区        
        # self._graph_inputs["atten_info"].decode_index.copy_(atten_info.decode_index)
        # self._graph_inputs["atten_info"].select_index.copy_(atten_info.select_index)
        
        # 更新输入缓冲区
        self._graph_inputs[0].copy_(input_ids) # 据填充 graph 的输入内存
        self._graph_inputs[1] = deepcopy(start_pos)

        self._graph_inputs[2].decode_index.copy_(atten_info.decode_index)
        self._graph_inputs[2].select_index.copy_(atten_info.select_index)

        for i, kv_buffer in enumerate(atten_info.kv_buffer):
            self._graph_inputs[2].kv_buffer[i].copy_(kv_buffer)

        # 逐个复制 kv_buffer 列表中的张量
        # for existing_buf, new_buf in zip(self._graph_inputs[2].kv_buffer, atten_info.kv_buffer):
        #     existing_buf.copy_(new_buf)

        self._cuda_graph.replay()

        return self._graph_output
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
class ModelRunner:
    def __init__(self, model, model_config, 
                max_gpu_num_blocks:int,
                kv_mem_manager: KVCacheMemoryManager,
                max_gen_len: int = 128, seq_len: int=1, start_pos = 8
    ):
        self.model = model
        self.model_config = model_config
        self.max_gpu_num_blocks = max_gpu_num_blocks
        self.kv_mem_manager = kv_mem_manager

        self.vocab_size = self.model_config.vocab_size
        self.graph_max_batch_size=self.model_config.max_batch_size

        self.max_seq_len = model_config.max_seq_len
        self.seq_len = seq_len
        self.max_gen_len = max_gen_len
        self.start_pos = start_pos

        self.graph_runners = {}

    def build_atten_info(self, batch_size, dtype=torch.float16, device="cuda"):
        """针对 decode 阶段, 构建 attention 输入信息结构体"""

        atten_info = AttentionInfo
        atten_info.kv_buffer = self.kv_mem_manager.gpu_kv_buffer

        atten_info.select_index = torch.zeros((512), dtype=torch.long, device=device)
        decode_index, _, _ = self.kv_mem_manager.alloc_contiguous_kvcache(batch_size)
        atten_info.decode_index = decode_index

        intermediate_tensors = torch.zeros(
            (batch_size, self.model_config.num_kv_heads, self.model_config.head_dim), # seq_len = 1
            dtype=dtype,
            device=device
        )
             
        for layer_index in range(self.model_config.num_layers):
            atten_info.kv_buffer[layer_index][decode_index, :self.model_config.num_kv_heads, :] = intermediate_tensors
            atten_info.kv_buffer[layer_index][decode_index, self.model_config.num_kv_heads:, :] = intermediate_tensors

        return atten_info
    
    def capture_decode_graph(self, ):
        """
        针对 decode 阶段捕获 CUDA 图
        """
        # 获取要捕获的批量大小列表，确保批量大小不超过最大批量大小
        batch_size_capture_list = [bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= self.graph_max_batch_size]
        print("cuda graph support batch list", batch_size_capture_list)
        
        # NOTE: Capturing the largest batch size first may help reduce the memory usage of CUDA graph.
        for batch_size in reversed(batch_size_capture_list):
            # 构造输入 tokens id 张量
            input_ids = torch.randint(0, self.vocab_size, (batch_size, 1)).cuda()
            atten_info = self.build_atten_info(batch_size,)
            print("apply cuda grpah atten_info.decode_index shape ", atten_info.decode_index.shape)
            
            graph_intput = (input_ids, self.start_pos, atten_info)
            graph_runner = CUDAGraphRunner(self.model)
            # graph 图捕捉输入
            graph_runner.capture(*graph_intput)
            self.graph_runners[batch_size] = graph_runner

    def decode(self, x: torch.Tensor, start_pos, atten_info: AttentionInfo):
        # TODO: 灵活适应不同模型. graph 输入参数 x 和 start_pos 必须和模型要求的输入参数一样
        batch_size = x.shape[0]
        if batch_size in self.graph_runners:
            # print("INFO: CUDA graph captured for this batch size, apply cuda graph to decode inference.")
            model_executable = self.graph_runners[batch_size]
        else:
            print("Warning: CUDA graph not captured for this batch size, falling back to original model.")
            model_executable = self.model
        
        logits = model_executable(x, start_pos, atten_info)
        return logits