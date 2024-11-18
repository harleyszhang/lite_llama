import torch, json, time
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

from .mem_manager import ComputeMaxAvailableBlocks, KVCacheMemoryManager

from ..models.model_config import LlamaConfig
from ..models.llama import Llama
from .cuda_graph import ModelRunner

def convert_hf_to_triton(hf_sd, model_args):
    """
    将 Hugging Face 模型的权重字典转换为自定义模型的权重字典。

    参数:
        hf_sd (dict): Hugging Face 模型的状态字典。
        model_args (LlamaConfig): 自定义模型的配置参数。

    返回:
        dict: 转换后的状态字典。
    """
    mapping = {
        "tok_embeddings.weight": "embed_tokens.weight",
        "norm.weight": "norm_weight", 
        "output.weight": "lm_head.weight",
    }

    layers = {
        # key 是原始权重值, value 是自定义模型结构权重参数
        "layers.{i}.attention.wq.weight": "layers.{i}.attention.wq.weight",
        "layers.{i}.attention.wk.weight": "layers.{i}.attention.wk.weight",
        "layers.{i}.attention.wv.weight": "layers.{i}.attention.wv.weight",
        "layers.{i}.attention.wo.weight": "layers.{i}.attention.wo.weight",
        "layers.{i}.feed_forward.w1.weight": "layers.{i}.feed_forward.gate_proj.weight",
        "layers.{i}.feed_forward.w3.weight": "layers.{i}.feed_forward.up_proj.weight",
        "layers.{i}.feed_forward.w2.weight": "layers.{i}.feed_forward.down_proj.weight",
        "layers.{i}.attention_norm.weight": "layers.{i}.attention_norm.weight",
        "layers.{i}.ffn_norm.weight": "layers.{i}.ffn_norm.weight",
    }

    # 根据 Transformer 层数量生成映射
    for i in range(model_args.n_layers):
        for hf_key, custom_key in layers.items():
            mapped_key = hf_key.format(i=i) # hf 权重参数字典 key
            custom_mapped_key = custom_key.format(i=i) # 自定义模型权重参数字典 key
            mapping[mapped_key] = custom_mapped_key

    # 创建新的状态字典
    new_sd = {}
    for hf_key, tensor in tqdm(hf_sd.items(), desc="Mapping weights"):
        custom_key = mapping.get(hf_key, None)
        if custom_key is not None:
            new_sd[custom_key] = tensor
        else:
            # 如果某些权重不需要映射，可以选择忽略或处理
            pass  # 忽略未映射的权重
    
    torch.save(new_sd, "/gemini/code/Llama-3.2-1B-Instruct/my_weight/my_llama3.2-1B.pth")

    return new_sd

@dataclass
class ModelRunnerConfig:
    block_size = 1
    checkpoints_dir = "/gemini/code/Llama-3.2-1B-Instruct"
    max_batch_size = 32
    gpu_memory_utilization=0.9

class AttentionInfo:
    def __init__(self):
        self.batch_size = None
        self.seq_len = None
        self.max_gen_len = 128

        self.select_index = None
        self.start_index = None
        self.end_index = None

        self.num_kv_heads = None
        self.kv_buffer = None
        self.is_prefill = None
        self.max_num_tokens = None

class ModelExecutor:
    model = Llama
    model_runner_config = ModelRunnerConfig
    atten_info = AttentionInfo

    @staticmethod
    def build(
        checkpoints_dir: str, 
        tokenizer_path: str, 
        max_batch_size: int,
        max_seq_len: int,
        load_model: bool, 
        triton_weight=True,
        compiled_model=True,
        device: str = "cuda", 
    ):
        """
        构建LLaMAInfer实例, 加载模型和分词器。

        参数:
            checkpoints_dir (str): 模型检查点目录路径。
            tokenizer_path (str): 分词器模型文件路径。
            load_model (bool): 是否加载模型权重。
            max_seq_len (int): 最大序列长度。
            max_batch_size (int): 最大批处理大小。
            device (str): 设备类型（'cuda'或'cpu'）。

        返回:
            GenerateText: 初始化后的 GenerateText 实例。
        """
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')

            checkpoint = torch.load(ckpt_path, map_location="cuda")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        else:
            checkpoint = None

        # 读取模型参数
        params_path = Path(checkpoints_dir) / "config.json"
        assert params_path.exists(), f"params.json not found in {checkpoints_dir}"
        with open(params_path, "r") as f:
            params = json.load(f)

        model_args: LlamaConfig = LlamaConfig(
            params,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
        )

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 设置默认张量类型
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        if load_model:
            if triton_weight:
                state_dict = checkpoint
                print("Load Triton weight directly!")
            else:
                state_dict = convert_hf_to_triton(checkpoint, model_args) # 转换权重名称
        else:
            state_dict = None

        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # 初始化自定义的 Llama 模型
        model = Llama(model_args).to(device)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            if 'rope.freqs' in checkpoint:
                del checkpoint['rope.freqs']  # 删除检查点中未匹配的键（例如rope.freqs）
            # 使用转换后的 state_dict 加载模型
            model.load_state_dict(state_dict, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        return ModelExecutor(model, compiled_model)

    def __init__(self, model: Llama, compiled_model=True, device="cuda"):
        self.model = model

        self.model_config = self._load_model_config()
        self.max_gpu_num_blocks, self.max_num_tokens = self._get_max_tokens(self.model_config)
        self.kv_mem_manager = self._init_mem_manager(
            self.model_config, self.max_gpu_num_blocks, self.max_num_tokens, block_size=1, 
            dtype=torch.float16,  device="cuda"
        )

        self.gpu_kv_buffer = self.kv_mem_manager.gpu_kv_buffer
        self.model_runner = None

        if compiled_model:
            self.apply_cuda_graph() # 真正的调用模型推理的代码

    def _load_model_config(self, device="cuda"):
        checkpoints_dir = self.model_runner_config.checkpoints_dir
        # 读取模型参数
        params_path = Path(checkpoints_dir) / "config.json"
        assert params_path.exists(), f"params.json not found in {checkpoints_dir}"
        with open(params_path, "r") as f:
            params = json.load(f)

        model_config: LlamaConfig = LlamaConfig(params, device=device,)

        return model_config

    def _get_max_tokens(self, model_config, gpu_memory_utilization=0.9, block_size=1):
        avaliable_blocks = ComputeMaxAvailableBlocks(model_config, gpu_memory_utilization, block_size)
        max_gpu_num_blocks = avaliable_blocks.compute_num_available_blocks()
        max_gpu_num_tokens = max_gpu_num_blocks * block_size

        return max_gpu_num_blocks, max_gpu_num_tokens
    
    def _init_mem_manager(self, model_config, gpu_num_blocks, max_num_tokens, block_size=1, dtype=torch.float16,  device="cuda"):
        kv_mem_manager = KVCacheMemoryManager(
            head_dim = self.model_config.head_dim,
            num_kv_heads = self.model_config.num_kv_heads,
            num_layers = self.model_config.num_layers,

            gpu_num_blocks = gpu_num_blocks,
            max_num_tokens = max_num_tokens,
            block_size = block_size,
            dtype = dtype,
            device=device
        )

        return kv_mem_manager

    def apply_cuda_graph(self,):
        """应用 cuda graph 优化
        参数:
            - input_ids: 输入 tokens id 列表, shape: (batch_size, seq_len)
            - prev_pos: 当前处于第几轮迭代循环, 生成第几个 token
        """
        self.model_runner = ModelRunner(self.model, vocab_size = self.args.vocab_size, max_batch_size=self.args.max_batch_size)
        self.model_runner.capture_decode_graph()

    def forward(self, input_ids, prev_pos, max_gen_len):
        batch_size, seq_len = input_ids.shape # 静态批处理, batch 中每个请求的 seq_len 都相等
        
        if seq_len > 1:
            is_prefill = True
            # 一次性分配最大所需 kv cache
            need_size = batch_size * seq_len
            select_index, start_index, end_index  = self.kv_mem_manager.alloc_contiguous_kvcache(need_size)  
        else:
            is_prefill = False
            # need_size = batch_size

        self.atten_info.batch_size = batch_size
        self.atten_info.seq_len = seq_len
        self.atten_info.max_gen_len = max_gen_len

        self.atten_info.select_index = select_index
        self.atten_info.start_index = start_index
        self.atten_info.end_index = end_index

        self.atten_info.num_kv_heads = self.model_config.num_kv_heads

        self.atten_info.kv_buffer = self.gpu_kv_buffer
        self.atten_info.max_num_tokens = self.max_num_tokens

        # print(input_ids.shape)
        if prev_pos == 0 or not self.compiled_model:
            logits = self.model.forward(input_ids, prev_pos, self.atten_info)
        else:
            logits = self.model_runner.decode(input_ids, prev_pos, self.atten_info)

        return logits