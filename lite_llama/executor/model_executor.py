import torch, json, time,logging, os, shutil, glob
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

from .mem_manager import ComputeMaxAvailableBlocks, KVCacheMemoryManager

from ..models.model_config import LlamaConfig, Qwen2Config
from ..models.llama import Llama
from ..models.qwen2 import Qwen2Model

from .cuda_graph import ModelRunner
from .executor_struct import AttentionInfo, ModelRunnerConfig

logger = logging.getLogger(__name__)

def indexs_convert(indexs: torch.tensor, batch_size: int):
    """
    prefill 阶段分配的kv cache 索引和 decode 阶段分配的索引合并在一起需要做变换
    TODO: 支持连续批处理开发时用上.
    """
    pass

def build_new_weight_dir(checkpoints_dir:str, new_sd):
    # 保存 lite_llama 模型权重并构建新的权重目录
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在的目录
    my_weight_dir = os.path.join(current_dir, "../../my_weight/") # 项目所在根目录
    os.makedirs(my_weight_dir, exist_ok=True) # 创建文件夹（如果不存在）
    torch.save(new_sd, os.path.join(my_weight_dir, "my_llama3.2-1B.pth"))

    # 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(checkpoints_dir, "*.json"))
    for file_path in json_files:
        shutil.copy(file_path, my_weight_dir) # 复制 hf 权重目录的所有 json 文件到新的目录
        print(f"已复制: {file_path} -> {my_weight_dir}")

def convert_llama_hf_to_triton(checkpoints_dir, hf_sd, model_args):
    """
    将 Hugging Face 模型的权重字典转换为自定义模型的权重字典。

    参数:
        checkpoints_dir: Hugging Face 模型的目录
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
            # 左边是 hf 权重参数字典 key, 右边是自定义模型权重参数字典 key
            mapping[hf_key.format(i=i)] = custom_key.format(i=i)

    # 创建新的状态字典
    new_sd = {}
    for hf_key, tensor in tqdm(hf_sd.items(), desc="Mapping weights"):
        if hf_key in mapping:
            new_sd[custom_key] = tensor
        else:
            print(f"Warning: Unmapped key {hf_key}")
    
    build_new_weight_dir(checkpoints_dir, new_sd)
    
    return new_sd

class ModelExecutor:
    # 定义类属性
    tokenizer = None
    model_config = None
    model = None
    # model_runner_config = ModelRunnerConfig
    atten_info = AttentionInfo

    # 通过静态方法 build 将类属性当作默认配置使用
    @staticmethod
    def build(
        checkpoints_dir: str, 
        tokenizer_path: str, 
        max_batch_size: int,
        max_seq_len: int,
        load_model: bool = True, 
        triton_weight: bool = True,
        device: str = "cuda", 
    ):
        """
        构建 ModelExecutor 实例, 加载模型、分词器和初始化推理信息结构体 atten_info。

        参数:
            checkpoints_dir (str): 模型检查点目录路径。
            tokenizer_path (str): 分词器模型文件路径。
            load_model (bool): 是否加载模型权重。
            max_seq_len (int): 最大序列长度。
            max_batch_size (int): 最大批处理大小。
            device (str): 设备类型（'cuda'或'cpu'）。

        返回:
            ModelExecutor: 初始化后的 ModelExecutor 实例。
        """
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model_config = ModelExecutor._load_model_config(checkpoints_dir, max_batch_size, max_seq_len, device=device)
        model = ModelExecutor._load_model_weight(model_config, checkpoints_dir, load_model, triton_weight, device=device) # 加载权重后的模型

        return ModelExecutor(tokenizer, model_config, model, True)

    @staticmethod
    def _load_model_weight(model_args, checkpoints_dir, load_model = True, triton_weight=True, device="cuda"):
        prev_time = time.time()
        
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')

            hf_sd = torch.load(ckpt_path, map_location="cuda")
            print(f"Loaded weight checkpoints files in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        else:
            hf_sd = None

        # 设置默认张量类型
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        if load_model:
            if triton_weight:
                state_dict = hf_sd
                print("Load Triton weight directly!")
            else:
                state_dict = convert_llama_hf_to_triton(checkpoints_dir, hf_sd, model_args) # 转换权重名称
        else:
            state_dict = None

        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        
        # 初始化自定义的 Llama 模型
        if model_args.model_type == "llama":
            model = Llama(model_args).to(device)
        elif model_args.model_type == "qwen2":
            model = Qwen2Model(model_args).to(device) # 将模型移动到设备并转换为半精度
        else:
            print("Error, unsupported model!")

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            if 'rope.freqs' in hf_sd:
                del hf_sd['rope.freqs']  # 删除检查点中未匹配的键（例如rope.freqs）
            # 使用转换后的 state_dict 加载模型
            model.load_state_dict(state_dict, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        return model
    
    @staticmethod
    def _load_model_config(checkpoints_dir, max_batch_size, max_seq_len, device="cuda"):
        checkpoints_dir = checkpoints_dir
        # 读取模型参数
        params_path = Path(checkpoints_dir) / "config.json"
        assert params_path.exists(), f"params.json not found in {checkpoints_dir}"
        with open(params_path, "r") as f:
            params = json.load(f)

        if params["model_type"]== "llama":
            model_config: LlamaConfig = LlamaConfig(
                params, 
                max_batch_size = max_batch_size,
                max_seq_len = max_seq_len,
                device=device
            )
        elif params["model_type"] == "qwen2":
            model_config: Qwen2Config = Qwen2Config(
                params, 
                max_batch_size = max_batch_size,
                max_seq_len = max_seq_len,
                device=device
            )

        return model_config

    def __init__(self, tokenizer:AutoTokenizer, model_config, model, compiled_model=True, device="cuda"):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.model = model

        self.model_runner = None
        self.compiled_model = False
        
        if self.compiled_model:
            self.max_gpu_num_blocks, self.kv_mem_manager = self.apply_cuda_graph() # 调用 cuda graph 优化
        else:
            self.max_gpu_num_blocks, self.max_num_tokens = self._get_max_tokens(self.model_config)
            self.kv_mem_manager = self._init_mem_manager(
                self.model_config, self.max_gpu_num_blocks, self.max_num_tokens, block_size=1, 
                dtype=torch.float16,  device="cuda")
        
        self.gpu_kv_buffer = self.kv_mem_manager.gpu_kv_buffer
        self.atten_info = AttentionInfo
        self.atten_info.kv_buffer = self.kv_mem_manager.gpu_kv_buffer

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
            block_size = block_size,
            dtype = dtype,
            device=device
        )

        return kv_mem_manager

    def apply_cuda_graph(self, ):
        """应用 cuda graph 优化
        参数:
            - input_ids: 输入 tokens id 列表, shape: (batch_size, seq_len)
            - prev_pos: 当前处于第几轮迭代循环, 生成第几个 token
        """
        max_gpu_num_blocks, max_num_tokens = self._get_max_tokens(self.model_config)
        
        kv_mem_manager = self._init_mem_manager(
            self.model_config, max_gpu_num_blocks, block_size=1, 
            dtype=torch.float16,  device="cuda"
        )
        self.model_runner = ModelRunner(
            self.model, 
            self.model_config, 
            max_gpu_num_blocks, 
            kv_mem_manager
        )
        self.model_runner.capture_decode_graph()

        return  max_gpu_num_blocks, kv_mem_manager

    def forward(self, input_ids, prev_pos):
        batch_size, seq_len = input_ids.shape # 静态批处理, batch 中每个请求的 seq_len 都相等
        if seq_len > 1:
            # 一次性分配最大所需 kv cache. seq0: [token0, token1, token2, token3,], seq1: [token0, token1, token2, token3,]
            need_size = batch_size * (seq_len)
            alloc_mem = self.kv_mem_manager.alloc_contiguous_kvcache(need_size)
            if alloc_mem is not None:
                select_index = alloc_mem[0]
            else:
                select_index, _, _  = self.kv_mem_manager.alloc_kvcache(need_size)
            self.atten_info.select_index = select_index
        else:
            alloc_mem = self.kv_mem_manager.alloc_contiguous_kvcache(batch_size)
            if alloc_mem is not None:
                decode_index = alloc_mem[0]
            else:
                decode_index, _, _  = self.kv_mem_manager.alloc_kvcache(batch_size)
            self.atten_info.decode_index = decode_index

        if prev_pos == 0 or not self.compiled_model:
            logits = self.model.forward(input_ids, prev_pos, self.atten_info)
        else:
            logits = self.model_runner.decode(input_ids, prev_pos, self.atten_info) # TODO: cuda graph 可能执行失败, 待解决
        
        if seq_len == 1 and self.atten_info.select_index.numel() > 0:
            select_index = torch.cat([self.atten_info.select_index.view(batch_size, -1),
                                     self.atten_info.decode_index.view(batch_size, -1)], dim=1).view(-1)
            self.atten_info.select_index = select_index
        
        return logits, self.atten_info.select_index