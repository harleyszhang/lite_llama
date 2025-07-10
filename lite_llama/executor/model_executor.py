import torch
import torch.nn as nn

import json, time
from pathlib import Path

from transformers import LlavaConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from .mem_manager import ComputeMaxAvailableBlocks, KVCacheMemoryManager
from .req_tokens_manager import ReqTokensManager

from .cuda_graph import ModelRunner
from .executor_struct import AttentionInfo
from ..models.model_config import LlamaConfig, Qwen2Config
from .weight_convert import (
    convert_llama_torch_to_litellama,
    convert_llavallama_hf_to_litellama,
    convert_qwen2_hf_to_litellama,
)
from ..kernels import update_kv_index
from ..utils.logger import log
from ..utils.common import get_model_dtype

def get_conversion_func(model_type: str):
    """
    根据模型类型获取相应的权重转换函数。

    参数:
        model_type (str): 模型类型。

    返回:
        function: 相应的权重转换函数，如果不支持则返回 None。
    """
    conversion_funcs = {
        "llama": convert_llama_torch_to_litellama,
        "qwen2": convert_qwen2_hf_to_litellama,
        "llava": convert_llavallama_hf_to_litellama,
    }
    return conversion_funcs.get(model_type.lower())


class ModelExecutor:
    # 定义类属性
    model_config = None
    model = None
    # model_runner_config = ModelRunnerConfig
    atten_info = AttentionInfo

    # 通过静态方法 build 将类属性当作默认配置使用
    @staticmethod
    def build(
        checkpoints_dir: str,
        max_seq_len: int,
        max_gpu_num_blocks: None,
        load_model: bool = True,
        triton_weight: bool = True,
        compiled_model: bool = False,
        device: str = "cuda",
    ):
        """
        构建 ModelExecutor 实例, 加载模型、分词器和初始化推理信息结构体 atten_info。

        参数:
            checkpoints_dir (str): 模型检查点目录路径。
            load_model (bool): 是否加载模型权重。
            max_seq_len (int): 最大序列长度。
            device (str): 设备类型（'cuda'或'cpu'）。

        返回:
            ModelExecutor: 初始化后的 ModelExecutor 实例。
        """
        model_config = ModelExecutor._load_model_config(
            checkpoints_dir, max_seq_len, device=device
        )
        # model = ModelExecutor._accelerate_load_weight(model_config, checkpoints_dir)
        model = ModelExecutor._load_model_weight(
            model_config, checkpoints_dir, load_model, triton_weight, device=device
        )  # 加载权重后的模型

        return ModelExecutor(
            model_config, model, max_gpu_num_blocks, compiled_model, device
        )

    @staticmethod
    def _accelerate_load_weight(
        model_config,
        checkpoints_dir,
        load_model=True,
        triton_weight=True,
        device="cuda",
    ):
        with init_empty_weights():
            model = ModelExecutor._initialize_model(model_config, device=device)

        # 假设 model 是使用 init_empty_weights 初始化的空模型
        model = load_checkpoint_and_dispatch(
            model, checkpoints_dir, device_map="auto", dtype=torch.float16
        )

        # 将模型转换为半精度, 并验证抓换
        model.to(device)
        model.half()
        for param in model.parameters():
            assert param.dtype == torch.float16, "Model parameters are not in FP16"
        log.info("Converted model to half precision (FP16)")

        return model

    @staticmethod
    def _load_model_weight(
        model_config,
        checkpoints_dir,
        load_model=True,
        triton_weight=True,
        device="cuda",
    ):
        start_time = time.time()
        hf_sd = None

        # 初始化模型
        with init_empty_weights():
            model = ModelExecutor._initialize_model(model_config, device=device)
            state_dict = None

        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, (
                f"no checkpoint files found in {checkpoints_dir}"
            )
            ckpt_path = str(checkpoints[0])
            log.debug("Type(ckpt_path) ", type(ckpt_path))
            log.info(f'Loading checkpoint "{ckpt_path}"')
            # 使用 torch.load 加载权重文件。torch.load 可以根据需要将权重加载到指定的设备上
            state_dict = torch.load(
                ckpt_path, mmap=True, weights_only=True, map_location=device
            )
        else:
            conversion_func = get_conversion_func(model_config.model_type)
            if conversion_func is None:
                log.error(f"Unsupported model type: {model_config.model_type}")
                raise ValueError(f"Unsupported model type: {model_config.model_type}")
            state_dict = conversion_func(checkpoints_dir, hf_sd, model_config)
            log.info(
                f"Weight conversion completed. Time elapsed: {time.time() - start_time:.2f} sec"
            )

        # Some checkpoints may use the HuggingFace naming style with a dot
        # before "weight" or "bias" for fused kv projections.  Rename these
        # keys to match our Module definition so that `load_state_dict` can
        # succeed without requiring an explicit conversion step.
        renamed_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if "kv_proj.weight" in k:
                new_key = k.replace("kv_proj.weight", "kv_proj_weight")
            elif "kv_proj.bias" in k:
                new_key = k.replace("kv_proj.bias", "kv_proj_bias")
            renamed_state_dict[new_key] = v
        state_dict = renamed_state_dict
        model.load_state_dict(
            state_dict, strict=True, assign=True
        )  # 将加载的 state_dict 应用到模型实例中。
        model.eval()
        log.info(f"Loaded state dict in {time.time() - start_time:.2f}s")

        # 将模型转换为半精度, 并验证转换
        model.half().to(device)
        for param in model.parameters():
            assert param.dtype == torch.float16, "Model parameters are not in FP16"
        log.info("Converted model to half precision (FP16)")

        return model

    @staticmethod
    def _initialize_model(model_config, device: str) -> nn.Module:
        """
        根据配置初始化模型并将其移动到指定设备。

        参数:
            model_config (LlamaConfig): 自定义模型的配置参数。
            device (str): 设备类型（'cuda'或'cpu'）。

        返回:
            nn.Module: 初始化后的模型。
        """
        model_type = model_config.model_type.lower()
        log.info(
            f"Initializing model of type '{model_type}' and moving it to device '{device}'..."
        )
        if model_type == "llama":
            from ..models.llama import LlamaModel

            model = LlamaModel(model_config)
        elif model_type == "qwen2":
            from ..models.qwen2 import Qwen2Model

            model = Qwen2Model(model_config)
        elif model_type == "llava":
            from ..models.llava import LlavaLlama

            model = LlavaLlama(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        log.info(f"The model has been initialized and moved to the device. '{device}'")
        return model

    @staticmethod
    def _load_model_config(checkpoints_dir, max_seq_len, device="cuda"):
        params_path = Path(checkpoints_dir) / "config.json"  # 定义模型配置文件
        assert params_path.exists(), f"config.json not found in {checkpoints_dir}"
        try:
            with open(params_path, "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            log.error(
                f"Configuration file '{params_path}' does not exist. Please check if the path is correct."
            )
            raise

        if params["model_type"] == "llama":
            model_config: LlamaConfig = LlamaConfig.from_dict(params)
        elif params["model_type"] == "qwen2":
            model_config: Qwen2Config = Qwen2Config(
                params, max_seq_len=max_seq_len, device=device
            )
        elif params["model_type"] == "llava":
            model_config = LlavaConfig.from_pretrained(checkpoints_dir)

        return model_config

    def __init__(
        self,
        model_config,
        model,
        max_gpu_num_blocks=None,
        compiled_model=False,
        device="cuda",
    ):
        self.model_config = model_config
        self.device = device
        if isinstance(model_config, LlavaConfig):
            self.llm_config = LlamaConfig.from_dict(model_config.text_config.to_dict())
            print(f"self.llm_config.max_seq_len: {self.llm_config.max_seq_len}")
        else:
            self.llm_config = model_config

        self.max_seq_len = self.llm_config.max_seq_len
        self.model_type = model_config.model_type
        self.model = model
        self.model_runner = None

        if max_gpu_num_blocks:
            self.kv_mem_manager = self._init_mem_manager(max_gpu_num_blocks)
            self.max_gpu_num_tokens = max_gpu_num_blocks
        else:
            max_gpu_num_blocks, self.max_gpu_num_tokens = (
                self._get_max_avaliable_tokens(gpu_memory_utilization=0.9, block_size=1)
            )
            self.kv_mem_manager = self._init_mem_manager(
                max_gpu_num_blocks, block_size=1
            )

        self.max_request_num = max_gpu_num_blocks // self.max_seq_len

        self.req_tokens_manager = ReqTokensManager(
            self.max_request_num, self.max_seq_len
        )
        self.atten_info = AttentionInfo()  # 创建 AttentionInfo 实例
        self.atten_info.kv_buffer = self.kv_mem_manager.gpu_kv_buffer
        self.atten_info.b_req_tokens_table = self.req_tokens_manager.b_req_tokens_table

        # TODO apply_cuda_graph 新代码有 bug，已经删去，后续等待修复
        self.compiled_model = False
        if self.compiled_model:
            self.apply_cuda_graph()  # 调用 cuda graph 优化

    def _get_max_avaliable_tokens(self, gpu_memory_utilization=0.9, block_size=1):
        avaliable_blocks = ComputeMaxAvailableBlocks(
            num_layers=self.llm_config.num_layers,
            hidden_size=self.llm_config.hidden_size,
            num_heads=self.llm_config.num_heads,
            num_kv_heads=self.llm_config.num_kv_heads,
            gpu_memory_utilization=gpu_memory_utilization,
            block_size=block_size,
        )
        max_gpu_num_blocks = avaliable_blocks.compute_num_available_blocks()
        max_gpu_num_tokens = max_gpu_num_blocks * block_size

        return max_gpu_num_blocks, max_gpu_num_tokens

    def _init_mem_manager(
        self, gpu_num_blocks, block_size=1, dtype=torch.float16, device="cuda"
    ):
        kv_mem_manager = KVCacheMemoryManager(
            num_layers=self.llm_config.num_layers,
            num_kv_heads=self.llm_config.num_kv_heads,
            head_dim=self.llm_config.head_dim,
            gpu_num_blocks=gpu_num_blocks,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )

        return kv_mem_manager

    def apply_cuda_graph(
        self,
    ):
        """应用 cuda graph 优化
        参数:
            - input_ids: 输入 tokens id 列表, shape: (batch_size, 1)
            - prev_pos: 当前处于第几轮迭代循环, 生成第几个 token
        """
        self.model_runner = ModelRunner(
            self.model,
            self.llm_config,
            self.max_gpu_num_tokens,
            self.kv_mem_manager,
            self.req_tokens_manager,
        )
        self.model_runner.capture_decode_graph()

    def init_req_to_tokens_table(
        self, b_req_tokens_table, b_req_idx, b_seq_len, alloc_mem_index
    ):
        """
        初始化 prefill 阶段已分配的批次请求项的 kv cache 所用 tokens 索引
        """
        # TODO: 性能等待优化
        start_index = 0
        batch_size = len(b_seq_len)
        b_seq_len_numpy = b_seq_len.cpu().numpy()
        b_req_idx_numpy = b_req_idx.cpu().numpy()
        b_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device=self.device)
        for i in range(batch_size):
            if i > 0:
                b_start_loc[i] = start_index
            cur_seq_len = b_seq_len_numpy[i]
            b_req_tokens_table[b_req_idx_numpy[i], :cur_seq_len] = alloc_mem_index[
                start_index : start_index + cur_seq_len
            ]
            start_index += cur_seq_len

        return b_start_loc

    def prefill_alloc_kv_cache(
        self,
        max_prompt_len,
        actual_prompt_lens,
        b_req_idx,
        image_batch_size=None,
        debug_mode=False,
    ):
        """
        start_index:        tensor([  0, 270, 540, 810], device='cuda:0', dtype=torch.int32)
        b_seq_len:          tensor([14, 12, 11, 11], device='cuda:0')
        Prefill Stage, cur_select_index: tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                                    270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283,
                                    540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553,
                                    810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823
                                ], device='cuda:0')
        Decode Stage, 0 step, cur_select_index: tensor([ 14, 282, 551, 821], device='cuda:0'), cur_b_seq_len: tensor([15, 13, 12, 12], device='cuda:0')
        Decode Stage, 1 step, cur_select_index: tensor([ 15, 283, 552, 822], device='cuda:0'), cur_b_seq_len: tensor([16, 14, 13, 13], device='cuda:0')
        """
        num_patch_indexs = None
        batch_size = len(actual_prompt_lens)
        self.atten_info.b_req_idx = b_req_idx

        if image_batch_size is not None:
            image_size = self.model_config.vision_config.image_size
            pathch_size = self.model_config.vision_config.patch_size
            number_patchs = image_size // pathch_size
            num_patch_indexs = number_patchs * number_patchs - 1
            max_prompt_len += num_patch_indexs
            actual_prompt_lens += num_patch_indexs
            print(f"num_patch_indexs: {num_patch_indexs}")

        context_num_tokens = max_prompt_len * batch_size
        # 一次性分配 bsz * seq_len + (number_patchs * number_patchs - 1) * img_batch_size 个索引
        self.atten_info.cur_select_index, _ = self.kv_mem_manager.alloc_kvcache_index(
            context_num_tokens
        )
        # 初始化每个批次项的实际提示词长度
        self.atten_info.b_seq_len = actual_prompt_lens  # 张量, 形状 [batch_size, 1]
        # 初始化批次请求的当前最大序列上下文长度(对应 kv cache 长度)
        self.atten_info.max_actual_seq_len = max_prompt_len  # int 类型

        self.atten_info.b_start_loc = self.init_req_to_tokens_table(
            self.atten_info.b_req_tokens_table,
            self.atten_info.b_req_idx,
            self.atten_info.b_seq_len,
            self.atten_info.cur_select_index,
        )

        if debug_mode:
            print(
                f"context_num_tokens: {context_num_tokens}, max_prompt_len:{max_prompt_len}, \n \
                    self.atten_info.cur_select_index: {self.atten_info.cur_select_index},\n \
                    self.atten_info.max_actual_seq_len: {self.atten_info.max_actual_seq_len},\n \
                    self.atten_info.b_seq_len: {self.atten_info.b_seq_len}, \n \
                    self.atten_info.b_start_loc: {self.atten_info.b_start_loc}, "
            )

        return self.atten_info.cur_select_index, num_patch_indexs

    def decode_alloc_kv_cache(self, batch_size):
        # TODO: torch.empty 创建的临时张量, 保存分配的非连续 kv_cache 索引空间
        self.atten_info.cur_select_index, _ = self.kv_mem_manager.alloc_kvcache_index(
            batch_size
        )
        update_kv_index(
            self.atten_info.b_req_tokens_table,
            self.atten_info.b_req_idx,
            self.atten_info.b_seq_len,
            self.atten_info.cur_select_index,
        )

        self.atten_info.b_seq_len += 1
        self.atten_info.max_actual_seq_len += 1

        return self.atten_info.cur_select_index  # shape [batch_size,]

    def forward(self, input_ids, position_ids, image_tensor=None):
        if self.model_type == "llava":
            logits = self.model.forward(
                input_ids, position_ids, self.atten_info, image_tensor
            )
        else:
            logits = self.model.forward(input_ids, position_ids, self.atten_info)
        return logits
