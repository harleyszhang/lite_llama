from dataclasses import dataclass,field, fields
from typing import Any, Dict, List, Optional, Tuple, Union
import os, json


@dataclass
class LlamaConfig:
    architectures: List[str] = field(default_factory=lambda: ["LlamaForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    
    initializer_range: float = 0.02
    
    hidden_size: int = 2048  # 默认值调整为2048，保持一致性
    intermediate_size: int = 8192
    max_position_embeddings: Optional[int] = None
    mlp_bias: bool = False
    model_type: str = "llama"
    num_heads: Optional[int] = 32  # 设置合理的默认值
    num_layers: Optional[int] = 32
    num_kv_heads: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-5
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: Optional[str] = None
    use_cache: bool = True
    vocab_size: Optional[int] = 32064

    _name_or_path: Optional[str] = None
    max_batch_size: int = 64
    max_seq_len: int = 2048
    device: str = "cuda"

    def __post_init__(self):
        if self.num_heads and self.hidden_size:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = None  # 或者设置一个默认值，例如 64

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LlamaConfig':
        # 定义字段映射
        key_mappings = {
            'num_attention_heads': 'num_heads',
            'num_hidden_layers': 'num_layers',
            'num_key_value_heads': 'num_kv_heads',
        }

        # 创建一个复制的字典，以避免修改原始数据
        data_copy = data.copy()

        # 应用字段映射
        for old_key, new_key in key_mappings.items():
            if old_key in data_copy:
                data_copy[new_key] = data_copy.pop(old_key)

        # 获取 LlamaConfig 类中所有的字段名
        valid_keys = {f.name for f in fields(cls)}

        # 过滤掉不在 LlamaConfig 中定义的键
        data_filtered = {k: v for k, v in data_copy.items() if k in valid_keys}

        # 设置默认值，确保所有必要字段都有值
        defaults = {
            'architectures': ["LlamaForCausalLM"],
            'attention_bias': False,
            'attention_dropout': 0.0,
            'bos_token_id': None,
            'eos_token_id': None,
            'hidden_act': "silu",
            'initializer_range': 0.02,
            'hidden_size': 2048,
            'intermediate_size': 8192,
            'max_position_embeddings': None,
            'mlp_bias': False,
            'model_type': "llama",
            'num_heads': 32,
            'num_layers': 32,
            'num_kv_heads': None,
            'pretraining_tp': 1,
            'rms_norm_eps': 1e-5,
            'rope_scaling': None,
            'rope_theta': 10000.0,
            'tie_word_embeddings': True,
            'torch_dtype': "bfloat16",
            'transformers_version': None,
            'use_cache': True,
            'vocab_size': 32064,
            '_name_or_path': None,
            'max_batch_size': 64,
            'max_seq_len': 2048,
            'device': "cuda",
        }

        # 更新缺失的字段
        for key, value in defaults.items():
            data_filtered.setdefault(key, value)

        return cls(**data_filtered)
    
@dataclass
class Qwen2Config:
    max_batch_size: int = 4
    max_seq_len: int = 2048

    architectures: Optional[list] = None
    attention_dropout: float = 0.0
    bos_token_id: Optional[int] = 151643
    eos_token_id: Optional[int] = 151645
    hidden_act: str = "silu"
    
    # dim: Optional[int] = None
    initializer_range: float = 0.02
    
    # 模型隐藏层大小, Qwen2.5-1.5B-Instruct
    hidden_size: Optional[int] = 1536
    intermediate_size: Optional[int] = 8960
    max_position_embeddings: Optional[int] = 32768

    mlp_bias: bool = False
    model_type: str = "qwen2"
    # 注意力头数，也就是 q heads 头数
    num_heads: Optional[int] = 12
    # 解码层数
    num_layers: Optional[int] = 28
    # 使用了 GQA 技术的 kv heads 头数
    num_kv_heads: Optional[int] = 2

    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 1000000.0

    torch_dtype: str = "bfloat16"
    transformers_version: Optional[str] = "4.43.1"
    use_cache: bool = True
    vocab_size: Optional[int] = 151936

    tie_word_embeddings: bool = False
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 21
    device: str = "cuda"

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None

        # 首先，设置默认属性值
        for field_name, field_def in self.__dataclass_fields__.items():
            setattr(self, field_name, field_def.default)

        # 如果提供了 config_dict，从中更新属性
        if config_dict is not None:
            for key, value in config_dict.items():
                # 处理名称映射
                if key == 'num_attention_heads':
                    self.num_heads = value
                elif key == 'num_hidden_layers':
                    self.num_layers = value
                elif key == 'num_key_value_heads':
                    self.num_kv_heads = value
                elif key == 'max_length':
                    self.max_seq_len = value
                else:
                    setattr(self, key, value)

        # 处理额外的关键字参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 如果属性不存在，可以选择存储在 extra_args 中，或者直接添加
                setattr(self, key, value)
        self.head_dim = self.hidden_size // self.num_heads
    
@dataclass
class CLIPVisionConfig():
    """
    This is the configuration class to store the configuration of a [`CLIPVisionModel`]. It is used to instantiate a
    CLIP vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.
    """
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    projection_dim: int = 512,
    num_layers: int = 12, # encoder_layer 层数
    num_heads: int = 12,  # attention 模块的头数目
    num_channels: int = 3,
    image_size: int = 224,
    patch_size: int = 32,
    hidden_act: int = "quick_gelu",
    layer_norm_eps: int = 1e-5,
    attention_dropout: int = 0.0,
    initializer_range: int = 0.02,
    initializer_factor: int = 1.0,

    model_type: str = "clip_vision_model"

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None

        # 首先，设置默认属性值
        for field_name, field_def in self.__dataclass_fields__.items():
            setattr(self, field_name, field_def.default)

        # 如果提供了 config_dict，从中更新属性, clip 模型的配置文件包含 text 和 vision 配置
        if config_dict is not None:
            # get the vision config dict if we are loading from CLIPConfig
            if config_dict.get("model_type") == "clip":
                config_dict = config_dict["vision_config"]
            else:
                print("Error! clip model config file not include vision config!")

            for key, value in config_dict.items():
                # 处理名称映射
                if key == 'num_attention_heads':
                    self.num_heads = value
                elif key == 'num_hidden_layers':
                    self.num_layers = value
                elif key == 'num_key_value_heads':
                    self.num_kv_heads = value
                else:
                    setattr(self, key, value)

        # 处理额外的关键字参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 如果属性不存在，可以选择存储在 extra_args 中，或者直接添加
                setattr(self, key, value)
        self.head_dim = self.hidden_size // self.num_heads


@dataclass
class VisionConfig:
    hidden_size: int
    image_size: int
    intermediate_size: int
    model_type: str
    num_attention_heads: int
    num_hidden_layers: int
    patch_size: int
    projection_dim: int
    vocab_size: int

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'VisionConfig':
        return VisionConfig(
            hidden_size=data.get("hidden_size", 768),
            image_size=data.get("image_size", 224),
            intermediate_size=data.get("intermediate_size", 3072),
            model_type=data.get("model_type", "clip_vision_model"),
            num_attention_heads=data.get("num_attention_heads", 12),
            num_hidden_layers=data.get("num_hidden_layers", 12),
            patch_size=data.get("patch_size", 16),
            projection_dim=data.get("projection_dim", 768),
            vocab_size=data.get("vocab_size", 1000)
        )

@dataclass
class LlavaConfig:
    architectures: List[str]
    ignore_index: int
    image_token_index: int
    model_type: str
    pad_token_id: int
    projector_hidden_act: str
    text_config: LlamaConfig
    tie_word_embeddings: bool
    torch_dtype: str
    transformers_version: str
    vision_config: VisionConfig
    vision_feature_layer: int
    vision_feature_select_strategy: str
    vocab_size: int
    image_seq_length: int = 576
    max_batch_size: int = 64
    max_seq_len: int = 10240
    device: str = "cuda"
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'LlavaConfig':
        text_cfg = LlamaConfig.from_dict(data.get("text_config", {}))
        vision_cfg = VisionConfig.from_dict(data.get("vision_config", {}))
        return LlavaConfig(
            architectures=data.get("architectures", []),
            ignore_index=data.get("ignore_index", -100),
            image_token_index=data.get("image_token_index", 32000),
            model_type=data.get("model_type", "llava"),
            pad_token_id=data.get("pad_token_id", 32001),
            projector_hidden_act=data.get("projector_hidden_act", "gelu"),
            text_config=text_cfg,
            tie_word_embeddings=data.get("tie_word_embeddings", False),
            torch_dtype=data.get("torch_dtype", "float16"),
            transformers_version=data.get("transformers_version", "4.36.0.dev0"),
            vision_config=vision_cfg,
            vision_feature_layer=data.get("vision_feature_layer", -2),
            vision_feature_select_strategy=data.get("vision_feature_select_strategy", "default"),
            vocab_size=data.get("vocab_size", 32064)
        )

    @classmethod
    def from_json(cls, json_path: str) -> 'LlavaConfig':
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        """类方法，用于从指定的 JSON 文件中读取数据并将其解析为字典对象"""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        
        # NOTE: 使用 json.loads 函数将读取到的 JSON 格式字符串解析为 Python 字典对象
        return json.loads(text)
