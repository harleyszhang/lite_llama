from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict, Optional

@dataclass
class LlamaConfig:
    architectures: Optional[list] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    
    # dim: Optional[int] = None
    initializer_range: float = 0.02
    
    # 模型隐藏层大小
    hidden_size: Optional[int] = 2048
    intermediate_size: Optional[int] = 8192
    max_position_embeddings: Optional[int] = None
    mlp_bias: bool = False
    model_type: str = "llama"
    # 注意力头数，也就是 q heads 头数
    num_heads: Optional[int] = None
    # 解码层数
    num_layers: Optional[int] = None
    # 使用了 GQA 技术的 kv heads 头数
    num_kv_heads: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-5
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: Optional[str] = None
    use_cache: bool = True
    vocab_size: Optional[int] = None
    max_batch_size: int = 4
    max_seq_len: int = 2048
    device: str = "cuda"

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, **kwargs):
        
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