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
