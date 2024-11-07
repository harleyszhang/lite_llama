import json
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    architectures: Optional[list] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    # 模型隐藏层大小
    dim: Optional[int] = None
    initializer_range: float = 0.02
    intermediate_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    mlp_bias: bool = False
    model_type: str = "llama"
    # 注意力头数，也就是 q heads 头数
    n_heads: Optional[int] = None
    # 解码层数
    n_layers: Optional[int] = None
    # 使用了 GQA 技术的 kv heads 头数
    n_kv_heads: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-5
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    torch_dtype: str = "float32"
    transformers_version: Optional[str] = None
    use_cache: bool = True
    vocab_size: Optional[int] = None
    max_batch_size: int = 4
    max_seq_len: int = 2048

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, **kwargs):
        # 首先，设置默认属性值
        for field_name, field_def in self.__dataclass_fields__.items():
            setattr(self, field_name, field_def.default)

        # 如果提供了 config_dict，从中更新属性
        if config_dict is not None:
            for key, value in config_dict.items():
                # 处理名称映射
                if key == 'hidden_size':
                    self.dim = value
                elif key == 'num_attention_heads':
                    self.n_heads = value
                elif key == 'num_hidden_layers':
                    self.n_layers = value
                elif key == 'num_key_value_heads':
                    self.n_kv_heads = value
                else:
                    setattr(self, key, value)

        # 处理额外的关键字参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 如果属性不存在，可以选择存储在 extra_args 中，或者直接添加
                setattr(self, key, value)

def load_config_from_json(json_file_path: str) -> LlamaConfig:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config = LlamaConfig(config_dict, max_seq_len = 2048)
    return config

if __name__ == "__main__":
    # 创建 LlamaConfig 实例，设置 max_batch_size=16
    config = LlamaConfig(max_batch_size=16)
    print("max_batch_size:", config.max_batch_size)

    # JSON 文件的路径
    json_file_path = '/gemini/code/Llama-3.2-1B-Instruct/config.json'

    # 加载配置
    config = load_config_from_json(json_file_path)

    # 访问配置参数
    print("模型类型:", config.model_type)
    print("隐藏层数 (n_layers):", config.n_layers)
    print("隐藏大小 (dim):", config.dim)
    print("词汇表大小:", config.vocab_size)
    print("旋转位置编码配置:", config.rope_scaling)
    print("最大支持序列长度:", config.max_seq_len)
    print("模型层数", config.n_layers)
    if config.rope_scaling is not None:
        print("rope 类型:", config.rope_scaling.get("rope_type"))
    else:
        print("rope_scaling is None")
