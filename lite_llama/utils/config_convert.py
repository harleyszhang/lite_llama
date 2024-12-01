import sys, os
import transformers
from transformers import LlavaNextConfig, LlavaConfig

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from ..models.model_config import LlamaConfig


def convert_transformers_to_custom_config(
    transformers_config: transformers.LlamaConfig
) -> LlamaConfig:
    # 将 transformers 配置转换为字典
    config_dict = transformers_config.to_dict()
    print("transformers.LlamaConfig dict: ", config_dict)
    
    return LlamaConfig(
            _name_or_path=config_dict.get("_name_or_path"),
            architectures=config_dict.get("architectures", ["LlamaForCausalLM"]),
            max_position_embeddings=config_dict.get("max_position_embeddings", 4096),
            model_type=config_dict.get("model_type", "llama"),
            rms_norm_eps=config_dict.get("rms_norm_eps", 1e-5),
            torch_dtype=config_dict.get("torch_dtype", "float16"),
            vocab_size=config_dict.get("vocab_size", 32064),
            
            hidden_size = config_dict.get("hidden_size", 4096),
            intermediate_size = config_dict.get("intermediate_size", 11008),
            num_hidden_layers = config_dict.get("num_hidden_layers", 32),
            num_attention_heads = config_dict.get("num_attention_heads", 32),
            num_key_value_heads = config_dict.get("num_key_value_heads", None),
        )

    # 创建自定义配置实例
    custom_config = LlamaConfig(config_dict=config_dict)

    return custom_config

if __name__ == "__main__":
    # 加载 transformers 的 LlamaConfig（请替换为实际模型名称）
    model_path = '/gemini/code/liuhaotian/llava-v1.5-7b'
    transformers_config = LlavaConfig.from_pretrained(model_path)

    # 转换为自定义配置
    custom_llama_config = convert_transformers_to_custom_config(transformers_config.text_config)

    # 打印自定义配置
    # print(json.dumps(custom_llama_config, indent=4, ensure_ascii=False))
    print(custom_llama_config)

"""
lamaConfig(architectures=None, attention_bias=False, attention_dropout=0.0, bos_token_id=1, eos_token_id=2, head_dim=128, hidden_act='silu', 
initializer_range=0.02, hidden_size=4096, intermediate_size=11008, max_position_embeddings=2048, mlp_bias=False, model_type='llama', 
num_heads=32, num_layers=32, num_kv_heads=32, pretraining_tp=1, rms_norm_eps=1e-06, rope_scaling=None, rope_theta=10000.0, 
tie_word_embeddings=False, torch_dtype=None, transformers_version='4.40.2', use_cache=True, vocab_size=32000, max_batch_size=4,
max_seq_len=2048, device='cuda')
"""