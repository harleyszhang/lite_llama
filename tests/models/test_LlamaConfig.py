import json, os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from lite_llama.models.model_config import LlamaConfig


def load_config_from_json(json_file_path: str) -> LlamaConfig:
    with open(json_file_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = LlamaConfig(config_dict, max_seq_len=2048)
    return config


if __name__ == "__main__":
    # 创建 LlamaConfig 实例，设置 max_batch_size=16
    config = LlamaConfig(max_batch_size=16)
    print("max_batch_size:", config.max_batch_size)

    # JSON 文件的路径
    json_file_path = "/gemini/code/Llama-3.2-1B-Instruct/config.json"

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
