from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import LlavaConfig
import sys, os

# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lite_llama.models.llava import LlavaLlama

hf_model_path = "/gemini/code/liuhaotian/llava-v1.5-7b"


def test_LlavaLlama_structure(hf_model_path):
    # 使用 init_empty_weights 初始化空模型
    with init_empty_weights():
        config = LlavaConfig.from_pretrained(hf_model_path)
        model = LlavaLlama(config)

        # 打印没有加载权重的 LlavaLlama 模型结构
        print(model)
        # 打印模型的简单摘要
        print(f"模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

        # 可选择打印部分参数信息
        for name, param in list(model.named_parameters())[:]:  # 打印模型参数
            print(name, param.shape)


if __name__ == "__main__":
    test_LlavaLlama_structure(hf_model_path)
