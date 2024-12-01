import sys, os

# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lite_llama.models.llava import LlavaLlama

if __name__ == "__main__":
    model_path = "/gemini/code/llm_weights/llava-hf/llava-1.5-7b-hf"
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import LlavaConfig

    # 使用 init_empty_weights 初始化空模型
    with init_empty_weights():
        llava_config = LlavaConfig.from_pretrained(model_path)
        # print(llava_config) # 打印配置以验证

        model = LlavaLlama(llava_config)        
        print(model) # 打印模型结构
        for name, param in list(model.named_parameters())[:]:  # 打印模型参数
            print(name, param.shape)