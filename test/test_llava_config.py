import json, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from lite_llama.models.model_config import LlavaConfig

def test_llava_config():
    # 示例配置 JSON 字符串
    config_json = """
    {
      "architectures": [
        "LlavaForConditionalGeneration"
      ],
      "ignore_index": -100,
      "image_token_index": 32000,
      "model_type": "llava",
      "pad_token_id": 32001,
      "projector_hidden_act": "gelu",
      "text_config": {
        "_name_or_path": "lmsys/vicuna-7b-v1.5",
        "architectures": [
          "LlamaForCausalLM"
        ],
        "max_position_embeddings": 4096,
        "model_type": "llama",
        "rms_norm_eps": 1e-05,
        "torch_dtype": "float16",
        "vocab_size": 32064
      },
      "tie_word_embeddings": false,
      "torch_dtype": "float16",
      "transformers_version": "4.36.0.dev0",
      "vision_config": {
        "hidden_size": 1024,
        "image_size": 336,
        "intermediate_size": 4096,
        "model_type": "clip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "patch_size": 14,
        "projection_dim": 768,
        "vocab_size": 32000
      },
      "vision_feature_layer": -2,
      "vision_feature_select_strategy": "default",
      "vocab_size": 32064
    }
    """

    # 将 JSON 字符串解析为字典
    config_dict = json.loads(config_json)

    # 从字典创建 LlavaConfig 实例
    llava_config = LlavaConfig.from_dict(config_dict)

    # 打印配置以验证
    print(llava_config)

if __name__ == "__main__":
    test_llava_config()