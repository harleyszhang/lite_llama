from transformers import LlavaConfig, AutoTokenizer


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


if __name__ == "__main__":
    model_path = "/gemini/code/lite_llama/my_weight/llava-1.5-7b-hf"
    print(get_model_name_from_path(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print(tokenizer)
