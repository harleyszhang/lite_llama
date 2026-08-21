"""Derive a short model name from a checkpoint path.

Normalises trailing slashes and returns the final path component, used to select the
matching prompter and stop-token set.

Usage:
    name = get_model_name_from_path("my_weight/Qwen2.5-0.5B")
"""

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
