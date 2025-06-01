import json
import time, os
import subprocess
from typing import List, Optional
import torch

def read_json(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    return data


def read_jsonl(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def detect_device():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return "nvidia"
    except:
        try:
            subprocess.check_output(["rocm-smi"], stderr=subprocess.DEVNULL)
            return "amd"
        except:
            return "cpu"


def getTime():
    return str(time.strftime("%m-%d %H:%M:%S", time.localtime()))


def getProjectPath():
    script_path = os.path.split(os.path.realpath(__file__))[0]
    return os.path.abspath(os.path.join(script_path, ".."))


def get_gpu_memory(gpu_type, device_id="0"):
    try:
        if gpu_type == "amd":
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", device_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for line in result.stdout.splitlines():
                if "VRAM Total Used Memory" in line:
                    used = line.split(":")[-1].strip().split()[0]
                    return float(used) / (10**9)  # Convert MiB to GiB
        elif gpu_type == "nvidia":
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,nounits,noheader",
                    "-i",
                    device_id,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return float(result.stdout.strip()) / 1024  # Convert MiB to GiB
        elif gpu_type == "cpu":
            return None
    except Exception as e:
        from lite_llama.utils.logger import log

        log.warning(f"Unable to fetch GPU memory: {e}")
        return None


def count_tokens(texts: List[str], tokenizer) -> int:
    total_tokens = 0
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        total_tokens += len(ids)
    return total_tokens


def get_model_type(checkpoint_path: str) -> str | None:
    from .logger import log

    model_type = ["llama", "falcon", "mpt", "qwen2", "llava"]

    config_content = read_json(os.path.join(checkpoint_path, "config.json"))
    for m in model_type:
        if m in config_content["model_type"].lower():
            if m == "llava":
                return "llama"
            return m
    log.error(f"No model type found: {checkpoint_path}")
    return None


def check_model_compatibility(model_path):
    """Check if the model is compatible for quantization"""
    # Check if model path exists and contains .pth files
    if not os.path.exists(model_path):
        return False, f"Model path does not exist: {model_path}"

    pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
    if not pth_files:
        return False, f"No .pth files found in {model_path}"

    # Check if required config files exist
    config_files = ["config.json", "tokenizer_config.json"]
    missing_configs = [f for f in config_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_configs:
        print(f"Warning: Missing config files: {missing_configs}")

    return True, "Model is compatible"


def get_model_info(model_path):
    """Get basic information about the model"""
    model_info = {
        "model_name": os.path.basename(model_path),
        "model_type": "unknown",
        "size": 0.0
    }

    # Detect model type from path or config
    model_name_lower = model_info["model_name"].lower()
    if "llava" in model_name_lower:
        model_info["model_type"] = "llava"
    elif "qwen2" in model_name_lower:
        model_info["model_type"] = "qwen2"
    elif "llama" in model_name_lower:
        model_info["model_type"] = "llama"

    # Try to read from config.json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "architectures" in config:
                    arch = config["architectures"][0].lower()
                    if "llava" in arch:
                        model_info["model_type"] = "llava"
                    elif "qwen2" in arch:
                        model_info["model_type"] = "qwen2"
                    elif "llama" in arch:
                        model_info["model_type"] = "llama"
        except:
            pass

    # Calculate total size
    total_size = 0
    for f in os.listdir(model_path):
        if f.endswith('.pth'):
            file_path = os.path.join(model_path, f)
            total_size += os.path.getsize(file_path)

    model_info["size"] = total_size / (1024 ** 3)  # Convert to GB

    return model_info


def get_model_dtype(checkpoints_dir: str):
    """
    Get the model dtype from config.json

    Args:
        checkpoints_dir: Path to model checkpoint directory

    Returns:
        torch.dtype or str: The dtype specified in config.json
    """
    config_path = os.path.join(checkpoints_dir, "config.json")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        torch_dtype_str = config.get("torch_dtype", "float16").lower()

        # Map string to torch dtype or string identifiers for quantized formats
        dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float": torch.float32,
            "int8": torch.int8,
            "int4": "int4",  # Placeholder, since PyTorch doesn't natively support int4
        }

        dtype = dtype_mapping.get(torch_dtype_str, torch.float16)
        print(f"Detected model dtype from config: {torch_dtype_str} -> {dtype}")

        return dtype

    except Exception as e:
        print(f"Warning: Could not read dtype from config.json: {e}")
        print("Defaulting to torch.float16")
        return torch.float16

    except Exception as e:
        print(f"Warning: Could not read dtype from config.json: {e}")
        print("Defaulting to torch.float16")
        return torch.float16

