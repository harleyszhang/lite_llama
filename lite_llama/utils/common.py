import json
import time, os
import subprocess
from typing import List, Optional


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


def check_model_compatibility(model_path):
    """Check if the model is a valid converted lite_llama model"""
    # Check for required files
    required_files = []
    optional_files = ["config.json", "tokenizer.model", "tokenizer.json"]

    # Find .pth file
    pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
    if not pth_files:
        return False, "No .pth file found in the model directory"

    # Check if already quantized
    if any('gptq' in f for f in pth_files):
        return False, "Model appears to be already quantized"

    # Check for config files
    found_configs = []
    for config_file in optional_files:
        if os.path.exists(os.path.join(model_path, config_file)):
            found_configs.append(config_file)

    if not found_configs:
        return False, "No configuration files found (config.json, tokenizer.json, etc.)"

    return True, "Model is compatible"


def get_model_info(model_path):
    """Extract model information from the directory"""
    info = {
        "model_name": os.path.basename(model_path),
        "model_type": "unknown",
        "size": 0
    }

    # Try to determine model type from name
    model_name_lower = info["model_name"].lower()
    if "llava" in model_name_lower:
        info["model_type"] = "llava"
    elif "qwen2" in model_name_lower:
        info["model_type"] = "qwen2"
    elif "llama" in model_name_lower:
        info["model_type"] = "llama"

    # Calculate model size
    pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
    if pth_files:
        model_file = os.path.join(model_path, pth_files[0])
        info["size"] = os.path.getsize(model_file) / (1024 ** 3)  # Size in GB

    return info