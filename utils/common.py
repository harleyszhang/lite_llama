import json
import time, os
import subprocess
from typing import List, Optional

def read_json(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    return data

def read_jsonl(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def detect_device():
    try:
        subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL)
        return "nvidia"
    except:
        try:
            subprocess.check_output(['rocm-smi'], stderr=subprocess.DEVNULL)
            return "amd"
        except:
            return "cpu"

def getTime():
    return str(time.strftime("%m-%d %H:%M:%S", time.localtime()))

def getProjectPath():
    script_path = os.path.split(os.path.realpath(__file__))[0]
    return os.path.abspath(os.path.join(script_path, ".."))


def get_gpu_memory(gpu_type="amd", device_id="0"):
    try:
        if gpu_type == "amd":
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", device_id],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            for line in result.stdout.splitlines():
                if "VRAM Total Used Memory" in line:
                    used = line.split(":")[-1].strip().split()[0]
                    return float(used) / (10 ** 9) # Convert MiB to GiB
        elif gpu_type == "nvidia":
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader", "-i", device_id],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return float(result.stdout.strip()) / 1024  # Convert MiB to GiB
        elif gpu_type == "cpu":
            return None
    except Exception as e:
        from utils.logger import log
        log.warning(f"Unable to fetch GPU memory: {e}")
        return None

def count_tokens(texts: List[str], tokenizer) -> int:

    total_tokens = 0
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        total_tokens += len(ids)
    return total_tokens

def get_model_type(checkpoint_path: str) -> str | None:
    from utils.logger import log
    model_type = ["llama", "falcon", "mpt", "qwen2", "llava"]

    config_content = read_json(os.path.join(checkpoint_path, "config.json"))
    for m in model_type:
        if m in config_content["model_type"].lower():
            if m == "llava":
                return "llama"
            return m
    log.error(f"No model type found: {checkpoint_path}")
    return None
