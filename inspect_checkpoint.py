import torch
import sys
from pathlib import Path
from safetensors.torch import load_file

def inspect(path):
    p = Path(path)
    if not p.exists():
        print(f"Path not found: {p}")
        return

    # 找第一个权重文件
    files = list(p.glob("*.safetensors")) + list(p.glob("*.bin")) + list(p.glob("*.pt")) + list(p.glob("*.pth"))
    if not files:
        print("No weight files found.")
        return
    
    target = files[0]
    print(f"Inspecting: {target.name}")
    
    try:
        if target.suffix == ".safetensors":
            sd = load_file(target)
        else:
            sd = torch.load(target, map_location="cpu")
            
        print("-" * 50)
        keys = list(sd.keys())
        for k in keys[:]:
            print(k)
        print("-" * 50)
        print(f"Total keys: {len(keys)}")
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py /path/to/weights")
    else:
        inspect(sys.argv[1])