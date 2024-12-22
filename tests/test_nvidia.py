import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    compute_capability = torch.cuda.get_device_capability(device)
    print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
else:
    print("CUDA device not available.")

from rich.console import Console
from rich.prompt import Prompt
import time

console = Console()

# 使用 print
print("Loading ", end='', flush=True)
for _ in range(3):
    time.sleep(1)
    print("hello", end='', flush=True)
print(" Done!")

# 使用 console.print
console.print("\nLoading", end=' ')
for _ in range(3):
    time.sleep(1)
    console.print("[bold blue]hello[/bold blue]", end=' ')
console.print("Done!")

Prompt.ask("[bold green]提示词[/bold green]").strip()
print("\033[91mHello, World!\033[0m")  # 红色文本

import torch

# 定义矩阵
input = torch.tensor([1.0, 2.0])
mat1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
mat2 = torch.tensor([[9.0, 10.0], [11.0, 12.0]])

# 使用 torch.addmm 计算
result = torch.addmm(input, mat1, mat2)

print("结果：\n", result)