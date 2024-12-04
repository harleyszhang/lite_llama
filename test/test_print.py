from rich.console import Console
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