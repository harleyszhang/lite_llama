import torch

# 执行一些操作
linear1 = torch.nn.Linear(in_features=2048, out_features=2048)

# 重置内存峰值统计
torch.cuda.reset_peak_memory_stats()

# 执行后续操作
linear3 = torch.nn.Linear(in_features=512, out_features=512)

# 获取当前内存统计信息, Get the peak memory allocation recorded by torch.
stats = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
print(stats)