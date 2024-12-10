import torch

# 定义矩阵
input = torch.tensor([1.0, 2.0])
mat1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
mat2 = torch.tensor([[9.0, 10.0], [11.0, 12.0]])

# 使用 torch.addmm 计算
result = torch.addmm(input, mat1, mat2)

print("结果：\n", result)