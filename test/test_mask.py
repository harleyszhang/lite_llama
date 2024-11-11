import torch 

seq_len = 4
start_pos = 0

mask = torch.full((seq_len, seq_len), float("-inf"))
print(mask)
mask1 = torch.triu(mask, diagonal=1) # 创建上三角矩阵
print(mask1)
mask2 = torch.hstack([torch.zeros((seq_len, start_pos)), mask1])

print(mask2)
print("mask shape is ", mask.shape)

# # When performing key-value caching, we compute the attention scores
# # only for the new sequence. Thus, the matrix of scores is of size
# # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
# # j > cache_len + i, since row i corresponds to token cache_len + i.
scores = torch.randn((seq_len, seq_len))
offs_m = torch.tensor([0, 1, 2, 3])
offs_k = torch.tensor([0, 1, 2, 3])
mask3 = offs_m[:, None] >= offs_k[None, :]
print(mask3)
mask4 = scores.masked_fill(mask3 == 0, float('-inf'))
print(mask4)

"""
tensor([[-inf, -inf, -inf, -inf],
        [-inf, -inf, -inf, -inf],
        [-inf, -inf, -inf, -inf],
        [-inf, -inf, -inf, -inf]])
tensor([[0., -inf, -inf, -inf],
        [0., 0., -inf, -inf],
        [0., 0., 0., -inf],
        [0., 0., 0., 0.]])
tensor([[0., -inf, -inf, -inf],
        [0., 0., -inf, -inf],
        [0., 0., 0., -inf],
        [0., 0., 0., 0.]])
mask shape is  torch.Size([4, 4])
tensor([[ True, False, False, False],
        [ True,  True, False, False],
        [ True,  True,  True, False],
        [ True,  True,  True,  True]])
tensor([[ 2.2425,    -inf,    -inf,    -inf],
        [-0.4196,  1.4955,    -inf,    -inf],
        [ 1.1759,  1.9087,  0.2180,    -inf],
        [-0.5477,  0.1412,  0.7192,  0.8276]])
"""