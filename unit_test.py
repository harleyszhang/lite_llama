import torch
import torch.nn as nn
from transformers.activations import ACT2FN
import pytest

from kernels.fused_linear import fused_ffn
from kernels.rmsnorm import rmsnorm
from kernels.layernorm import layernorm
from kernels.softmax import naive_softmax, softmax


class RMSNorm(nn.Module):
    """nlp 领域"""
    def __init__(self, dim, eps=1e-8):
        """
        :param dim: 输入的维度
        :param eps: 防止除以0的稳定项
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数
    
    def forward(self, x):
        # x 的形状为 [batch_size, seq_len, dim]        
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt( var + self.eps)
        return x / rms * self.weight # 归一化，并应用缩放参数

def _get_inputs(M, K, N, device="cuda"):
    """return 2D Tensor of input weight bias and redisual input"""

    torch.manual_seed(1337)
    x = torch.rand((M, K), device=device, dtype=torch.float32)
    w = torch.rand((K, N), device=device, dtype=torch.float32)
    b = torch.rand((N,), device=device, dtype=torch.float32)
    r = torch.rand_like(x, dtype=torch.float32)
    if K != N:
        r = r_torch = None
    
    return x, w, b, r
 
def torch_ffn(x, w, b=None, r=None):
    z = x @ w
    if b is not None:
        z += b
    z = ACT2FN["gelu_new"](z)
    if r is not None:
        z += r
    return z

@pytest.mark.parametrize("M,N,K", [(128, 128, 64)])
def test_fused_ffn(M, N, K):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_torch, w_torch, _, _ = _get_inputs(M, K, N, device)
    x, w, _, _ = _get_inputs(M, K, N, device)

    z_torch = torch_ffn(x_torch, w_torch, b=None, r=None)
    z = fused_ffn(x, w)
    assert torch.allclose(z, z_torch, atol=1e-2), (z - z_torch).abs().max()
    
    
@pytest.mark.parametrize("M", [128, 32])
@pytest.mark.parametrize("K", [32, 128, 64])
def test_rmsnorm(M, K):
    N = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device is ", device)
    
    x, *_ = _get_inputs(M, K, N, device)
    x_torch, *_ = _get_inputs(M, K, N, device)

    # 模块及其所有参数（如 self.weight）都位于指定设备上（CPU 或 GPU）
    rmsnorm_pytorch = RMSNorm(K).to(device)
    x_torch = rmsnorm_pytorch(x_torch)

    x = rmsnorm(x, rmsnorm_pytorch.weight.data).to(device)
    assert torch.allclose(x, x_torch, atol=1e-4)
    
@pytest.mark.parametrize("M", [128, 32, 64])
@pytest.mark.parametrize("K", [32, 128, 64])
def test_layernorm(M, K):
    N = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device is ", device)
    
    x, *_ = _get_inputs(M, K, N, device)
    x_torch, *_ = _get_inputs(M, K, N, device)

    # 模块及其所有参数（如 self.weight）都位于指定设备上（CPU 或 GPU）
    layernorm_pytorch = nn.LayerNorm(K).to(device)
    x_torch = layernorm_pytorch(x_torch)

    x = layernorm(x, layernorm_pytorch.weight.data, layernorm_pytorch.bias.data).to(device)
    assert torch.allclose(x, x_torch, atol=1e-5)


@pytest.mark.parametrize("M", [128, 32, 64])
@pytest.mark.parametrize("K", [32, 128, 64])
def test_softmax(M, K):
    N = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    x, *_ = _get_inputs(M, K, N, device)
    x_torch, *_ = _get_inputs(M, K, N, device)
    
    # 模块及其所有参数（如 self.weight）都位于指定设备上（CPU 或 GPU）
    output_torch = torch.softmax(x, axis=-1).to(device)
    output = softmax(x).to(device)
    assert torch.allclose(output, output_torch, atol=1e-5)
