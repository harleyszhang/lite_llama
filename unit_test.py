import torch
from transformers.activations import ACT2FN
import pytest

from kernels.fused_linear import fused_ffn

def _get_inputs(M, K, N, device):
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
    
