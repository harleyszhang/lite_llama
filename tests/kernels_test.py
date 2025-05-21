import torch, math
import torch.nn as nn
from transformers.activations import ACT2FN
import pytest, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from lite_llama.lite_llama.kernels.others.fused_linear import fused_linear
from lite_llama.lite_llama.kernels.others.rmsnorm_v1 import rmsnorm
from lite_llama.lite_llama.kernels.others.layernorm import layernorm
from lite_llama.kernels.softmax import naive_softmax, softmax_fwd
from lite_llama.kernels.flashattention import flash_attention_v1
from lite_llama.lite_llama.kernels.others.rope_orig import rope
from typing import Callable, Dict, Tuple, Union
from lite_llama.tests.test_torch_rope import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)


class RMSNorm(nn.Module):
    """nlp 领域"""

    def __init__(self, dim):
        """
        :param dim: 输入的维度
        :param eps: 防止除以0的稳定项
        """
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def forward(self, x):
        # x 的形状为 [batch_size, seq_len, dim]
        var = torch.mean(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(var)
        return x / rms * self.weight  # 归一化，并应用缩放参数


def _get_attn_inputs(B, N, L, H, device):
    torch.manual_seed(1337)
    q = torch.rand((B, N, L, H), device=device, dtype=torch.float16)
    k = torch.rand_like(q)
    v = torch.rand_like(q)
    return q, k, v


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
    z = fused_linear(x, w)
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

    x = layernorm(x, layernorm_pytorch.weight.data, layernorm_pytorch.bias.data).to(
        device
    )
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
    output = softmax_fwd(x).to(device)
    assert torch.allclose(output, output_torch, atol=1e-5)


def torch_attention(q, k, v, attention_mask=None, is_causal=False):
    assert q.shape == k.shape == v.shape
    B, N, L, H = q.shape
    sm_scale = 1 / math.sqrt(H)
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.nn.functional.softmax(p, dim=-1)

    ref_out = torch.matmul(p.to(v.dtype), v)
    if attention_mask is not None:
        p += attention_mask
    if is_causal:
        m_size = q.size(2)
        n_size = k.size(2)
        M = torch.tril(torch.ones((m_size, n_size), device="cuda"))
        p = torch.where(M == 0, float("-inf"), p)

    return ref_out


@pytest.mark.parametrize("B,N", [(4, 8), (8, 16), (24, 32), (64, 20)])
@pytest.mark.parametrize(
    "L",
    [
        128,
        256,
    ],
)
@pytest.mark.parametrize("H", [32, 64])
def test_flash_attention_v1(B, N, L, H):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q, k, v = _get_attn_inputs(B, N, L, H, device)
    batch, heads, m_size, dhead = q.shape
    atten_out = torch.empty_like(q)
    sm_scale = 1 / math.sqrt(dhead)
    z_torch = torch_attention(q, k, v)
    z = flash_attention_v1(q, k, v, sm_scale)
    print(f"z_torch: {z_torch[0][0][0][0]}, z: {z[0][0][0][0]}")
    assert torch.allclose(z[0], z_torch[0], atol=1e-3), (z - z_torch).abs().max()


def get_tol(dtype: torch.dtype) -> Dict:
    if dtype == torch.bfloat16:
        return dict(atol=1e-2, rtol=1e-2)
    elif dtype == torch.float16:
        return dict(atol=1e-3, rtol=1e-3)
    return dict(atol=1e-5, rtol=1.3e-6)


# Gradient is a broadcasted scalar
def _overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    return output.sum() * 2


# Gradient is a full tensor
def _non_overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    t = torch.ones_like(output)
    return torch.sum(output * t)


####################################### triton 版rope 算法单元测试 #################################
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("seq_length", [1024, 2048])
@pytest.mark.parametrize("hidden_size", [64, 128])
@pytest.mark.parametrize("rotary_percent", [1.0])
@pytest.mark.parametrize("margin", [0, 10])
@pytest.mark.parametrize("transpose", [None])
@pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_triton_rope(
    dtype: torch.dtype,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    margin: int,
    transpose: Union[Tuple, None],
    tensor_format: str,
    loss_func: Callable,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    t = torch.rand(
        (seq_length - margin, batch_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    if tensor_format == "bshd":
        t = t.transpose(0, 1).contiguous()

    if transpose:
        t = t.transpose(*transpose).contiguous().transpose(*transpose)

    t.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(seq_length)

    # triton
    output_triton = rope(t, emb, tensor_format=tensor_format)

    loss_triton = loss_func(output_triton)
    loss_triton.backward()
    grad_triton = t.grad.detach().clone()
    t.grad = None

    # te
    output_te = apply_rotary_pos_emb(
        t,
        emb,
        tensor_format=tensor_format,
        fused=True,
    )

    loss_te = loss_func(output_te)
    loss_te.backward()
    grad_te = t.grad.detach().clone()
    t.grad = None

    torch.testing.assert_close(output_te, output_triton, **get_tol(dtype))
    torch.testing.assert_close(grad_te, grad_triton, **get_tol(dtype))
    assert output_te.is_contiguous()
