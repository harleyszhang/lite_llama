# test_model.py

import torch
import pytest, os, sys
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from lite_llama.kernels import fused_linear, rmsnorm
from lite_llama.models.llama import (
    ModelArgs,
    FusedAttention,
    FusedMLP,
    LlamaDecoderLayer,
    Llama,
)


@dataclass
class TestArgs:
    dim: int = 512
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    vocab_size: int = 1000
    multiple_of: int = 256
    ffn_dim_multiplier: float = 1.5
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_scaled_rope: bool = True
    max_batch_size: int = 4
    max_seq_len: int = 64
    device: str = "cuda"


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        pytest.skip(
            "CUDA is not available. Skipping tests that require CUDA.",
            allow_module_level=True,
        )


@pytest.fixture
def model_args(device):
    return TestArgs(device=device)


def test_rmsnorm(device):
    batch_size, seq_len, dim = 2, 64, 128
    x = torch.randn(batch_size, seq_len, dim, device=device)
    norm = torch.ones(batch_size * seq_len, device=device)
    eps = 1e-5
    output = rmsnorm(x.view(-1, dim), norm, eps)
    expected = x.view(-1, dim) / (
        torch.sqrt(torch.mean(x.view(-1, dim) ** 2, dim=1, keepdim=True)) + eps
    )
    assert torch.allclose(output, expected, atol=1e-4), (
        "RMSNorm does not match expected output."
    )


def test_fused_attention(model_args):
    batch_size, seq_len, dim = 2, 16, model_args.dim
    x = torch.randn(batch_size, seq_len, dim, device=model_args.device)
    start_pos = 0
    attention = FusedAttention(model_args).to(model_args.device)
    output = attention(x, start_pos)
    # Since FusedAttention uses Triton kernels, compare shapes
    assert output.shape == (batch_size, seq_len, dim), (
        "FusedAttention output shape mismatch."
    )


def test_fused_mlp(model_args):
    batch_size, seq_len, dim = 2, 16, model_args.dim
    x = torch.randn(batch_size, seq_len, dim, device=model_args.device)

    mlp = FusedMLP(model_args).to(model_args.device)
    output = mlp(x)
    # The output dimension should match the input dimension
    assert output.shape == (batch_size, seq_len, dim), "FusedMLP output shape mismatch."


def test_llama_decoder_layer(model_args):
    batch_size, seq_len, dim = 2, 16, model_args.dim
    x = torch.randn(batch_size, seq_len, dim, device=model_args.device)
    start_pos = 0
    layer = LlamaDecoderLayer(model_args).to(model_args.device)
    output = layer(x, start_pos)
    assert output.shape == (batch_size, seq_len, dim), (
        "LlamaDecoderLayer output shape mismatch."
    )


def test_llama_model(model_args):
    batch_size, seq_len = 2, 16
    tokens = torch.randint(
        0, model_args.vocab_size, (batch_size, seq_len), device=model_args.device
    )

    model = Llama(model_args).to(model_args.device)
    output = model(tokens, start_pos=0)
    assert output.shape == (batch_size, seq_len, model_args.vocab_size), (
        "Llama model output shape mismatch."
    )


if __name__ == "__main__":
    pytest.main([__file__])
