from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from kernels import *
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 128256 # Later set in the build method

    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = 1.5
    norm_eps: float = 1e-5
    rope_theta: float =  500000.0
    use_scaled_rope: bool = True
    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = "cuda"

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class FusedAttention(nn.Module):
    def __init__(self,  args: ModelArgs):
        super().__init__()
        self.args = args
        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads
        self.hidden_size = args.n_heads * self.head_dim

        self.q_proj_weight = nn.Parameter(torch.rand( args.dim, args.n_heads * self.head_dim))
        self.k_proj_weight = nn.Parameter(torch.rand( args.dim, self.n_kv_heads * self.head_dim))
        self.v_proj_weight = nn.Parameter(torch.rand( args.dim, self.n_kv_heads * self.head_dim))

        self.o_proj_weight = nn.Parameter(torch.rand(args.dim, args.n_heads * self.head_dim))

        # Initialize caches to store Key, Values at start. (KV Cache Implementation)
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)

    def forward(self, 
        x: torch.Tensor,
        start_pos: int,
    ):
        batch_size, seq_len, _ = x.shape  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)

        # decode stage: (B, 1, Dim) -> (B, 1, H_Q * Head_Dim). H_Q: heads of query, H_K: heads of key
        print(f"self.k_proj_weight.data {self.k_proj_weight.data.shape}")

        # x[bsz,seq_len,dim] * wq[dim,n_kv_heads * head_dim] -> k[bsz,seq_len,n_kv_heads * head_dim]
        xq = fused_linear(x, self.q_proj_weight.data)
        xk = fused_linear(x, self.k_proj_weight.data)
        xv = fused_linear(x, self.v_proj_weight.data)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim), 
        print(xq.shape, xk.shape, xv.shape)
        print(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = xq.contiguous().view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.contiguous().view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.contiguous().view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Compute rotation matrix for each position in the sequence
        freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len * 2)
        freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
        # Apply RoPE to Queries and Keys embeddings
        xq = rope(xq, freqs_cis)
        xk = rope(xk, freqs_cis)

        # 2. kv cache 
        # Replace the entry in the cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # 3. GQA
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # 4. sel-attention (flash_attention)
        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)
        # flashattention: softmax(qk^t) * v
        output = flash_attention_v1(xq, keys, values)

        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        output = fused_linear(x, self.o_proj_weight.data) # (B, 1, Dim) -> (B, 1, Dim)
        
        return output

    def get_fwd_flops(self, num_tokens):
        h = self.hidden_size
        layer_norm = num_tokens * h + num_tokens * h
        c_attn = num_tokens * (3 * h) * (2 * h) + num_tokens * (3 * h)
        c_proj = num_tokens * h * (2 * h) + num_tokens * h
        return layer_norm + c_attn + c_proj

    
class FusedMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.gate_proj_weight = nn.Parameter(torch.rand(args.dim, hidden_dim)) # w1
        self.down_proj_weight = nn.Parameter(torch.rand(hidden_dim, args.dim)) # w2
        self.up_proj_weight = nn.Parameter(torch.rand(args.dim, hidden_dim))   # w3

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_gate_silu = fused_linear(x, self.gate_proj_weight.data, add_silu = True)
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_up = fused_linear(x, self.up_proj_weight.data)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = x_gate_silu * x_up
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = fused_linear(x, self.down_proj_weight.data)
        
        return x
    

class LlamaDecoderLayer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.post_attention_weight = nn.Parameter(torch.ones((self.dim,)))
        self.input_layernorm_weight = nn.Parameter(torch.ones((self.dim,)))

        self.attention = FusedAttention(args)
        self.feed_forward = FusedMLP(args)
    
    def forward(self, x: torch.Tensor, start_pos: int):
        # Normalization BEFORE the attention block
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        post_attention = rmsnorm(x, self.post_attention_weight.data, eps=self.args.norm_eps)
        h = x + self.attention.forward(
           post_attention, start_pos
        )
        # Normalization BEFORE the feed forward block
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        input_rmsnorm = rmsnorm(h, self.input_layernorm_weight.data, eps=self.args.norm_eps)
        out = h + self.feed_forward.forward(input_rmsnorm)
        return out


class Llama(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        self.norm_weight = nn.Parameter(torch.ones((args.dim,)))
        self.lm_head_weight = nn.Parameter(torch.rand(args.dim, self.vocab_size))
        self.freqs_complex = precompute_freqs_cis(self.args.dim // self.args.n_heads, 
                                                  self.args.max_seq_len * 2, 
                                                  device=self.args.device)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(args) for layer_id in range(args.n_layers)]
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)
        
        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos)

        h = rmsnorm(h, self.norm_weight.data, eps=self.args.norm_eps)
        output = fused_linear(h, self.lm_head_weight.data).float()

        return output

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_args: ModelArgs = ModelArgs()
    
    # x = torch.randint(1000, [2, 64]).to(device)
    # model = Llama(model_args).to(device)
    # output = model(x, 0)
    
    # print(output.shape)
    x_norm = torch.randn((2, 64, 2048), device=device)
    # rms_norm = RMSNorm(dim=ModelArgs.dim)
    # x_norm = rms_norm(x)
    
    attention = FusedAttention(ModelArgs).to(device)
    x_out = attention(x_norm, start_pos=0).to(device)
    print(f"x_out.shape: {x_out.shape}")
