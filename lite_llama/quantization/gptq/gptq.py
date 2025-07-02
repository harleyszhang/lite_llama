import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any
from tqdm.auto import tqdm
import triton
import triton.language as tl
import time, gc, psutil, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lite_llama.utils.common import get_gpu_memory  # Replace with actual GPU mem check if needed



class GPTQ:
    def __init__(
            self,
            layer: nn.Module = None,
            wbits: int = 4,
            groupsize: int = 8,
            actorder: bool = False,
            percdamp: float = 0.01,
            blocksize: int = 128,
            device: str = "cuda"
    ):
        self.layer = layer
        self.wbits = wbits
        self.groupsize = groupsize if groupsize != -1 else float('inf')
        self.actorder = actorder
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.device = device
        self.maxq = 2 ** wbits - 1

    def relative_error_loss(self, w_original: torch.Tensor, w_reconstructed: torch.Tensor,
                            eps: float = 1e-5) -> torch.Tensor:
        """Compute relative error loss with better handling of small weights"""
        abs_diff = (w_original - w_reconstructed).abs()

        # Use adaptive epsilon based on weight magnitude distribution
        w_abs = w_original.abs()
        adaptive_eps = torch.maximum(
            torch.tensor(eps, device=w_original.device),
            0.01 * w_abs.median()  # Use median as robust estimate
        )

        rel_err = abs_diff / (w_abs + adaptive_eps)

        # Use robust loss to handle outliers
        return rel_err.mean() + 0.1 * rel_err.pow(2).mean()

    def optimize_for_relative_error(self, w_group: torch.Tensor, max_iter: int = 200) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Optimize scale and zero specifically for minimal relative error"""
        device = w_group.device

        # Separate handling for near-zero and normal weights
        w_abs = w_group.abs()
        w_median = w_abs.median()
        small_weight_threshold = 0.1 * w_median

        # Initialize with better starting points
        w_min = w_group.min(dim=-1, keepdim=True)[0]
        w_max = w_group.max(dim=-1, keepdim=True)[0]

        # For groups with many small weights, use tighter bounds
        if (w_abs < small_weight_threshold).float().mean() > 0.3:
            # Use percentile-based bounds for groups with many small weights
            w_flat = w_group.view(w_group.shape[0], -1)
            w_sorted = torch.sort(w_flat, dim=-1)[0]
            n = w_sorted.shape[-1]
            w_min = w_sorted[:, max(0, int(0.05 * n)):max(1, int(0.05 * n) + 1)]
            w_max = w_sorted[:, min(n - 1, int(0.95 * n)):min(n, int(0.95 * n) + 1)]

        range_val = w_max - w_min
        range_val = torch.where(range_val < 1e-8, torch.tensor(1e-6, device=device), range_val)

        # Initialize parameters
        scale = nn.Parameter((range_val / self.maxq).clamp(min=1e-8))
        zero = nn.Parameter(torch.round(-w_min / scale).clamp(0, self.maxq))

        optimizer = torch.optim.AdamW([scale, zero], lr=0.005, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)

        best_loss = float('inf')
        best_scale = scale.data.clone()
        best_zero = zero.data.clone()
        patience = 20
        no_improve = 0

        for i in range(max_iter):
            optimizer.zero_grad()

            # Ensure valid range
            scale.data.clamp_(min=1e-8, max=1e3)
            zero.data.clamp_(0, self.maxq)

            # Quantize and dequantize
            q = torch.clamp(torch.round(w_group / scale + zero), 0, self.maxq)
            w_rec = (q - zero) * scale

            # Use relative error loss
            loss = self.relative_error_loss(w_group, w_rec)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_scale = scale.data.clone()
                best_zero = zero.data.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([scale, zero], 1.0)

            optimizer.step()
            scheduler.step()

        return best_scale.detach(), best_zero.detach()

    def magnitude_aware_quantization(self, w_group: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use different strategies based on weight magnitudes"""
        device = w_group.device
        w_abs = w_group.abs()
        w_std = w_group.std(dim=-1, keepdim=True)
        w_mean = w_group.mean(dim=-1, keepdim=True)

        # Strategy 1: For groups with large dynamic range, use log-scale quantization
        dynamic_range = w_abs.max(dim=-1, keepdim=True)[0] / (w_abs.min(dim=-1, keepdim=True)[0] + 1e-8)

        if dynamic_range.mean() > 100:  # High dynamic range
            # Use log-space quantization for better relative precision
            sign = torch.sign(w_group)
            w_abs_log = torch.log(w_abs + 1e-8)

            log_min = w_abs_log.min(dim=-1, keepdim=True)[0]
            log_max = w_abs_log.max(dim=-1, keepdim=True)[0]

            scale_log = (log_max - log_min) / (self.maxq - 1)
            zero_log = torch.round(-log_min / scale_log).clamp(0, self.maxq - 1)

            # Convert back to linear scale
            q_log = torch.clamp(torch.round((w_abs_log - log_min) / scale_log), 0, self.maxq - 1)
            w_abs_rec = torch.exp(log_min + q_log * scale_log)
            w_rec = sign * w_abs_rec

            # Compute equivalent linear scale and zero
            scale = (w_group.max(dim=-1, keepdim=True)[0] - w_group.min(dim=-1, keepdim=True)[0]) / self.maxq
            zero = torch.round(-w_group.min(dim=-1, keepdim=True)[0] / scale).clamp(0, self.maxq)

        else:
            # Strategy 2: For normal range, use adaptive clipping
            # Use robust statistics to set bounds
            median = w_group.median(dim=-1, keepdim=True)[0]
            mad = (w_group - median).abs().median(dim=-1, keepdim=True)[0]  # Median Absolute Deviation

            # Set bounds using robust statistics
            bound = 3.0 * mad
            w_min = torch.maximum(w_group.min(dim=-1, keepdim=True)[0], median - bound)
            w_max = torch.minimum(w_group.max(dim=-1, keepdim=True)[0], median + bound)

            range_val = w_max - w_min
            range_val = torch.where(range_val < 1e-8, torch.tensor(1e-6, device=device), range_val)

            scale = range_val / self.maxq
            zero = torch.round(-w_min / scale).clamp(0, self.maxq)

        return scale, zero

    def quantize(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantization optimized specifically for minimal relative error
        Returns: [O, I] int4, [O, num_groups] zero, [O, num_groups] scale
        """
        assert W.ndim == 2
        rows, cols = W.shape
        device = W.device

        # Use very small groups for maximum precision
        effective_groupsize = min(int(self.groupsize), 8) if self.groupsize != float('inf') else 8
        effective_groupsize = max(effective_groupsize, 4)  # Minimum 4 for 4-bit
        num_groups = (cols + effective_groupsize - 1) // effective_groupsize

        qweight = torch.zeros((rows, cols), dtype=torch.uint8, device=device)
        scales = torch.zeros((rows, num_groups), dtype=torch.float16, device=device)
        zeros = torch.zeros((rows, num_groups), dtype=torch.float16, device=device)

        # Process each group with relative error optimization
        for g in range(num_groups):
            start_col = g * effective_groupsize
            end_col = min((g + 1) * effective_groupsize, cols)

            # Get current group
            W_group = W[:, start_col:end_col].clone()

            # Try different methods and pick best for relative error
            methods = []

            # Method 1: Relative error optimization
            try:
                scale_rel, zero_rel = self.optimize_for_relative_error(W_group, max_iter=100)
                q_rel = torch.clamp(torch.round(W_group / scale_rel + zero_rel), 0, self.maxq)
                w_rec_rel = (q_rel - zero_rel) * scale_rel
                rel_error_rel = self.relative_error_loss(W_group, w_rec_rel).item()
                methods.append(('rel_opt', scale_rel, zero_rel, q_rel, rel_error_rel))
            except Exception as e:
                print(f"Relative opt failed for group {g}: {e}")

            # Method 2: Magnitude-aware quantization
            try:
                scale_mag, zero_mag = self.magnitude_aware_quantization(W_group)
                q_mag = torch.clamp(torch.round(W_group / scale_mag + zero_mag), 0, self.maxq)
                w_rec_mag = (q_mag - zero_mag) * scale_mag
                rel_error_mag = self.relative_error_loss(W_group, w_rec_mag).item()
                methods.append(('mag_aware', scale_mag, zero_mag, q_mag, rel_error_mag))
            except Exception as e:
                print(f"Magnitude aware failed for group {g}: {e}")

            # Method 3: Ultra-conservative approach for small weights
            w_abs = W_group.abs()
            if w_abs.max() < 0.01:  # Very small weights
                # Use much finer quantization resolution
                w_min = W_group.min(dim=-1, keepdim=True)[0]
                w_max = W_group.max(dim=-1, keepdim=True)[0]

                # Tighten the range for small weights
                range_val = w_max - w_min
                range_val = torch.where(range_val < 1e-8, torch.tensor(1e-8, device=device), range_val)

                scale_small = range_val / self.maxq * 0.8  # Use 80% of range for safety
                zero_small = torch.round(-w_min / scale_small).clamp(0, self.maxq)

                q_small = torch.clamp(torch.round(W_group / scale_small + zero_small), 0, self.maxq)
                w_rec_small = (q_small - zero_small) * scale_small
                rel_error_small = self.relative_error_loss(W_group, w_rec_small).item()
                methods.append(('small_weights', scale_small, zero_small, q_small, rel_error_small))

            # Pick the method with lowest relative error
            if methods:
                best_method = min(methods, key=lambda x: x[4])
                method_name, scale_best, zero_best, q_best, _ = best_method

                qweight[:, start_col:end_col] = q_best.to(torch.uint8)
                scales[:, g] = scale_best.squeeze(-1)
                zeros[:, g] = zero_best.squeeze(-1)
            else:
                # Ultimate fallback
                print(f"All methods failed for group {g}, using zero quantization")
                qweight[:, start_col:end_col] = 0
                scales[:, g] = 1.0
                zeros[:, g] = 0

        return qweight, zeros.to(torch.float16), scales.to(torch.float16)


    def dequantize(self, qweight: torch.Tensor, zeros: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        [O, I] int4, [O, num_groups] zero, [O, num_groups] scale => [O, I] float16
        """
        rows, cols = qweight.shape
        # Use same effective groupsize as quantization
        effective_groupsize = min(self.groupsize, 8)
        effective_groupsize = max(effective_groupsize, 4)
        num_groups = (cols + effective_groupsize - 1) // effective_groupsize
        W = torch.zeros_like(qweight, dtype=torch.float16)

        for g in range(num_groups):
            start = g * effective_groupsize
            end = min((g + 1) * effective_groupsize, cols)
            scale = scales[:, g].unsqueeze(1)  # [O, 1]
            zero = zeros[:, g].unsqueeze(1)  # [O, 1]
            q = qweight[:, start:end].float()
            W[:, start:end] = (q - zero) * scale

        return W

def quantize_gptq(
        model_state_dict: Dict[str, torch.Tensor],
        calibration_data: Optional[torch.Tensor] = None,
        wbits: int = 4,
        groupsize: int = 8,
        target_layers: Optional[list] = None,
        device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Quantize model weights using GPTQ algorithm

    Args:
        model_state_dict: Original model state dictionary
        calibration_data: Optional calibration data for computing Hessian
        wbits: Number of bits for quantization (default: 4)
        groupsize: Group size for quantization (default: 128)
        target_layers: List of layer names to quantize (if None, quantize all linear layers)
        device: Device to perform quantization on

    Returns:
        Dictionary containing quantized weights and quantization parameters
    """
    quantized_state_dict = {}

    # Default target layers if not specified
    if target_layers is None:
        target_layers = []
        for name in model_state_dict.keys():
            if any(pattern in name for pattern in [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "kv_proj"
            ]):
                target_layers.append(name)

    print(f"Quantizing {len(target_layers)} layers to {wbits} bits...")

    for name, param in tqdm(model_state_dict.items(), desc="Processing layers"):
        if name in target_layers and param.dim() == 2:
            # Create GPTQ quantizer for this layer
            gptq = GPTQ(
                layer=None,  # We're working directly with tensors
                wbits=wbits,
                groupsize=groupsize,
                device=device
            )

            # Move weight to device
            weight = param.to(device).float()

            # Quantize the weight
            qweight, qzeros, scales = gptq.quantize(weight)

            # Store quantized parameters
            base_name = name.replace(".weight", "").replace("_weight", "")
            quantized_state_dict[f"{base_name}.qweight"] = qweight.cpu()
            quantized_state_dict[f"{base_name}.qzeros"] = qzeros.cpu()
            quantized_state_dict[f"{base_name}.scales"] = scales.cpu()
            quantized_state_dict[f"{base_name}.wbits"] = torch.tensor(wbits)
            quantized_state_dict[f"{base_name}.groupsize"] = torch.tensor(groupsize)

        else:
            # Keep non-quantized parameters as is
            quantized_state_dict[name] = param.cpu()

    return quantized_state_dict

@triton.autotune(
        configs=[
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=5,
                num_warps=2,
            ),
        ],
        key=["M", "N", "K"],
    )

@triton.jit
def int4_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    bscales_ptr, bzeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_mask = offs_am[:, None] < M
    b_mask = offs_bn[None, :] < N

    a_ptrs = a_ptr + stride_am * offs_am[:, None] + stride_ak * offs_k[None, :]
    b_ptrs = b_ptr + stride_bn * offs_bn[None, :] + stride_bk * (offs_k[:, None] // 2)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    for k in range(0, K, BLOCK_SIZE_K):
        b_q = tl.load(b_ptrs, mask=b_mask)

        a = tl.load(a_ptrs, mask=a_mask).to(tl.float16)

        # Compute per-group index
        k_offset = k + offs_k  # shape: [BLOCK_SIZE_K]
        group_idx = k_offset // GROUP_SIZE  # [BLOCK_SIZE_K]

        # Load scale and zero for each [N, G]
        scale = tl.load(bscales_ptr + stride_bn * offs_bn[None, :] + group_idx[:, None]).to(tl.float16)  # [BLOCK_SIZE_K, BLOCK_SIZE_N]
        zero = tl.load(bzeros_ptr + stride_bn * offs_bn[None, :] + group_idx[:, None]).to(tl.float16)   # same shape

        # Extract int4 values from uint8
        shift = (k_offset[:, None] % 2) * 4
        q = (b_q.to(tl.uint8) >> shift) & 0xF
        b_deq = (q.to(tl.float16) - zero) * scale

        accumulator += tl.dot(a, b_deq, out_dtype=tl.float16)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_int4_gemm(
    inp: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 64
) -> torch.Tensor:


    weight = weight.t().contiguous()  # [K/2, N]
    c_shape = inp.shape[:-1] + weight.shape[-1:]
    inp = inp.view(-1, inp.shape[-1]).contiguous()

    PAD_TO = 256
    if inp.shape[0] % PAD_TO != 0:
        c_crop = inp.shape[0]
        new_inp = inp.new_zeros(((inp.shape[0] + PAD_TO - 1) // PAD_TO * PAD_TO, inp.shape[1]))
        new_inp[:c_crop] = inp
        inp = new_inp
    else:
        c_crop = None

    M, K = inp.shape
    N = weight.shape[1]



    c = torch.empty((M, N), device=inp.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    int4_gemm_kernel[grid](
        inp, weight, c,
        scales, zeros,
        M, N, K,
        inp.stride(0), inp.stride(1),
        weight.stride(0), weight.stride(1),
        c.stride(0), c.stride(1),
        GROUP_SIZE=group_size,
    )

    return c[:c_crop] if c_crop is not None else c.view(c_shape)

class GPTQLinear(nn.Module):
    """
    4-bit quantized linear layer using Triton kernels
    """

    def __init__(self, in_features, out_features, bias=True, groupsize=64, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        self.device = device

        self.tile_cols = groupsize
        self.original_out_features = out_features

        # Quantized params (assigned later)
        self.register_buffer("packed_weight", None)
        self.register_buffer("scales", None)
        self.register_buffer("zeros", None)
        self.register_buffer("bias", None if not bias else torch.empty(out_features))

    @staticmethod
    def pack_weight(weight):
        rows, cols = weight.shape
        if cols % 2 != 0:
            weight = torch.nn.functional.pad(weight, (0, 1), value=0)
            cols += 1
        packed = (weight[:, 0::2] & 0xF) | ((weight[:, 1::2] & 0xF) << 4)
        return packed.contiguous()

    def get_weight(self, packed: torch.Tensor) -> torch.Tensor:
        """
        [rows, ceil(cols/2)] uint8 -> [rows, cols] uint8 in [0, 15]
        """
        rows, packed_cols = packed.shape
        qweight = torch.empty((rows, packed_cols * 2), dtype=torch.uint8, device=packed.device)
        qweight[:, 0::2] = packed & 0xF
        qweight[:, 1::2] = (packed >> 4) & 0xF
        return qweight[:, :self.in_features].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(-1, self.in_features)
        # Compute quantized matmul
        output = triton_int4_gemm(
            x_flat.float(),
            self.packed_weight,
            self.scales,
            self.zeros,
            group_size=self.groupsize,
        )

        if self.bias is not None:
            output += self.bias

        return output.view(*x.shape[:-1], self.out_features)


def get_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() // (1024 ** 2)
    return 0


def test_gptqlinear_vs_nnlinear(
    in_features=2048,
    out_features=4096,
    groupsize=64,
    wbits=4,
    device="cuda"
):
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- Create single input vector ----
    x = torch.randn(in_features, device=device, dtype=torch.float16)
    linear = nn.Linear(in_features, out_features, bias=True, device=device, dtype=torch.float16).eval()

    weight = linear.weight.detach().to(device).float()
    bias = linear.bias.detach().to(device).float() if linear.bias is not None else None

    # --- Quantize using GPTQ ---
    gptq = GPTQ(wbits=wbits, groupsize=groupsize, device=device)
    qweight, qzeros, qscales = gptq.quantize(weight)
    packed_weight = GPTQLinear.pack_weight(qweight)

    gptqlinear = GPTQLinear(in_features, out_features, bias=True, groupsize=groupsize, device=device).to(device)
    gptqlinear.packed_weight = packed_weight
    gptqlinear.scales = qscales
    gptqlinear.zeros = qzeros
    gptqlinear.bias = bias if bias is not None else None
    gptqlinear.eval()

    # ---- Memory ----
    gc.collect()
    torch.cuda.empty_cache()
    mem0 = get_gpu_memory()
    _ = linear(x)
    mem_fp = get_gpu_memory()
    del _
    gc.collect()
    torch.cuda.empty_cache()
    mem1 = get_gpu_memory()
    _ = gptqlinear(x)
    mem_q = get_gpu_memory()
    del _
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Print ----


    print("\n== Memory Usage (VRAM, MB) ==")
    print(f"nn.Linear (fp16): {mem_fp} MB MB)")
    print(f"GPTQLinear:       {mem_q} MB MB)")

    print("\n== Latency ==")
    time_fp = triton.testing.do_bench(lambda: linear(x))
    time_q = triton.testing.do_bench(lambda: gptqlinear(x))
    print(f"nn.Linear (fp16): {time_fp:.3f} ms")
    print(f"GPTQLinear:       {time_q:.3f} ms")


    print("\n== VRAM saving ratio ==")
    print(f"GPTQLinear / nn.Linear: {(mem_q-mem1)/(mem_fp-mem0 + 1e-9):.3f}x")
    print(f"Speedup: {time_fp/time_q:.2f}x\n")

if __name__ == "__main__":
    test_gptqlinear_vs_nnlinear()