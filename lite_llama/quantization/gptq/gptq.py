import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm.auto import tqdm
import math


def pack_int4_weights(qweight: torch.Tensor, wbits: int = 4) -> torch.Tensor:
    """
    Pack quantized weights into int32 for efficient storage.
    For 4-bit quantization, pack 8 weights into one int32.

    Args:
        qweight: Quantized weight tensor of shape (rows, cols) with values in [0, 15]
        wbits: Number of bits per weight (4 for int4)

    Returns:
        Packed weight tensor of shape (rows, cols // 8)
    """
    assert wbits == 4, "This function currently only supports 4-bit packing"

    rows, cols = qweight.shape
    pack_factor = 32 // wbits  # 8 for 4-bit

    # Ensure we can pack evenly
    if cols % pack_factor != 0:
        # Pad columns to make it divisible by pack_factor
        pad_cols = pack_factor - (cols % pack_factor)
        qweight = torch.nn.functional.pad(qweight, (0, pad_cols), value=0)
        cols = qweight.shape[1]

    packed_cols = cols // pack_factor
    packed = torch.zeros((rows, packed_cols), dtype=torch.int32, device=qweight.device)

    # Pack weights
    for i in range(pack_factor):
        packed |= (qweight[:, i::pack_factor].to(torch.int32) & 0xF) << (i * 4)

    return packed


def unpack_int4_weights(packed: torch.Tensor, original_cols: int, wbits: int = 4) -> torch.Tensor:
    """
    Unpack int4 weights from int32 storage.

    Args:
        packed: Packed weight tensor
        original_cols: Original number of columns before packing
        wbits: Number of bits per weight

    Returns:
        Unpacked weight tensor
    """
    assert wbits == 4, "This function currently only supports 4-bit unpacking"

    rows, packed_cols = packed.shape
    pack_factor = 32 // wbits  # 8 for 4-bit

    # Calculate unpacked dimensions
    unpacked_cols = packed_cols * pack_factor
    unpacked = torch.zeros((rows, unpacked_cols), dtype=torch.int32, device=packed.device)

    # Unpack weights
    for i in range(pack_factor):
        unpacked[:, i::pack_factor] = (packed >> (i * 4)) & 0xF

    # Remove padding if necessary
    return unpacked[:, :original_cols]


class GPTQ:
    """
    Implementation of GPTQ (Generalized Post-Training Quantization) algorithm
    for quantizing model weights to lower bit precision.
    """

    def __init__(
            self,
            layer: nn.Module,
            wbits: int = 4,
            groupsize: int = 128,
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

        # Initialize quantization parameters
        self.H = None
        self.dead = None
        self.rows = None
        self.columns = None

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """Add a batch of data to compute Hessian matrix"""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        if self.H is None:
            self.H = torch.zeros((inp.shape[0], inp.shape[0]), device=self.device)

        self.H += 2 / tmp * inp.matmul(inp.t())

    def quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Quantize the weight matrix using GPTQ algorithm

        Returns:
            - qweight: quantized weights (packed if 4-bit)
            - qzeros: zero points for each group
            - scales: scales for each group
            - original_cols: original number of columns (for unpacking)
        """
        W = weight.clone()
        if not self.actorder:
            # Standard quantization order
            W = W.float()

        rows, columns = W.shape[0], W.shape[1]
        original_cols = columns

        # Initialize Hessian
        if self.H is None:
            self.H = torch.eye(columns, device=self.device)

        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1

        # Add dampening
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=self.device)
        H[diag, diag] += damp

        # Prepare quantization
        scales = torch.zeros((rows, (columns + self.groupsize - 1) // self.groupsize), device=self.device)
        qzeros = torch.zeros_like(scales, dtype=torch.int32)
        qweight = torch.zeros_like(W, dtype=torch.int32)

        # Cholesky decomposition
        try:
            H = torch.linalg.cholesky(H)
        except:
            print("Cholesky decomposition failed, using eigenvalue decomposition")
            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            eigenvalues = eigenvalues.clamp(min=1e-10)
            H = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T

        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # Quantize blocks
        for i1 in range(0, columns, self.blocksize):
            i2 = min(i1 + self.blocksize, columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # Find optimal quantization
                if self.groupsize != float('inf'):
                    g_idx = (i1 + i) // self.groupsize
                    scale = scales[:, g_idx]
                    zero = qzeros[:, g_idx]

                    if scale.sum() == 0:  # Initialize scale and zero
                        scale = W1[:, i].abs().max() / (self.maxq / 2)
                        scales[:, g_idx] = scale
                        zero = torch.round(-W1[:, i].min() / scale).clamp(0, self.maxq)
                        qzeros[:, g_idx] = zero.to(torch.int32)
                else:
                    scale = W1[:, i].abs().max() / (self.maxq / 2)
                    zero = torch.round(-W1[:, i].min() / scale).clamp(0, self.maxq)

                # Quantize
                q = torch.clamp(torch.round(w / scale) + zero, 0, self.maxq)
                Q1[:, i] = q

                # Dequantize and compute error
                dq = (q - zero) * scale
                err = (w - dq) / d
                Err1[:, i] = err

                # Update remaining weights
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)

            qweight[:, i1:i2] = Q1.to(torch.int32)

        # Pack weights if 4-bit
        if self.wbits == 4:
            qweight = pack_int4_weights(qweight, self.wbits)

        return qweight, qzeros, scales, original_cols


def quantize_gptq(
        model_state_dict: Dict[str, torch.Tensor],
        calibration_data: Optional[torch.Tensor] = None,
        wbits: int = 4,
        groupsize: int = 128,
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

            # If no calibration data, use identity Hessian
            if calibration_data is None:
                gptq.H = torch.eye(weight.shape[1], device=device)

            # Quantize the weight
            qweight, qzeros, scales, original_cols = gptq.quantize(weight)

            # Store quantized parameters
            base_name = name.replace(".weight", "").replace("_weight", "")
            quantized_state_dict[f"{base_name}.qweight"] = qweight.cpu()
            quantized_state_dict[f"{base_name}.qzeros"] = qzeros.cpu()
            quantized_state_dict[f"{base_name}.scales"] = scales.cpu()
            quantized_state_dict[f"{base_name}.wbits"] = torch.tensor(wbits)
            quantized_state_dict[f"{base_name}.groupsize"] = torch.tensor(groupsize)
            quantized_state_dict[f"{base_name}.original_cols"] = torch.tensor(original_cols)

        else:
            # Keep non-quantized parameters as is
            quantized_state_dict[name] = param.cpu()

    return quantized_state_dict


def dequantize_weight(qweight, qzeros, scales, wbits=4, original_cols=None):
    """
    Dequantize weight for inference

    Args:
        qweight: Quantized weights (packed if 4-bit)
        qzeros: Zero points
        scales: Scales
        wbits: Number of bits used for quantization
        original_cols: Original number of columns (for unpacking)

    Returns:
        Dequantized weight tensor
    """
    # Unpack if 4-bit
    if wbits == 4 and original_cols is not None:
        qweight = unpack_int4_weights(qweight, original_cols, wbits)

    # Get dimensions
    rows, columns = qweight.shape
    groupsize = columns // scales.shape[1]

    # Prepare output tensor
    weight = torch.zeros((rows, columns), dtype=torch.float32, device=qweight.device)

    # Dequantize each group
    for g in range(scales.shape[1]):
        start_idx = g * groupsize
        end_idx = min((g + 1) * groupsize, columns)

        # Extract group quantized values
        group_qweight = qweight[:, start_idx:end_idx].float()
        group_scales = scales[:, g].unsqueeze(1)
        group_zeros = qzeros[:, g].unsqueeze(1).float()

        # Dequantize
        weight[:, start_idx:end_idx] = (group_qweight - group_zeros) * group_scales

    return weight