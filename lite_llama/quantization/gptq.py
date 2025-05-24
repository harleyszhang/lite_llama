"""
GPTQ Quantization for Lite-LLaMA Models

Key improvements in this version:
1. Proper weight packing for 4-bit quantization (2 weights per byte)
2. Stores quantized weights as uint8/int16 instead of float32
3. Separate storage of scale/zero parameters
4. Accurate compression ratio calculation
5. Support for LLaVA models with vision weight skipping
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer
import gc


def pack_4bit_weights(qweight, n_rows, n_cols):
    """Pack 4-bit weights into uint8 format (2 weights per byte)"""
    # Ensure even number of columns for packing
    if n_cols % 2 != 0:
        # Pad with zeros if odd number of columns
        qweight = torch.nn.functional.pad(qweight, (0, 1), value=0)
        n_cols += 1

    # Pack two 4-bit values into one 8-bit value
    packed = torch.zeros((n_rows, n_cols // 2), dtype=torch.uint8, device=qweight.device)
    for i in range(0, n_cols, 2):
        # First 4-bit value in lower nibble, second in upper nibble
        packed[:, i // 2] = (qweight[:, i] & 0xF) | ((qweight[:, i + 1] & 0xF) << 4)

    return packed


def unpack_4bit_weights(packed, n_rows, original_n_cols):
    """Unpack 4-bit weights from uint8 format"""
    n_packed_cols = packed.shape[1]
    unpacked = torch.zeros((n_rows, n_packed_cols * 2), dtype=torch.uint8, device=packed.device)

    for i in range(n_packed_cols):
        # Extract lower and upper nibbles
        unpacked[:, i * 2] = packed[:, i] & 0xF
        unpacked[:, i * 2 + 1] = (packed[:, i] >> 4) & 0xF

    # Remove padding if it was added
    return unpacked[:, :original_n_cols]


class GPTQ:
    """
    GPTQ Quantizer for custom lite_llama models
    """

    def __init__(
            self,
            layer,
            wbits: int = 4,
            groupsize: int = 128,
            actorder: bool = False,
            percdamp: float = 0.01,
            device: str = "cuda"
    ):
        self.layer = layer
        self.device = device
        self.wbits = wbits
        self.actorder = actorder
        self.percdamp = percdamp

        # Handle groupsize
        W = layer.weight.data
        if groupsize == -1:
            self.groupsize = W.shape[0]
        else:
            self.groupsize = groupsize

        # Check if groupsize is compatible
        if W.shape[0] % self.groupsize != 0:
            print(f"Warning: Weight dimension {W.shape[0]} not divisible by groupsize {self.groupsize}")
            print(f"Last group will have {W.shape[0] % self.groupsize} elements")

        # Calculate quantization parameters
        self.maxq = 2 ** self.wbits - 1
        self.nsamples = 0

        # Initialize Hessian and other matrices
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = None  # Will be initialized when first batch is added
        self.quantized = False

    def add_batch(self, inp):
        """Add calibration batch to compute Hessian"""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        # Update sample count
        if self.nsamples == 0:
            self.H = torch.zeros((self.columns, self.columns), device=self.device)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        # Ensure numerical stability
        inp = inp.float()

        # Add small noise for numerical stability
        inp = inp + torch.randn_like(inp) * 1e-4

        # Update Hessian
        self.H += 2 / self.nsamples * inp.matmul(inp.t())

    def quantize(self):
        """Perform GPTQ quantization"""
        W = self.layer.weight.data.clone()
        W = W.float()

        # Check if we have calibration data
        if self.H is None or self.nsamples == 0:
            print("Warning: No calibration data added, initializing with identity matrix")
            self.H = torch.eye(self.columns, device=self.device) * 0.01
            self.nsamples = 1

        # Compute inverse Hessian
        H = self.H
        del self.H

        # Add damping for numerical stability
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp

        # Try Cholesky decomposition with fallback
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
        except torch._C._LinAlgError:
            print("Warning: Cholesky decomposition failed, using eigendecomposition instead")
            # Fallback to eigendecomposition
            try:
                # Add more damping
                H[diag, diag] += damp * 10
                eigenvalues, eigenvectors = torch.linalg.eigh(H)

                # Ensure all eigenvalues are positive
                eigenvalues = eigenvalues.clamp(min=1e-5)

                # Reconstruct inverse
                Hinv = eigenvectors @ torch.diag(1.0 / eigenvalues) @ eigenvectors.T
            except:
                print("Warning: Eigendecomposition also failed, using diagonal approximation")
                # Last resort: diagonal approximation
                diagonal = torch.diag(H).clamp(min=1e-5)
                Hinv = torch.diag(1.0 / diagonal)

        # Initialize quantization parameters
        n_groups = (self.rows + self.groupsize - 1) // self.groupsize
        scale = torch.zeros((n_groups, 1), device=self.device)
        zero = torch.zeros((n_groups, 1), device=self.device)

        # Create quantized weight tensor with appropriate dtype
        if self.wbits <= 8:
            Q = torch.zeros((self.rows, self.columns), dtype=torch.uint8, device=self.device)
        else:
            # For now, use int16 for >8 bits, though this won't save space
            Q = torch.zeros((self.rows, self.columns), dtype=torch.int16, device=self.device)

        # Quantize layer weights
        for i1 in range(0, self.columns, 128):
            i2 = min(i1 + 128, self.columns)
            count = i2 - i1

            # Extract block
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # Quantize groups
                for j in range(0, self.rows, self.groupsize):
                    j2 = min(j + self.groupsize, self.rows)
                    group_idx = j // self.groupsize

                    # Find optimal scale and zero point
                    w_group = w[j:j2]

                    # Handle empty groups
                    if w_group.numel() == 0:
                        continue

                    w_min = w_group.min()
                    w_max = w_group.max()

                    # Avoid division by zero
                    if w_max == w_min:
                        scale_val = 1.0
                        zero_val = 0.0
                    else:
                        scale_val = (w_max - w_min) / self.maxq
                        zero_val = torch.round(-w_min / scale_val)

                    if group_idx < scale.shape[0]:
                        scale[group_idx] = scale_val
                        zero[group_idx] = zero_val

                    # Quantize
                    q = torch.clamp(torch.round(w_group / scale_val + zero_val), 0, self.maxq)
                    Q1[j:j2, i] = q

                    # Dequantize for error computation
                    dequant = (q - zero_val) * scale_val
                    Err1[j:j2, i] = (w_group - dequant) / d if d != 0 else 0

                # Update remaining weights
                if i + 1 < count:
                    # Ensure proper matrix multiplication dimensions
                    err_col = Err1[:, i:i + 1]  # Shape: (rows, 1)
                    hinv_row = Hinv1[i, i + 1:].unsqueeze(0)  # Shape: (1, remaining_cols)
                    update = err_col.matmul(hinv_row)  # Shape: (rows, remaining_cols)
                    W1[:, i + 1:] -= update

            # Store in compact format
            Q[:, i1:i2] = Q1.to(Q.dtype)

        # Store quantized weights in packed format
        self.qweight = Q
        self.scale = scale
        self.zero = zero
        self.quantized = True

        return Q, scale, zero


def prepare_calibration_data(
        tokenizer,
        dataset_path: str = None,
        num_samples: int = 128,
        seq_length: int = 2048
) -> List[torch.Tensor]:
    """
    Prepare calibration dataset for GPTQ

    Args:
        tokenizer: Model tokenizer
        dataset_path: Path to calibration dataset (text file)
        num_samples: Number of calibration samples
        seq_length: Sequence length for each sample

    Returns:
        List of tokenized samples
    """
    # Fix padding token issue (common with LLaMA models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            # If still None, use a common token
            tokenizer.pad_token = tokenizer.unk_token
            if tokenizer.pad_token is None:
                # Last resort - add a padding token
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if dataset_path is None:
        # Use a default calibration text if no dataset provided
        default_text = """
        The quick brown fox jumps over the lazy dog. 
        Machine learning is transforming the world of technology.
        Large language models have revolutionized natural language processing.
        Artificial intelligence is rapidly advancing across various domains.
        Deep learning has enabled breakthroughs in computer vision and NLP.
        Transformer architectures have become the foundation of modern AI.
        """ * 50

        texts = [default_text[i:i + 1000] for i in range(0, len(default_text) - 1000, 1000)][:num_samples]
    else:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
            # Split into chunks
            chunk_size = max(1000, len(text) // (num_samples + 1))
            texts = [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size, chunk_size // 2)][:num_samples]

    # Tokenize
    calibration_data = []
    for text in texts[:num_samples]:
        # Skip empty texts
        if not text.strip():
            continue

        tokens = tokenizer(
            text,
            return_tensors='pt',
            max_length=seq_length,
            truncation=True,
            padding='max_length'
        )
        calibration_data.append(tokens.input_ids)

    # Ensure we have enough samples
    if len(calibration_data) < num_samples:
        print(f"Warning: Only {len(calibration_data)} calibration samples available (requested {num_samples})")

    return calibration_data


def quantize_litellama_model(
        model_path: str,
        output_path: str,
        calibration_data_path: Optional[str] = None,
        wbits: int = 4,
        groupsize: int = 128,
        device: str = "cuda",
        num_samples: int = 128,
        seq_length: int = 2048,
        skip_vision: bool = False,
) -> None:
    """
    Main function to quantize a lite_llama model using GPTQ

    Args:
        model_path: Path to converted lite_llama model directory
        output_path: Path to save quantized model
        calibration_data_path: Path to calibration dataset
        wbits: Quantization bits (4, 8, etc.)
        groupsize: Group size for quantization
        device: Device to use for quantization
    """
    print(f"Loading model from {model_path}")

    # Load model weights
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
    if not model_files:
        raise ValueError(f"No .pth file found in {model_path}")

    model_file = os.path.join(model_path, model_files[0])
    state_dict = torch.load(model_file, map_location=device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Prepare calibration data
    print("Preparing calibration data...")
    calibration_data = prepare_calibration_data(
        tokenizer,
        calibration_data_path,
        num_samples=num_samples,
        seq_length=seq_length
    )

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Quantize each layer
    quantized_state_dict = {}
    quantization_config = {
        "wbits": wbits,
        "groupsize": groupsize,
        "layers": {}
    }

    # Detect if this is a LLaVA model by checking for language_model prefix
    is_llava = any("language_model" in key for key in state_dict.keys())

    # Get all weight keys that need quantization
    weight_keys_to_quantize = []

    # Updated patterns for both regular and LLaVA models
    patterns = [
        # Regular model patterns
        "q_proj", "kv_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head", "embed_tokens",
        # LLaVA specific patterns (without _weight suffix in search)
        "q_proj_weight", "kv_proj_weight", "o_proj_weight"
    ]

    for key in state_dict.keys():
        # For LLaVA models, also check for language_model prefix
        if is_llava and "language_model" in key:
            if any(pattern in key for pattern in patterns):
                weight_keys_to_quantize.append(key)
        elif any(pattern in key for pattern in patterns) and ("weight" in key or key.endswith(("_weight", ".weight"))):
            weight_keys_to_quantize.append(key)

    print(f"Found {len(weight_keys_to_quantize)} weights to quantize")
    if is_llava:
        print("Detected LLaVA model structure")

    # Process each weight
    skipped_vision_count = 0
    for key in tqdm(weight_keys_to_quantize, desc="Quantizing layers"):
        weight = state_dict[key]
        is_vision = "vision" in key.lower() or "multi_modal_projector" in key.lower()

        # Skip if weight is too small
        if weight.numel() < 1024:
            print(f"\nSkipping {key} (too small: {weight.numel()} parameters)")
            quantized_state_dict[key] = weight
            continue

        # Skip vision weights if requested
        if skip_vision and is_vision:
            quantized_state_dict[key] = weight
            skipped_vision_count += 1
            continue

        print(f"\nQuantizing {key} (shape: {weight.shape})...")

        # Create a dummy layer for GPTQ
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        layer.weight.data = weight.to(device)

        # Adjust percdamp for different layer types
        percdamp = 0.01
        if "embed" in key or "lm_head" in key:
            percdamp = 0.1  # Higher damping for embeddings
            print(f"  Using higher damping (0.1) for {key}")

        # Initialize GPTQ
        gptq = GPTQ(
            layer=layer,
            wbits=wbits,
            groupsize=groupsize,
            device=device,
            percdamp=percdamp
        )

        # Add calibration data (simplified - in practice, you'd run forward passes)
        # Get embedding weight if available
        embed_key = None

        # Search for embedding key - handle both regular and LLaVA models
        embed_patterns = [
            "embed_tokens.weight",
            "language_model.embed_tokens.weight"
        ]

        for k in state_dict.keys():
            for pattern in embed_patterns:
                if pattern in k:
                    embed_key = k
                    break
            if embed_key:
                break

        if embed_key and len(calibration_data) > 0:
            embed_weight = state_dict[embed_key].to(device)
            # Use actual token embeddings as input
            for i in range(min(len(calibration_data), 32)):
                tokens = calibration_data[i][0].to(device)
                # Get embeddings for these tokens
                embeddings = torch.embedding(embed_weight, tokens)
                # Average pool to get input dimension
                if embeddings.shape[1] > weight.shape[1]:
                    # Use adaptive pooling to match dimensions
                    embeddings = torch.nn.functional.adaptive_avg_pool1d(
                        embeddings.transpose(1, 2),
                        weight.shape[1]
                    ).transpose(1, 2)
                elif embeddings.shape[1] < weight.shape[1]:
                    # Skip if embedding dimension doesn't match
                    continue

                # Take mean across sequence length for this layer's input
                fake_inp = embeddings.mean(dim=0, keepdim=True)
                if fake_inp.shape[1] == weight.shape[1]:
                    gptq.add_batch(fake_inp)
        else:
            # Fallback to random data if no embeddings available
            for _ in range(min(len(calibration_data), 32)):
                fake_inp = torch.randn(1, weight.shape[1], device=device) * 0.1
                gptq.add_batch(fake_inp)

        # Quantize
        qweight, scale, zero = gptq.quantize()

        # Pack weights if 4-bit quantization
        if wbits == 4:
            # Store original shape for unpacking
            original_shape = qweight.shape
            packed_weight = pack_4bit_weights(qweight, qweight.shape[0], qweight.shape[1])
            quantized_state_dict[key] = packed_weight.cpu()
            # Store scale and zero as tensors with specific keys
            quantized_state_dict[f"{key}.scale"] = scale.cpu()
            quantized_state_dict[f"{key}.zero"] = zero.cpu()
            quantization_config["layers"][key] = {
                "groupsize": groupsize,
                "wbits": wbits,
                "original_shape": list(original_shape),
                "packed": True
            }
        else:
            # For 8-bit or other, store directly
            quantized_state_dict[key] = qweight.cpu()
            quantized_state_dict[f"{key}.scale"] = scale.cpu()
            quantized_state_dict[f"{key}.zero"] = zero.cpu()
            quantization_config["layers"][key] = {
                "groupsize": groupsize,
                "wbits": wbits,
                "packed": False
            }

        # Clean up
        del layer, gptq
        torch.cuda.empty_cache()
        gc.collect()

    # Copy non-quantized weights
    for key in state_dict.keys():
        if key not in quantized_state_dict:
            quantized_state_dict[key] = state_dict[key]

    # Save quantized model - now everything is in the state dict
    model_id = os.path.basename(model_path)
    torch.save(
        quantized_state_dict,
        os.path.join(output_path, f"{model_id}-{wbits}bit-gptq.pth")
    )

    # Save quantization config
    with open(os.path.join(output_path, "quantization_config.json"), "w") as f:
        json.dump(quantization_config, f, indent=2)

    # Copy other files
    for file in os.listdir(model_path):
        if file.endswith('.json') and file != "quantization_config.json":
            src = os.path.join(model_path, file)
            dst = os.path.join(output_path, file)
            with open(src, 'r') as f_in, open(dst, 'w') as f_out:
                f_out.write(f_in.read())

    if os.path.exists(os.path.join(model_path, "tokenizer.model")):
        import shutil
        shutil.copy(
            os.path.join(model_path, "tokenizer.model"),
            os.path.join(output_path, "tokenizer.model")
        )

    print(f"Quantization complete! Model saved to {output_path}")

    if skipped_vision_count > 0:
        print(f"Skipped {skipped_vision_count} vision model weights")

    # Print compression statistics
    original_size = sum(p.numel() * p.element_size() for p in state_dict.values())

    # Calculate quantized size more accurately
    quantized_size = 0
    for key, tensor in quantized_state_dict.items():
        # Add size of each tensor in the quantized state dict
        quantized_size += tensor.numel() * tensor.element_size()

    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0

    print(f"\nOriginal model size: {original_size / 1e9:.2f} GB")
    print(f"Quantized model size: {quantized_size / 1e9:.2f} GB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Space saved: {(1 - quantized_size / original_size) * 100:.1f}%")

    # Expected compression ratios
    expected_ratio = 32 / (wbits + 0.5)  # 0.5 for metadata overhead
    if compression_ratio < expected_ratio * 0.8:
        print(f"\nNote: Compression ratio is lower than expected ({expected_ratio:.1f}x)")
        print("This may be due to non-quantized layers (embeddings, layer norms, etc.)")


def dequantize_weight(
        quantized_weight: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        wbits: int = 4,
        groupsize: int = 128,
        original_shape: Optional[Tuple[int, int]] = None,
        packed: bool = False
) -> torch.Tensor:
    """
    Dequantize a weight tensor

    Args:
        quantized_weight: Quantized weight tensor (possibly packed)
        scale: Scale parameters
        zero: Zero point parameters
        wbits: Quantization bits
        groupsize: Group size used in quantization
        original_shape: Original shape before packing (for 4-bit)
        packed: Whether the weights are packed

    Returns:
        Dequantized weight tensor
    """
    # Unpack if necessary
    if packed and wbits == 4:
        if original_shape is None:
            raise ValueError("original_shape required for unpacking 4-bit weights")
        quantized_weight = unpack_4bit_weights(
            quantized_weight,
            original_shape[0],
            original_shape[1]
        )

    weight = torch.zeros_like(quantized_weight, dtype=torch.float32)

    for i in range(0, quantized_weight.shape[0], groupsize):
        j = min(i + groupsize, quantized_weight.shape[0])
        group_idx = i // groupsize

        if group_idx < scale.shape[0]:
            weight[i:j] = (quantized_weight[i:j].float() - zero[group_idx]) * scale[group_idx]

    return weight


def load_quantized_model(model_path: str, device: str = "cpu"):
    """
    Load a quantized model

    Args:
        model_path: Path to the quantized .pth file
        device: Device to load the model to

    Returns:
        state_dict with quantized weights and metadata
    """
    return torch.load(model_path, map_location=device)


def dequantize_model(model_path: str, quantization_config: dict, output_path: str = None):
    """
    Fully dequantize a quantized model back to fp16/fp32

    Args:
        model_path: Path to quantized model
        quantization_config: Quantization configuration dict
        output_path: Where to save dequantized model
    """
    # Load quantized model
    state_dict = load_quantized_model(model_path)

    # Dequantize each layer
    dequantized_dict = {}
    for key, tensor in state_dict.items():
        # Skip scale and zero tensors
        if key.endswith('.scale') or key.endswith('.zero'):
            continue

        # Check if this is a quantized layer
        if key in quantization_config["layers"]:
            layer_config = quantization_config["layers"][key]
            scale = state_dict[f"{key}.scale"]
            zero = state_dict[f"{key}.zero"]

            dequantized = dequantize_weight(
                tensor,
                scale,
                zero,
                wbits=layer_config["wbits"],
                groupsize=layer_config["groupsize"],
                original_shape=layer_config.get("original_shape"),
                packed=layer_config.get("packed", False)
            )
            dequantized_dict[key] = dequantized
        else:
            dequantized_dict[key] = tensor

    # Save if output path provided
    if output_path:
        torch.save(dequantized_dict, output_path)
        print(f"Dequantized model saved to {output_path}")

    return dequantized_dict


# Integration with your existing code
def quantize_after_conversion(
        checkpoints_dir: str,
        model_type: str,  # "llama", "qwen2", or "llava"
        calibration_data_path: Optional[str] = None,
        wbits: int = 4,
        groupsize: int = 128,
        num_samples: int = 128,
        seq_length: int = 2048,
        skip_vision: bool = False
):
    """
    Quantize model after it has been converted to lite_llama format

    Args:
        checkpoints_dir: Original HF model directory
        model_type: Type of model ("llama", "qwen2", or "llava")
        calibration_data_path: Path to calibration dataset
        wbits: Quantization bits
        groupsize: Group size for quantization (0 for auto-detect)
        num_samples: Number of calibration samples
        seq_length: Sequence length for calibration
        skip_vision: Skip quantization of vision weights (for LLaVA)
    """
    # Construct paths
    model_id = os.path.basename(os.path.normpath(checkpoints_dir))
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to converted model
    converted_model_path = os.path.join(current_dir, f"../../my_weight/{model_id}")

    # Path for quantized model
    quantized_model_path = os.path.join(current_dir, f"../../my_weight/{model_id}-{wbits}bit-gptq")

    # Perform quantization
    quantize_litellama_model(
        model_path=converted_model_path,
        output_path=quantized_model_path,
        calibration_data_path=calibration_data_path,
        wbits=wbits,
        groupsize=groupsize,
        num_samples=num_samples,
        seq_length=seq_length,
        skip_vision=skip_vision
    )

    return quantized_model_path