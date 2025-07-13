from dataclasses import dataclass, field
from typing import List


@dataclass
class AWQConfig:

    """Configuration for AWQ quantization"""
    w_bit: int = 4  # Weight quantization bits
    group_size: int = 128  # Group size for quantization
    zero_point: bool = True  # Whether to use zero point
    version: str = "GEMM"  # GEMM or GEMV
    calib_data_size: int = 128  # Calibration dataset size
    search_scale: bool = False  # Whether to search for optimal scales
    auto_scale: bool = True  # Automatic scaling
    device: str = "cuda"
    alpha = 0.5


@dataclass
class GPTQConfig:
    """Configuration for AWQ quantization"""
    w_bit: int = 4  # Weight quantization bits
    group_size: int = 128  # Group size for quantization
    device: str = "cuda"


@dataclass
class SmoothQuantConfig:
    """Configuration for SmoothQuant"""
    alpha: float = 0.5  # Smoothing factor balance between act and weight
    w_bit: int = 8      # Weight quantization bits
    a_bit: int = 8      # Activation quantization bits
    device: str = "cuda"
    symmetric_weight: bool = True    # Use symmetric quantization for weights
    symmetric_activation: bool = False  # Use asymmetric quantization for activations
    per_channel_weight: bool = True  # Per-channel quantization for weights
    per_token_activation: bool = True  # Per-token quantization for activations
    calibration_samples: int = 128   # Number of calibration samples
    smooth_layers: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])