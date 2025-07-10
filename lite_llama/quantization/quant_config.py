from dataclasses import dataclass


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
