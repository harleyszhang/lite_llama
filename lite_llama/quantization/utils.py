import torch
from .quant_config import GPTQConfig

def pack_weight(weight):
    """Pack two 4-bit values into one uint8 value"""
    rows, cols = weight.shape
    if cols % 2 != 0:
        weight = torch.cat([weight, torch.zeros(rows, 1, dtype=weight.dtype, device=weight.device)], dim=1)
        cols += 1
    packed = (weight[:, 0::2] & 0xF) | ((weight[:, 1::2] & 0xF) << 4)
    return packed.contiguous()

def unpack_weight(packed_weight, original_cols):
    """Unpack uint8 values back to two 4-bit values"""
    rows, packed_cols = packed_weight.shape
    unpacked = torch.zeros((rows, packed_cols * 2), dtype=torch.uint8, device=packed_weight.device)
    unpacked[:, 0::2] = packed_weight & 0xF
    unpacked[:, 1::2] = (packed_weight >> 4) & 0xF
    return unpacked[:, :original_cols].contiguous()