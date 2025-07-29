import torch


def pack_weight(weight):
    """
    Pack two 4-bit values into one uint8 value consistently

    Args:
        weight: Tensor of shape [out_features, in_features] with values in [0, 15]

    Returns:
        packed: Tensor of shape [out_features, in_features//2] with packed values
    """
    rows, cols = weight.shape

    # Ensure even number of columns for packing
    if cols % 2 != 0:
        weight = torch.cat([weight, torch.zeros(rows, 1, dtype=weight.dtype, device=weight.device)], dim=1)
        cols += 1

    # Pack: lower 4 bits from even indices, upper 4 bits from odd indices
    # Format: [odd_value << 4] | even_value
    packed = (weight[:, 0::2] & 0xF) | ((weight[:, 1::2] & 0xF) << 4)

    return packed.contiguous().to(torch.uint8)


def unpack_weight(packed_weight, original_cols):
    """
    Unpack uint8 values back to two 4-bit values consistently

    Args:
        packed_weight: Packed tensor of shape [out_features, packed_cols]
        original_cols: Original number of columns before packing

    Returns:
        unpacked: Tensor of shape [out_features, original_cols] with unpacked values
    """
    rows, packed_cols = packed_weight.shape

    # Allocate unpacked tensor
    unpacked = torch.zeros((rows, packed_cols * 2), dtype=torch.uint8, device=packed_weight.device)

    # Unpack: even positions get lower 4 bits, odd positions get upper 4 bits
    unpacked[:, 0::2] = packed_weight & 0xF  # Lower 4 bits
    unpacked[:, 1::2] = (packed_weight >> 4) & 0xF  # Upper 4 bits

    # Trim to original size
    return unpacked[:, :original_cols].contiguous()