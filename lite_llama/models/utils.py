import itertools
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Literal, Mapping,
                    Optional, Protocol, Tuple, Union, overload)

import torch
import torch.nn as nn
from torch.func import functional_call
from transformers import PretrainedConfig

NestedTensors = Union[List["NestedTensors"], List[torch.Tensor], torch.Tensor]
"""
Uses a list instead of a tensor if the dimensions of each element do not match.
"""

def weak_ref_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    """
    return torch.ops._C.weak_ref_tensor(tensor)

def _flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """
    递归地将嵌套的张量结构 (NestedTensors) 在最后一个维度之外的所有维度展平, 并将它们连接成一个单一的二维张量。
    """

    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the NestedTensors.
    """

    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(
        _embedding_count_expression(inner) for inner in embeddings)

def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    is_multimodal: torch.Tensor,
    multimodal_embeddings: NestedTensors,
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    num_expected_tokens = is_multimodal.sum().item()
    assert isinstance(num_expected_tokens, int)

    flattened = _flatten_embeddings(multimodal_embeddings)
    if flattened.shape[0] != num_expected_tokens:
        expr = _embedding_count_expression(multimodal_embeddings)
        raise ValueError(
            f"Attempted to assign {expr} = {flattened.shape[0]} "
            f"multimodal tokens to {num_expected_tokens} placeholders")

    inputs_embeds[is_multimodal] = flattened
    return inputs_embeds

# def _merge_multimodal_embeddings(
#     inputs_embeds: torch.Tensor,
#     is_multimodal: torch.Tensor,
#     multimodal_embeddings: NestedTensors,
# ) -> torch.Tensor:
#     num_expected_tokens = is_multimodal.sum().item()
#     assert isinstance(num_expected_tokens, int)

#     flattened = _flatten_embeddings(multimodal_embeddings)
#     print(f"Attempted to assign {num_expected_tokens} = {flattened.shape[0]} \
#           multimodal tokens to {num_expected_tokens} placeholders")
    
#     if flattened.shape[0] != num_expected_tokens:
#         expr = _embedding_count_expression(multimodal_embeddings)
#         raise ValueError(
#             f"Attempted to assign {expr} = {flattened.shape[0]} "
#             f"multimodal tokens to {num_expected_tokens} placeholders")
    
#     # Ensure that the assignment is valid
#     if flattened.shape[0] > num_expected_tokens:
#         # Option 1: Truncate the embeddings
#         flattened = flattened[:num_expected_tokens]
#     elif flattened.shape[0] < num_expected_tokens:
#         # Option 2: Repeat or pad the embeddings
#         # 例如，可以重复使用某些嵌入或使用零填充
#         padding = inputs_embeds.new_zeros(num_expected_tokens - flattened.shape[0], flattened.shape[1])
#         flattened = torch.cat([flattened, padding], dim=0)

#     inputs_embeds[is_multimodal] = flattened
#     return inputs_embeds

def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    placeholder_token_id: int,
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    return _merge_multimodal_embeddings(
        inputs_embeds,
        (input_ids == placeholder_token_id),
        multimodal_embeddings,
    )

def embed_multimodal(
    input_ids: torch.Tensor,
    multimodal_token_id: int,
    get_text_embeds: Callable[[torch.Tensor], torch.Tensor],
    get_multimodal_embeds: Callable[[torch.Tensor], Union[torch.Tensor,
                                                          List[torch.Tensor]]],
) -> torch.Tensor:
    """
    Embed token IDs and multimodal inputs and combine their embeddings.

    ``multimodal_token_id`` is used to determine whether a token ID should
    be embedded using ``get_text_embeds`` or ``get_multimodal_embeds``.

    Compared to ``merge_multimodal_embeddings`, this avoids running
    ``get_text_embeds`` on ``input_ids[input_ids == multimodal_token_id]``
    which causes issues when the placeholder token ID exceeds the
    vocabulary size of the language model.
    """
    is_multimodal = input_ids == multimodal_token_id
    is_text = ~is_multimodal

    text_embeds = get_text_embeds(input_ids[is_text])
    multimodal_embeds = get_multimodal_embeds(input_ids[is_multimodal])

    merged_embeds = torch.empty(
        (input_ids.shape[0], text_embeds.shape[1]),
        dtype=text_embeds.dtype,
        device=text_embeds.device,
    )

    merged_embeds[is_text] = text_embeds

    return _merge_multimodal_embeddings(
        merged_embeds,
        is_multimodal,
        multimodal_embeds,
    )


def merge_input_ids_with_image_features2(
    image_features, 
    inputs_embeds, 
    input_ids, 
    attention_mask,
    pad_token_id,
    image_token_index
):
    num_images, num_image_patches, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    # NOTE: 检查每个样本的最后一个 token 是否为填充 token
    # NOTE: 如果最后一个 token 不是填充 token，则为 True，表示存在左侧填充；否则为 False。
    left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))
    
    # 1. 创建图像 token 的掩码来获取特殊图像 token 的位置, 并计算新序列最大长度
    # NOTE: 一个布尔张量，标识 input_ids 中哪些位置是图像 token（即等于 image_token_index 的位置）
    special_image_token_mask = input_ids == image_token_index
    # NOTE: 每个样本中图像 token 的数量, 形状为 [batch_size, ]
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    
    # 计算合并图像特征后的新序列最大长度。
    # NOTE: 每个图像 token 位置会被替换为 (num_image_patches - 1) 个图像 paches embedding token。
    max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
    # NOTE: 通过 torch.where 获取所有非图像 token 的位置索引。
    # NOTE: 当仅提供 condition 参数时，torch.where 等同于 torch.nonzero(condition, as_tuple=True)，返回满足条件的元素的索引。
    batch_indices, non_image_indices = torch.where(input_ids != image_token_index) # 满足条件的样本索引和序列 token 索引

    # 2. 计算文本应写入的位置
    # NOTE: 每个图像 token 会增加 (num_image_patches - 1) 个位置。
    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1] # 计算需要的图像填充数量，以达到 max_embed_dim。
    # 如果存在左侧填充 (left_padding 为 True)，则将 new_token_positions 进行偏移调整。
    if left_padding:
        new_token_positions += nb_image_pad[:, None]  # offset for left padding
    # NOTE: 获取需要覆盖的文本 token 在合并序列中的新位置。
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    # 3. 初始化最终的嵌入与注意力掩码
    final_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    final_attention_mask = torch.zeros(
        batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
    )
    
    # NOTE: 如果视觉模型或语言模型已卸载到 CPU，我们需要手动将相应的张量设置到正确的目标设备中。
    target_device = inputs_embeds.device
    batch_indices, non_image_indices, text_to_overwrite = (
        batch_indices.to(target_device),
        non_image_indices.to(target_device),
        text_to_overwrite.to(target_device),
    )
    attention_mask = attention_mask.to(target_device)

    # 4. 填充文本嵌入与注意力掩码. 
    # If we have ["hey" "<image>", "how", "are"]. we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
    # NOTE: 使用 batch_indices 和 text_to_overwrite 将 inputs_embeds 中的非图像 token 嵌入复制到 final_embedding 的相应位置。
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]

    # 5. 填充图像特征与更新注意力掩码和位置 ID.
    image_to_overwrite = torch.all(final_embedding == 0, dim=-1) # 找出 final_embedding 中所有维度为0的位置（即尚未填充的地方）。
    # NOTE: 使用 cumsum 计算累积和，确保这些位置在填充数量 (nb_image_pad) 之后。
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

    if image_to_overwrite.sum() != image_features.shape[:-1].numel():  # 如果需要填充的位置数量不等于 image_features 的数量，抛出错误。
        raise ValueError(      
            f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
            f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
        )

    # NOTE: 将 image_features 重新排列为 (batch_size * num_image_patches, embed_dim)，并填充到 final_embedding 的相应位置。
    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
    final_attention_mask |= image_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

    # 6. 处理填充位置的嵌入, 将填充位置的嵌入设为0：
    batch_indices, pad_indices = torch.where(input_ids == pad_token_id)
    indices_to_mask = new_token_positions[batch_indices, pad_indices]

    final_embedding[batch_indices, indices_to_mask] = 0

    return final_embedding, final_attention_mask, position_ids

def merge_input_ids_with_image_features(
    input_ids: torch.Tensor, 
    inputs_embeds: torch.Tensor, 
    image_features: torch.Tensor,
    pad_token_id: int,
    image_token_index: int
):
    """
    将 input_ids 与 image_features 合并，生成最终的嵌入和位置 ID。
    
    Args:
        input_ids (torch.Tensor): 输入的 token IDs, 形状为 (batch_size, sequence_length)
        inputs_embeds (torch.Tensor): 文本嵌入，形状为 (batch_size, sequence_length, embed_dim)
        image_features (torch.Tensor): 视觉编码后的图像特征，形状为 (num_images, num_image_patches, embed_dim)
        pad_token_id (int): 填充 token 的 ID
        image_token_index (int): 图像 token 的 ID
    
    Returns:
        final_embedding (torch.Tensor): 合并后的嵌入张量，形状为 (batch_size, max_embed_dim, embed_dim)
        position_ids (torch.Tensor): 位置 ID, 形状为 (batch_size, max_embed_dim)
    """
    target_device = input_ids.device
    # 1, 基础 shape 信息提取
    num_images, num_image_patches, embed_dim = image_features.shape # torch.Size([1, 576, 4096])
    batch_size, sequence_length = input_ids.shape # torch.Size([1, 22])

    # 2, 掩码与填充处理
    attention_mask = (input_ids != pad_token_id).long()
    left_padding = not torch.sum(input_ids[:, -1] == pad_token_id).bool().any()
    batch_image_token_mask  = input_ids == image_token_index
    
    # 3, 计算新序列长度
    num_special_image_tokens = torch.sum(batch_image_token_mask , dim=-1) # 统计每个样本（batch 中每条序列）里出现了多少个“图像占位符” token。
    max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
    batch_indices, non_image_indices = torch.where(input_ids != image_token_index) 

    # 4, 位置映射计算
    # 得到每个原始 token 在新序列中占据的开始位置索引。
    new_token_positions = torch.cumsum((batch_image_token_mask  * (num_image_patches - 1) + 1), -1) - 1 
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1] 
    if left_padding:
        new_token_positions += nb_image_pad[:, None]  # offset for left padding
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]
    
    # 5，构建融合张量
    final_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices] # 填充文本嵌入

    # 确定图像特征插入位置，通过找到 final_embedding 中所有全 0 的位置
    image_to_overwrite = torch.all(final_embedding == 0, dim=-1)  # 找出 final_embedding 中所有维度为0的位置
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

    # 将 image_features 重新排列为 (num_images * num_image_patches, embed_dim)，并填充到 final_embedding 的相应位置。
    final_embedding[image_to_overwrite] = image_features.contiguous().view(-1, embed_dim).to(target_device)
    
    # 6，生成新的 position_ids
    position_ids = torch.arange(max_embed_dim, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)

    # 7，处理填充位置的嵌入, 将填充位置的嵌入设为0
    batch_indices_pad, pad_indices = torch.where(input_ids == pad_token_id)
    indices_to_mask = new_token_positions[batch_indices_pad, pad_indices]

    final_embedding[batch_indices_pad, indices_to_mask] = 0

    return final_embedding, position_ids

def unit_test_merge_input_ids_with_image_features():
    """
    单元测试函数，测试 merge_input_ids_with_image_features 的各种场景。
    """
    # 初始化配置
    pad_token_id = 0
    image_token_index = 999

    # 示例1：统一尺寸的 image_features
    print("=== 示例1: 统一尺寸的 image_features ===")
    batch_size = 2
    # 计算总图像 token 数量
    input_ids = torch.tensor([
        [101, 102, 999, 103, 104],
        [201, 999, 202, 999, 203]
    ], dtype=torch.long)
    num_image_tokens = torch.sum(input_ids == image_token_index).item()  # 3

    num_images = num_image_tokens  # 3
    num_image_patches = 4
    embed_dim = 768
    sequence_length = 5

    # inputs_embeds: batch_size x sequence_length x embed_dim
    inputs_embeds = torch.randn(batch_size, sequence_length, embed_dim)

    # image_features: num_images x num_image_patches x embed_dim
    image_features = torch.randn(num_images, num_image_patches, embed_dim)

    # 合并
    final_embedding, position_ids = merge_input_ids_with_image_features(
        image_features=image_features,
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        pad_token_id=pad_token_id,
        image_token_index=image_token_index
    )

    print("Final Embedding Shape:", final_embedding.shape)  # Expected: (2, 13, 768)
    print("Position IDs Shape:", position_ids.shape)      # Expected: (2, 13)
    print()

    # 示例2：没有图像输入
    print("=== 示例2：没有图像输入 ===")
    # image_features 为一个空的张量，形状为 (0, 0, embed_dim)
    image_features_empty = torch.tensor([]).reshape(0, 0, embed_dim)

    # input_ids 不包含任何图像 token
    input_ids_no_image = torch.tensor([
        [101, 102, 103, 104, 105],
        [201, 202, 203, 204, 205]
    ], dtype=torch.long)

    num_image_tokens_no_image = torch.sum(input_ids_no_image == image_token_index).item()  # 0

    num_images_no_image = num_image_tokens_no_image  # 0
    sequence_length_no_image = 5

    inputs_embeds_no_image = torch.randn(batch_size, sequence_length_no_image, embed_dim)

    final_embedding_empty, position_ids_empty = merge_input_ids_with_image_features(
        image_features=image_features_empty,
        inputs_embeds=inputs_embeds_no_image,
        input_ids=input_ids_no_image,
        pad_token_id=pad_token_id,
        image_token_index=image_token_index
    )

    print("Final Embedding Shape (Empty):", final_embedding_empty.shape)  # Expected: (2, 5, 768)
    print("Position IDs Shape (Empty):", position_ids_empty.shape)        # Expected: (2, 5)
    print()

    # 示例3：错误的 image_features 类型
    print("=== 示例3：错误的 image_features 类型 ===")
    try:
        # image_features 不是 tensor
        image_features_invalid = "invalid_image_features"
        final_embedding_invalid, position_ids_invalid = merge_input_ids_with_image_features(
            image_features=image_features_invalid,  # 传入字符串，应该是 tensor
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            pad_token_id=pad_token_id,
            image_token_index=image_token_index
        )
    except Exception as e:
        print(f"Caught Exception: {e}")
    print()

    # 示例4：image_features 与图像 token 数量不匹配
    print("=== 示例4：image_features 与图像 token 数量不匹配 ===")
    try:
        # input_ids_mismatch 中有 7 个 image tokens
        input_ids_mismatch = torch.tensor([
            [101, 999, 999, 999, 104],
            [999, 999, 999, 999, 203]
        ], dtype=torch.long)
        num_image_tokens_mismatch = torch.sum(input_ids_mismatch == image_token_index).item()  # 7

        num_images_mismatch = num_image_tokens_mismatch  # 7
        num_image_patches_mismatch = 4
        embed_dim_mismatch = 768

        image_features_mismatch = torch.randn(num_images_mismatch, num_image_patches_mismatch, embed_dim_mismatch)  # 正确

        final_embedding_mismatch, position_ids_mismatch = merge_input_ids_with_image_features(
            image_features=image_features_mismatch,
            inputs_embeds=torch.randn(2, 5, embed_dim_mismatch),
            input_ids=input_ids_mismatch,
            pad_token_id=pad_token_id,
            image_token_index=image_token_index
        )
    except ValueError as e:
        print(f"Caught ValueError: {e}")
    print()

    # 示例5：单个样本，单个图像 token
    print("=== 示例5：单个样本，单个图像 token ===")
    batch_size_single = 1
    input_ids_single = torch.tensor([
        [101, 999, 102, 103]
    ], dtype=torch.long)
    num_image_tokens_single = torch.sum(input_ids_single == image_token_index).item()  # 1

    num_images_single = num_image_tokens_single  # 1
    num_image_patches_single = 3
    embed_dim_single = 768
    sequence_length_single = 4

    inputs_embeds_single = torch.randn(batch_size_single, sequence_length_single, embed_dim_single)

    image_features_single = torch.randn(num_images_single, num_image_patches_single, embed_dim_single)

    final_embedding_single, position_ids_single = merge_input_ids_with_image_features(
        image_features=image_features_single,
        inputs_embeds=inputs_embeds_single,
        input_ids=input_ids_single,
        pad_token_id=pad_token_id,
        image_token_index=image_token_index
    )

    print("Final Embedding Shape (Single):", final_embedding_single.shape)  # Expected: (1, 6, 768)
    print("Position IDs Shape (Single):", position_ids_single.shape)      # Expected: (1, 6)
    print()

if __name__ == "__main__":
    unit_test_merge_input_ids_with_image_features()

