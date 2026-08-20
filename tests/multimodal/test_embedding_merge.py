import unittest
import torch

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def merge_input_ids_with_image_features(
    image_features: torch.Tensor,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
    image_token_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 input_ids 与 image_features 合并，生成最终的嵌入和位置 ID。

    Args:
        image_features (torch.Tensor): 视觉编码后的图像特征，形状为 (batch_size, num_image_patches, embed_dim)
        inputs_embeds (torch.Tensor): 文本嵌入，形状为 (batch_size, sequence_length, embed_dim)
        input_ids (torch.Tensor): 输入的 token IDs, 形状为 (batch_size, sequence_length)
        pad_token_id (int): 填充 token 的 ID
        image_token_index (int): 图像 token 的 ID

    Returns:
        final_embedding (torch.Tensor): 合并后的嵌入，形状为 (batch_size, max_embed_dim, embed_dim)
        position_ids (torch.Tensor): 位置 ID, 形状为 (batch_size, max_embed_dim)
    """
    batch_size, sequence_length = input_ids.shape
    _, num_image_patches, embed_dim = image_features.shape

    # 创建 attention_mask 从 input_ids
    attention_mask = (input_ids != pad_token_id).long()

    # 创建图像 token 的掩码
    special_image_token_mask = input_ids == image_token_index

    # 每个样本中图像 token 的数量
    num_special_image_tokens = special_image_token_mask.sum(
        dim=1
    )  # shape: (batch_size,)

    # 计算每个样本的新序列长度
    new_sequence_length_per_sample = sequence_length + num_special_image_tokens * (
        num_image_patches - 1
    )

    # 获取批次中最大的序列长度
    max_embed_dim = new_sequence_length_per_sample.max().item()

    # 初始化最终的嵌入
    final_embedding = torch.zeros(
        (batch_size, max_embed_dim, embed_dim),
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )

    # 初始化 position_ids
    position_ids = (
        torch.arange(max_embed_dim, dtype=torch.long, device=inputs_embeds.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    for i in range(batch_size):
        curr_pos = 0
        for j in range(sequence_length):
            if special_image_token_mask[i, j]:
                # 插入图像特征
                if curr_pos + num_image_patches > max_embed_dim:
                    raise ValueError(f"Sample {i} exceeds max_embed_dim.")
                final_embedding[i, curr_pos : curr_pos + num_image_patches, :] = (
                    image_features[i]
                )
                curr_pos += num_image_patches
            else:
                if curr_pos >= max_embed_dim:
                    raise ValueError(f"Sample {i} exceeds max_embed_dim.")
                final_embedding[i, curr_pos, :] = inputs_embeds[i, j, :]
                curr_pos += 1
        # 剩余位置已被初始化为0（填充）

    # 处理 pad_token_id，将对应位置的嵌入设为0
    batch_indices_pad, pad_indices = torch.where(input_ids == pad_token_id)
    for idx in range(batch_indices_pad.size(0)):
        sample = batch_indices_pad[idx]
        position = pad_indices[idx]
        # 计算新位置
        new_position = (
            torch.sum(
                special_image_token_mask[sample, :position] * (num_image_patches - 1)
            ).item()
            + position
        )
        if new_position < max_embed_dim:
            final_embedding[sample, new_position, :] = 0.0
        else:
            logger.warning(
                f"Pad token position {position} exceeds max_embed_dim for sample {sample}"
            )

    return final_embedding, position_ids


class TestMergeInputIdsWithImageFeatures(unittest.TestCase):
    def test_merge_basic(self):
        # 定义参数
        batch_size = 2
        sequence_length = 5
        num_image_patches = 3
        embed_dim = 4
        pad_token_id = 0
        image_token_index = 1

        # 创建 mock input_ids
        # 样本1: [2, 1, 3, 1, 4] - 两个图像 token
        # 样本2: [1, 2, 3, 0, 0] - 一个图像 token，两个 pad token
        input_ids = torch.tensor([[2, 1, 3, 1, 4], [1, 2, 3, 0, 0]], dtype=torch.long)

        # 创建 mock inputs_embeds
        inputs_embeds = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2],
                    [1.3, 1.4, 1.5, 1.6],
                    [1.7, 1.8, 1.9, 2.0],
                ],
                [
                    [2.1, 2.2, 2.3, 2.4],
                    [2.5, 2.6, 2.7, 2.8],
                    [2.9, 3.0, 3.1, 3.2],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float,
        )

        # 创建 mock image_features
        # 样本1: 两个图像 token，每个替换为3个图像 patches
        # 样本2: 一个图像 token，替换为3个图像 patches
        image_features = torch.tensor(
            [
                [
                    [10.1, 10.2, 10.3, 10.4],
                    [10.5, 10.6, 10.7, 10.8],
                    [10.9, 11.0, 11.1, 11.2],
                ],
                [
                    [20.1, 20.2, 20.3, 20.4],
                    [20.5, 20.6, 20.7, 20.8],
                    [20.9, 21.0, 21.1, 21.2],
                ],
            ],
            dtype=torch.float,
        )

        # 调用函数，确保参数顺序正确
        final_embedding, position_ids = merge_input_ids_with_image_features(
            image_features=image_features,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            pad_token_id=pad_token_id,
            image_token_index=image_token_index,
        )

        # 定义预期的形状
        # 样本1: original sequence_length=5, num_special_image_tokens=2
        # new_sequence_length=5 +2*(3-1)=9
        # 样本2: original sequence_length=5, num_special_image_tokens=1
        # new_sequence_length=5 +1*(3-1)=7
        # max_embed_dim=9

        expected_final_embedding_shape = (batch_size, 9, embed_dim)
        expected_position_ids_shape = (batch_size, 9)

        self.assertEqual(final_embedding.shape, expected_final_embedding_shape)
        self.assertEqual(position_ids.shape, expected_position_ids_shape)

        # 检查样本1
        # Original: [2, 1, 3, 1, 4]
        # Replaced: [2, image_features, 3, image_features, 4]
        # Expected final_embedding:
        # [0] = inputs_embeds[0,0]
        # [1,2,3] = image_features[0]
        # [4] = inputs_embeds[0,2]
        # [5,6,7] = image_features[0]
        # [8] = inputs_embeds[0,4]
        expected_sample1 = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [10.1, 10.2, 10.3, 10.4],
                [10.5, 10.6, 10.7, 10.8],
                [10.9, 11.0, 11.1, 11.2],
                [0.9, 1.0, 1.1, 1.2],
                [10.1, 10.2, 10.3, 10.4],
                [10.5, 10.6, 10.7, 10.8],
                [10.9, 11.0, 11.1, 11.2],
                [1.7, 1.8, 1.9, 2.0],
            ],
            dtype=torch.float,
        )

        self.assertTrue(torch.allclose(final_embedding[0], expected_sample1))

        # 检查样本2
        # Original: [1, 2, 3, 0, 0]
        # Replaced: [image_features, 2, 3, 0, 0]
        # Expected final_embedding:
        # [0,1,2] = image_features[1]
        # [3] = inputs_embeds[1,1]
        # [4] = inputs_embeds[1,2]
        # [5,6,7,8] = pad (already zero)
        expected_sample2 = torch.tensor(
            [
                [20.1, 20.2, 20.3, 20.4],
                [20.5, 20.6, 20.7, 20.8],
                [20.9, 21.0, 21.1, 21.2],
                [2.5, 2.6, 2.7, 2.8],
                [2.9, 3.0, 3.1, 3.2],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float,
        )

        self.assertTrue(torch.allclose(final_embedding[1], expected_sample2))

    def test_merge_no_image_tokens(self):
        # 测试没有图像 token 的情况
        batch_size = 1
        sequence_length = 3
        num_image_patches = 2
        embed_dim = 3
        pad_token_id = 0
        image_token_index = 1

        input_ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
        inputs_embeds = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]], dtype=torch.float
        )
        image_features = torch.tensor(
            [[[10.1, 10.2, 10.3], [10.4, 10.5, 10.6]]], dtype=torch.float
        )

        final_embedding, position_ids = merge_input_ids_with_image_features(
            image_features=image_features,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            pad_token_id=pad_token_id,
            image_token_index=image_token_index,
        )

        # 期望输出与 inputs_embeds 相同
        expected_final_embedding = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]], dtype=torch.float
        )

        expected_position_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)

        self.assertEqual(final_embedding.shape, (batch_size, 3, embed_dim))
        self.assertEqual(position_ids.shape, (batch_size, 3))
        self.assertTrue(torch.allclose(final_embedding, expected_final_embedding))
        self.assertTrue(torch.all(position_ids == expected_position_ids))

    def test_merge_all_image_tokens(self):
        # 测试所有 token 都是图像 token
        batch_size = 1
        sequence_length = 3
        num_image_patches = 2
        embed_dim = 3
        pad_token_id = 0
        image_token_index = 1

        input_ids = torch.tensor([[1, 1, 1]], dtype=torch.long)
        inputs_embeds = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float
        )
        image_features = torch.tensor(
            [[[10.1, 10.2, 10.3], [10.4, 10.5, 10.6]]], dtype=torch.float
        )

        final_embedding, position_ids = merge_input_ids_with_image_features(
            image_features=image_features,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            pad_token_id=pad_token_id,
            image_token_index=image_token_index,
        )

        # 每个图像 token 替换为2个图像 patches
        # 新序列长度 = 3 + 3*(2-1) =6
        expected_final_embedding = torch.tensor(
            [
                [
                    [10.1, 10.2, 10.3],
                    [10.4, 10.5, 10.6],
                    [10.1, 10.2, 10.3],
                    [10.4, 10.5, 10.6],
                    [10.1, 10.2, 10.3],
                    [10.4, 10.5, 10.6],
                ]
            ],
            dtype=torch.float,
        )

        expected_position_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)

        self.assertEqual(final_embedding.shape, (batch_size, 6, embed_dim))
        self.assertEqual(position_ids.shape, (batch_size, 6))
        self.assertTrue(torch.allclose(final_embedding, expected_final_embedding))
        self.assertTrue(torch.all(position_ids == expected_position_ids))

    def test_merge_with_pad_tokens(self):
        # 测试含 pad token 的图像 token
        batch_size = 1
        sequence_length = 5
        num_image_patches = 2
        embed_dim = 3
        pad_token_id = 0
        image_token_index = 1

        input_ids = torch.tensor([[2, 1, 3, 1, 0]], dtype=torch.long)
        inputs_embeds = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                    [1.0, 1.1, 1.2],
                    [0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float,
        )
        image_features = torch.tensor(
            [[[10.1, 10.2, 10.3], [10.4, 10.5, 10.6]]], dtype=torch.float
        )

        final_embedding, position_ids = merge_input_ids_with_image_features(
            image_features=image_features,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            pad_token_id=pad_token_id,
            image_token_index=image_token_index,
        )

        # 期望:
        # Original: [2, 1, 3, 1, 0]
        # Replaced: [2, image_features, 3, image_features, 0]
        # new_sequence_length=5 +2*(2-1)=7
        expected_final_embedding = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3],
                    [10.1, 10.2, 10.3],
                    [10.4, 10.5, 10.6],
                    [0.7, 0.8, 0.9],
                    [10.1, 10.2, 10.3],
                    [10.4, 10.5, 10.6],
                    [0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float,
        )

        expected_position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], dtype=torch.long)

        self.assertEqual(final_embedding.shape, (batch_size, 7, embed_dim))
        self.assertEqual(position_ids.shape, (batch_size, 7))
        self.assertTrue(torch.allclose(final_embedding, expected_final_embedding))
        self.assertTrue(torch.all(position_ids == expected_position_ids))


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
