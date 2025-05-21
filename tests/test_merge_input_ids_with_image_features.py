import unittest
import torch
import torch.nn as nn


class Config:
    def __init__(self, image_token_index=32000, pad_token_id=0, ignore_index=-100):
        self.image_token_index = image_token_index
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index


class MockModel:
    def __init__(self, config):
        self.config = config


class MultiModalModel:
    def __init__(self, config):
        self.config = config
        self.model = MockModel(config)

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids, attention_mask
    ):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        # NOTE: 检查每个样本的最后一个 token 是否为填充 token
        # NOTE: 如果最后一个 token 不是填充 token，则为 True，表示存在左侧填充；否则为 False。
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.config.pad_token_id)
        )

        # 1. 创建图像 token 的掩码来获取特殊图像 token 的位置, 并计算新序列最大长度
        # NOTE: 一个布尔张量，标识 input_ids 中哪些位置是图像 token（即等于 image_token_index 的位置）
        special_image_token_mask = input_ids == self.config.image_token_index
        # NOTE: 每个样本中图像 token 的数量, 形状为 [batch_size, ]
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)

        # 计算合并图像特征后的新序列最大长度。
        # NOTE: 每个图像 token 位置会被替换为 (num_image_patches - 1) 个图像 paches embedding token。
        max_embed_dim = (
            num_special_image_tokens.max() * (num_image_patches - 1)
        ) + sequence_length
        # NOTE: 通过 torch.where 获取所有非图像 token 的位置索引。
        # NOTE: 当仅提供 condition 参数时，torch.where 等同于 torch.nonzero(condition, as_tuple=True)，返回满足条件的元素的索引。
        batch_indices, non_image_indices = torch.where(
            input_ids != self.config.image_token_index
        )  # 满足条件的样本索引和序列 token 索引

        # 2. 计算文本应写入的位置
        # NOTE: 每个图像 token 会增加 (num_image_patches - 1) 个位置。
        new_token_positions = (
            torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1)
            - 1
        )
        nb_image_pad = (
            max_embed_dim - 1 - new_token_positions[:, -1]
        )  # 计算需要的图像填充数量，以达到 max_embed_dim。
        # 如果存在左侧填充 (left_padding 为 True)，则将 new_token_positions 进行偏移调整。
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        # NOTE: 获取需要覆盖的文本 token 在合并序列中的新位置。
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. 初始化最终的嵌入与注意力掩码
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
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
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_image_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_image_indices
        ]

        # 5. 填充图像特征与更新注意力掩码和位置 ID.
        image_to_overwrite = torch.all(
            final_embedding == 0, dim=-1
        )  # 找出 final_embedding 中所有维度为0的位置（即尚未填充的地方）。
        # NOTE: 使用 cumsum 计算累积和，确保这些位置在填充数量 (nb_image_pad) 之后。
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[
            :, None
        ].to(target_device)

        if (
            image_to_overwrite.sum() != image_features.shape[:-1].numel()
        ):  # 如果需要填充的位置数量不等于 image_features 的数量，抛出错误。
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        # NOTE: 将 image_features 重新排列为 (batch_size * num_image_patches, embed_dim)，并填充到 final_embedding 的相应位置。
        final_embedding[image_to_overwrite] = (
            image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
            (final_attention_mask == 0), 1
        )

        # 6. 处理填充位置的嵌入, 将填充位置的嵌入设为0：
        batch_indices, pad_indices = torch.where(input_ids == self.config.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        return final_embedding, final_attention_mask, position_ids


class TestMergeInputIDsWithImageFeaturesDebug(unittest.TestCase):
    def setUp(self):
        # 初始化配置对象
        self.config = Config(image_token_index=32000, pad_token_id=0, ignore_index=-100)
        # 初始化模型
        self.model = MultiModalModel(self.config)
        self.device = torch.device("cpu")  # 使用CPU进行测试

    def test_merge_without_padding_debug(self):
        """
        测试在没有填充且每个样本有一个图像token的情况下的合并功能，并打印中间变量。
        """
        # Batch size 2, sequence length 7
        input_ids = torch.tensor(
            [[1, 2, 32000, 4, 5, 6, 7], [8, 32000, 10, 11, 12, 13, 14]],
            dtype=torch.long,
        )

        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.long
        )

        # Image features: 2 images, 3 patches each, embed_dim=256
        image_features = torch.randn(2, 3, 256)

        # Inputs embeds: batch_size=2, sequence_length=7, embed_dim=256
        inputs_embeds = torch.randn(2, 7, 256)

        # 调用方法
        final_embedding, final_attention_mask, position_ids = (
            self.model._merge_input_ids_with_image_features(
                image_features=image_features,
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )

        # 打印关键变量
        print("Test Merge Without Padding - Debug:")
        print(f"Input IDs:\n{input_ids}")
        print(f"Attention Mask:\n{attention_mask}")
        print(f"Image Features Shape: {image_features.shape}")
        print(f"Inputs Embeds Shape: {inputs_embeds.shape}")
        print(f"Final Embedding Shape: {final_embedding.shape}")
        print(f"Final Attention Mask:\n{final_attention_mask}")
        print(f"Position IDs:\n{position_ids}\n")

    def test_merge_with_padding_debug(self):
        """
        测试在有填充且样本中有一个图像token的情况下的合并功能，并打印中间变量。
        """
        # Batch size 1, sequence length 5 with padding
        input_ids = torch.tensor([[1, 32000, 3, 0, 0]], dtype=torch.long)

        attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

        # Image features: 1 image, 3 patches each, embed_dim=256
        image_features = torch.randn(1, 3, 256)

        # Inputs embeds: batch_size=1, sequence_length=5, embed_dim=256
        inputs_embeds = torch.randn(1, 5, 256)

        # 调用方法
        final_embedding, final_attention_mask, position_ids = (
            self.model._merge_input_ids_with_image_features(
                image_features=image_features,
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )

        # 打印关键变量
        print("Test Merge With Padding - Debug:")
        print(f"Input IDs:\n{input_ids}")
        print(f"Attention Mask:\n{attention_mask}")
        print(f"Image Features Shape: {image_features.shape}")
        print(f"Inputs Embeds Shape: {inputs_embeds.shape}")
        print(f"Final Embedding Shape: {final_embedding.shape}")
        print(f"Final Attention Mask:\n{final_attention_mask}")
        print(f"Position IDs:\n{position_ids}\n")

    def test_merge_no_image_tokens_debug(self):
        """
        测试在没有图像token的情况下的合并功能，并打印中间变量。
        """
        # Batch size 1, sequence length 4, no image tokens
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)

        # Image features: 0 images, 0 patches each, embed_dim=256
        image_features = torch.empty(0, 3, 256)

        # Inputs embeds: batch_size=1, sequence_length=4, embed_dim=256
        inputs_embeds = torch.randn(1, 4, 256)

        # 调用方法
        final_embedding, final_attention_mask, position_ids = (
            self.model._merge_input_ids_with_image_features(
                image_features=image_features,
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )

        # 打印关键变量
        print("Test Merge No Image Tokens - Debug:")
        print(f"Input IDs:\n{input_ids}")
        print(f"Attention Mask:\n{attention_mask}")
        print(f"Image Features Shape: {image_features.shape}")
        print(f"Inputs Embeds Shape: {inputs_embeds.shape}")
        print(f"Final Embedding Shape: {final_embedding.shape}")
        print(f"Final Attention Mask:\n{final_attention_mask}")
        print(f"Position IDs:\n{position_ids}\n")

    def test_merge_all_image_tokens_debug(self):
        """
        测试在所有 token 都是图像 token 的情况下的合并功能，并打印中间变量。
        """
        # Batch size 1, sequence length 3, all image tokens
        input_ids = torch.tensor([[32000, 32000, 32000]], dtype=torch.long)

        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

        # Image features: 3 images, 2 patches each, embed_dim=256
        image_features = torch.randn(3, 2, 256)

        # Inputs embeds: batch_size=1, sequence_length=3, embed_dim=256
        inputs_embeds = torch.randn(1, 3, 256)

        # 调用方法
        final_embedding, final_attention_mask, position_ids = (
            self.model._merge_input_ids_with_image_features(
                image_features=image_features,
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )

        # 打印关键变量
        print("Test Merge All Image Tokens - Debug:")
        print(f"Input IDs:\n{input_ids}")
        print(f"Attention Mask:\n{attention_mask}")
        print(f"Image Features Shape: {image_features.shape}")
        print(f"Inputs Embeds Shape: {inputs_embeds.shape}")
        print(f"Final Embedding Shape: {final_embedding.shape}")
        print(f"Final Attention Mask:\n{final_attention_mask}")
        print(f"Position IDs:\n{position_ids}\n")

    def test_merge_invalid_image_tokens_debug(self):
        """
        测试当图像token数量与提供的图像特征数量不匹配时，是否正确抛出错误，并打印相关信息。
        """
        # Batch size 1, sequence length 4, two image tokens but only one image feature
        input_ids = torch.tensor([[1, 32000, 32000, 4]], dtype=torch.long)

        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)

        # Image features: 1 image, 3 patches each, embed_dim=256
        image_features = torch.randn(1, 3, 256)

        # Inputs embeds: batch_size=1, sequence_length=4, embed_dim=256
        inputs_embeds = torch.randn(1, 4, 256)

        # 断言会抛出 ValueError，并打印相关信息
        print("Test Merge Invalid Image Tokens - Debug:")
        try:
            self.model._merge_input_ids_with_image_features(
                image_features=image_features,
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        except ValueError as e:
            print(f"Raised ValueError as expected: {e}\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
