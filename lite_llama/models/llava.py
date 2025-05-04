from typing import Optional, Dict

import torch,os
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from transformers import AutoModel, LlavaConfig
from .llama import LlamaModel
from .model_config import LlamaConfig
from ..kernels import gelu
from .utils import merge_input_ids_with_image_features, merge_multimodal_embeddings


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, vision_hidden_size: int, text_hidden_size: int,
                 projector_hidden_act: str="gelu"):
        super().__init__()

        self.linear_1 = nn.Linear(vision_hidden_size,
                                  text_hidden_size,
                                  bias=True)
        self.linear_2 = nn.Linear(text_hidden_size,
                                  text_hidden_size,
                                  bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = gelu(hidden_states) # GELU 激活函数
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaLlama(nn.Module):
    def __init__(self, llava_config: LlavaConfig):
        super().__init__()
        self.device = "cuda"
        self.llava_config = llava_config
        text_config = self.llava_config.text_config # TODO: 将 text_config 转换成 LlamaConfig 类型
        self.llama_config = LlamaConfig.from_dict(text_config.to_dict())
        
        self.select_layer = llava_config.vision_feature_layer
        self.select_feature = llava_config.vision_feature_select_strategy

        # 视觉处理模块（vision_tower）初始化
        self.vision_tower = AutoModel.from_config(llava_config.vision_config)

        # 多模态投影器（multi_modal_projector）初始化
        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size = llava_config.vision_config.hidden_size,
            text_hidden_size = llava_config.text_config.hidden_size,
            projector_hidden_act = llava_config.projector_hidden_act)
        
        # 语言模型初始化
        self.language_model = LlamaModel(self.llama_config)
        
        self.pad_token_id = self.llava_config.pad_token_id if self.llava_config.pad_token_id is not None else -1
    
    def _select_image_features(
        self, 
        image_features: torch.Tensor,
        strategy: str
    ) -> torch.Tensor:
        """根据策略选择图像特征"""
        # Copied from https://github.com/huggingface/transformers/blob/39c3c0a72af6fbda5614dde02ff236069bb79827/src/transformers/models/llava/modeling_llava.py#L421  # noqa
        if strategy == "default" or strategy == "patch":
            return image_features[:, 1:].contiguous()
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")
    
    def vision_encode(self, image_tensor):
        x = image_tensor.half().to(device=self.device)
        
        # 1. 通过视觉处理模块提取图像特征
        x = self.vision_tower(x, output_hidden_states = True)
        x = x.hidden_states[self.select_layer]
        x = self._select_image_features(x, self.select_feature)
        
        # 2. 通过多模态投影器将图像特征转换为多模态嵌入
        image_features = self.multi_modal_projector(x)

        assert not torch.isnan(image_features).any(), f"After vision_tower image_features tensor contains NaN values!"
        return image_features
    
    def get_multi_modal_input_embeddings(
        self,
        input_ids: torch.Tensor,
        vision_embeddings = None,
    ) -> torch.Tensor:
        """获取输入嵌入，包括文本和视觉嵌入的合并。"""
        llm_inputs_embeds = self.language_model.get_input_embeddings(input_ids) # torch.Size([1, 22]) --> torch.Size([1, 22, 4096])
        
        # torch.Size([1, 576, 4096]) torch.Size([1, 22, 4096]) torch.Size([1, 22])
        # print("self.llava_config.image_token_index is ", self.llava_config.image_token_index)
        if vision_embeddings is not None:
            inputs_embeds, position_ids = merge_input_ids_with_image_features(
                input_ids, llm_inputs_embeds, vision_embeddings, 
                self.llava_config.pad_token_id,
                self.llava_config.image_token_index,
            )
        
        assert not torch.isnan(inputs_embeds).any(), f"After merge inputs_embeds tensor contains NaN values!"

        return inputs_embeds, position_ids
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        position_ids: torch.Tensor,  
        atten_info, 
        image_tensor: Optional[torch.FloatTensor] = None,
    ):
        input_ids = input_ids.to(self.device) # 将 input_ids 移动到设备
        if position_ids is not None: # 如果提供了 position_ids，将其移动到设备
            position_ids = position_ids.to(self.device)
            
        if input_ids.shape[1] != 1: # 判断是不是首次 token 输出
            vision_embeddings = self.vision_encode(image_tensor) #  torch.Size([1, 3, 336, 336]) --> torch.Size([1, 576, 4096])
            inputs_embeds, position_ids = self.get_multi_modal_input_embeddings(input_ids, vision_embeddings)
        else: # 进入 decode 阶段, 无需再做视觉编码
            inputs_embeds = None
       
        hidden_states = self.language_model(input_ids = input_ids,
                                            position_ids = position_ids,
                                            atten_info = atten_info,
                                            inputs_embeds = inputs_embeds
                                            )
        
        return hidden_states
