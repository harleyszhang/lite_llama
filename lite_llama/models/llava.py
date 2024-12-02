#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
from PIL import Image

import torch,os
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from transformers import AutoConfig,AutoModel, AutoModelForCausalLM, LlavaConfig, \
                         LlamaModel, LlamaForCausalLM

from .llama import Llama
from .model_config import LlamaConfig

from .utils import merge_input_ids_with_image_features
from ..utils.llava_image_procss import process_images
from ..utils.config_convert import convert_transformers_to_custom_config

from ..kernels import ACT2FN

class LlavaMultiModalProjector(nn.Module):
    def __init__(self, vision_hidden_size: int, text_hidden_size: int,
                 projector_hidden_act: str):
        super().__init__()

        self.linear_1 = nn.Linear(vision_hidden_size,
                                  text_hidden_size,
                                  bias=True)
        self.act = ACT2FN[projector_hidden_act]
        self.linear_2 = nn.Linear(text_hidden_size,
                                  text_hidden_size,
                                  bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaLlama(nn.Module):
    def __init__(self, llava_config: LlavaConfig):
        super().__init__()
        self.device = "cuda"
        self.llava_config = llava_config
        text_config = self.llava_config.text_config # TODO: 将 text_config 转换成 LlamaConfig 类型
        # self.llama_config = convert_transformers_to_custom_config(text_config)
        self.llama_config = LlamaConfig.from_dict(text_config.to_dict())
        
        self.select_layer = llava_config.vision_feature_layer
        self.select_feature = llava_config.vision_feature_select_strategy

        # 视觉处理模块（vision_tower）初始化
        self.vision_tower = AutoModel.from_config(llava_config.vision_config)
        print("self.vision_tower ", self.vision_tower)
        # 多模态投影器（multi_modal_projector）初始化
        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size = llava_config.vision_config.hidden_size,
            text_hidden_size = llava_config.text_config.hidden_size,
            projector_hidden_act = llava_config.projector_hidden_act)
        
        # 语言模型初始化
        self.language_model = Llama(self.llama_config)
        
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

        return image_features
    
    def get_multi_modal_input_embeddings(
        self,
        input_ids: torch.Tensor,
        vision_embeddings = None,
    ) -> torch.Tensor:
        """获取输入嵌入，包括文本和视觉嵌入的合并。"""
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        
        if vision_embeddings is not None:
            inputs_embeds, position_ids = merge_input_ids_with_image_features(
                inputs_embeds, input_ids,
                self.llava_config.pad_token_id, 
                self.llava_config.image_token_index)
        
        return inputs_embeds
    
    def forward(
        self, 
        input_ids, start_pos, atten_info, 
        image_tensor: Optional[torch.FloatTensor] = None,
        position_ids: torch.Tensor = None,
    ):
        input_ids = input_ids.to(self.device) # 将 input_ids 移动到设备
        if position_ids is not None: # 如果提供了 position_ids，将其移动到设备
            position_ids = position_ids.to(self.device)
            
        if input_ids.shape[1] != 1:
            vision_embeddings = self.vision_encode(image_tensor)
            inputs_embeds = self.get_multi_modal_input_embeddings(input_ids, vision_embeddings)
        else: # 进入 decode 阶段, 无需再做视觉编码
            inputs_embeds = None

        hidden_states = self.language_model(input_ids = input_ids,
                                            start_pos = start_pos,
                                            atten_info = atten_info,
                                            position_ids = position_ids,
                                            inputs_embeds = inputs_embeds
                                            )
        
        return hidden_states

    def load_hf_weight(self, config, weight_dir):
        from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

        config = AutoConfig.from_pretrained(weight_dir, trust_remote_code = True)

        self.select_layer = config.vision_feature_layer
        self.select_feature = config.vision_feature_select_strategy

        processor = AutoProcessor.from_pretrained(weight_dir)
        self.images_processor = processor.image_processor

        llava_model = LlavaForConditionalGeneration.from_pretrained(weight_dir, torch_dtype=torch.float16,)
        self.vision_tower = llava_model.vision_tower # 复用 LlavaForConditionalGeneration 的 vision_tower
        self.multi_modal_projector = None
        self.language_model = None
        
        # 视觉特征映射层权重
        self.projector_weights = self._load_projector_weights(weight_dir)

    def _load_projector_weights(self, weight_dir):
        """load projector weights"""
        self.projector_weights = {}
        for f in os.listdir(weight_dir):
            if f.endswith(".safetensors"):
                d = safe_open(os.path.join(weight_dir, f), 'pt', 'cpu')
                for k in d.keys():
                    if "multi_modal_projector.linear_1" in k:
                        self.projector_weights[k.replace("multi_modal_projector.linear_1", "model.mm_projector.0")] = d.get_tensor(k).half()
                    if "multi_modal_projector.linear_2" in k:
                        self.projector_weights[k.replace("multi_modal_projector.linear_2", "model.mm_projector.2")] = d.get_tensor(k).half()
        
        return self.projector_weights
    
    def _load_language_model_weights(self, weight_dir):
        pass

    def llava_multi_modal_projector(self, x, projector_weights: Dict = None):
        batch_size, seq_len, hidden_size = x.shape
        x = F.linear(
            x,
            weight=self.projector_weights["model.mm_projector.0.weight"],
            bias=self.projector_weights["model.mm_projector.0.bias"],
        )
        x = F.gelu(x)
        x = F.linear(
            x,
            weight=self.projector_weights["model.mm_projector.2.weight"],
            bias=self.projector_weights["model.mm_projector.2.bias"],
        )

        return x

if __name__ == "__main__":
    model_path = "/gemini/code/liuhaotian/llava-v1.5-7b"
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import LlavaConfig

    # 使用 init_empty_weights 初始化空模型
    with init_empty_weights():
        config = LlavaConfig.from_pretrained(model_path)
        model = LlavaLlama(config)
        
        # 打印模型结构
        print(model)
        # 打印模型的简单摘要
        print(f"模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

        # 可选择打印部分参数信息
        for name, param in list(model.named_parameters())[:]:  # 打印模型参数
            print(name, param.shape)
