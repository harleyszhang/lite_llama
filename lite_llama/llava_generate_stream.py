from typing import Optional
import torch, logging, re
from PIL import Image

from typing import Optional, TypedDict, Generator, Union
from .executor.model_executor import ModelExecutor
from .utils.constants import *
from .utils.file_interface import get_model_name_from_path

from transformers import AutoTokenizer, AutoProcessor

# 新增导入
from .quantization.quant_manager import QuantizationType

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: list[str]  # not required
    logprobs: list[float]  # not required


def tokenizer_image_token(
        prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    """
    处理包含特殊标记 <image> 的文本提示, 将其转换为相应的 token 序列，并在 <image> 位置插入指定的图像 token 索引。
    """
    prompt_chunks = re.split(r"\s?<image>", prompt)
    token_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]

    input_ids = []
    offset = 0
    if (
            len(token_chunks) > 0
            and len(token_chunks[0]) > 0
            and token_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(token_chunks[0][0])

    for i, chunk in enumerate(token_chunks):
        input_ids.extend(chunk[offset:])
        offset = 0
        if i < len(token_chunks) - 1:
            input_ids.append(image_token_index)

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")

    return input_ids


class LlavaGeneratorStream:
    """
    支持量化的LlavaGeneratorStream类
    """

    def __init__(
            self,
            checkpoints_dir: str,
            tokenizer_path: str,
            max_gpu_num_blocks=None,
            max_seq_len=2048,
            compiled_model=False,
            device="cuda",
            quantization: Optional[str] = None,  # 新增参数
    ):
        self.checkpoints_dir = checkpoints_dir
        self.compiled_model = compiled_model
        self.max_seq_len = max_seq_len
        self.device = device
        self.quantization = quantization  # 存储量化类型

        # 创建ModelExecutor时传入量化参数
        self.model_executor = ModelExecutor.build(
            checkpoints_dir=checkpoints_dir,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            device=device,
            quantization=quantization,  # 新增参数
        )
        self.tokenizer = self.load_tokenizer(tokenizer_path)

        # 记录量化信息
        if self.quantization and self.quantization != QuantizationType.NONE:
            logger.info(f"使用量化推理 (LLaVA): {self.quantization}")

    def load_tokenizer(self, pretrained_model_name_or_path):
        model_name = get_model_name_from_path(pretrained_model_name_or_path)

        if "llava" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, use_fast=False
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, use_fast=True
            )

        return tokenizer

    def encode_images(self, image_items: list[Union[str, Image.Image]]):
        processor = AutoProcessor.from_pretrained(self.checkpoints_dir)
        self.image_processor = processor.image_processor
        images = []
        for item in image_items:
            if isinstance(item, Image.Image):
                image = item
            elif item.startswith("http://") or item.startswith("https://"):
                import requests

                image = Image.open(requests.get(item, stream=True).raw)
            else:
                image = Image.open(item)
            images.append(image.convert("RGB"))

        image_tensors = self.image_processor.preprocess(images, return_tensors="pt")[
            "pixel_values"
        ]
        if type(image_tensors) is list:
            image_tensors = [
                image.to(self.device, dtype=torch.float16) for image in image_tensors
            ]
        else:
            image_tensors = image_tensors.to(self.device, dtype=torch.float16)

        return image_tensors

    @torch.inference_mode()
    def generate_stream(
            self,
            prompt_tokens: list[list[int]],
            image_tensors: Optional[torch.FloatTensor] = None,
            max_gen_len: int = 2048,
            temperature: float = 0.6,
            top_p: float = 0.9,
            echo: bool = False,
    ) -> Generator[tuple[list[str], Optional[list[float]]], None, None]:
        """
        基于提供的 prompt_tokens, 使用量化的LLaVA模型逐个生成 token, 并在生成时立即输出。
        """
        bsz = len(prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.max_seq_len
        total_seq_len = min(self.max_seq_len, max_gen_len + max_prompt_len)
        actual_prompt_lens = torch.tensor(
            [len(t) for t in prompt_tokens], dtype=torch.long, device=self.device
        )
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        # 预分配 tokens 张量
        tokens = torch.full(
            (bsz, total_seq_len), pad_id, dtype=torch.long, device=self.device
        )
        input_text_mask = tokens != pad_id
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        last_yielded_pos = [
            len(prompt_tokens[i]) if not echo else 0 for i in range(bsz)
        ]

        # 填充提示词到 tokens 张量
        for seq_id, token_ids in enumerate(prompt_tokens):
            tokens[seq_id, : len(token_ids)] = (
                token_ids.clone().detach().to(dtype=torch.long, device=self.device)
            )

        # 计算输入图像待分配空间
        img_batch_size, _, _, _ = image_tensors.shape
        b_req_idx = torch.arange(bsz, device=self.device)
        all_select_index_list = []
        prefill_select_index, _ = self.model_executor.prefill_alloc_kv_cache(
            max_prompt_len, actual_prompt_lens, b_req_idx, img_batch_size
        )
        all_select_index_list.append(prefill_select_index)

        position_ids = None
        start_pos = len(prefill_select_index)
        input_ids = tokens[:, :max_prompt_len]  # [batch_size, seq_len]
        for cur_pos in range(max_prompt_len, total_seq_len):
            batch_size, _ = input_ids.shape

            # 使用量化模型进行前向推理
            logits = self.model_executor.forward(
                input_ids, position_ids, image_tensors
            )

            start_pos += bsz
            position_ids = (
                torch.arange(start_pos, start_pos + 1, device=input_ids.device)
                .unsqueeze(0)  # shape: [1, seq_len]
                .repeat(batch_size, 1)  # shape: [batch_size, seq_len], 不分配额外内存
            )

            decode_select_index = self.model_executor.decode_alloc_kv_cache(bsz)
            all_select_index_list.append(decode_select_index)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            input_ids = next_token  # [batch_size, 1]
            mask = ~input_text_mask[:, cur_pos]  # [batch_size]
            tokens[:, cur_pos] = torch.where(
                mask, next_token.reshape(-1), tokens[:, cur_pos]
            )

            eos_reached = eos_reached | (
                    mask & (next_token == self.tokenizer.eos_token_id)
            )

            # 为整个批次收集输出
            batch_outputs = []
            for i in range(bsz):
                start = last_yielded_pos[i]
                end = cur_pos + 1
                if start < end:
                    token = tokens[i, start:end].tolist()
                    text = self.tokenizer.decode(token, skip_special_tokens=True)
                    batch_outputs.append(text)
                    last_yielded_pos[i] = end
                else:
                    batch_outputs.append("")

            # 将整个批次的输出一次性 yield
            yield batch_outputs

            if eos_reached.all():
                break

        # 减少 kv cache 内存管理器的引用计数
        all_select_indexs = torch.concat(all_select_index_list)
        self.model_executor.kv_mem_manager.release_ref(all_select_indexs)

    def text_completion_stream(
            self,
            prompts: list[str],
            image_items: list[Union[str, Image.Image]],
            temperature: float = 0.6,
            top_p: float = 0.9,
            max_gen_len: Optional[int] = None,
            echo: bool = False,
    ) -> Generator[list[CompletionPrediction], None, None]:
        """每次迭代时，生成器返回一个包含多个 CompletionPrediction 字典的列表。"""

        if max_gen_len is None:
            max_gen_len = self.max_seq_len - 1

        prompt_tokens = [
            tokenizer_image_token(
                x, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            for x in prompts
        ]
        image_tensors = self.encode_images(image_items)

        stream = self.generate_stream(
            prompt_tokens=prompt_tokens,
            image_tensors=image_tensors,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
        )

        # 初始化每个样本的生成结果
        completions = [{"generation": "", "tokens": []} for _ in prompts]
        for batch_outputs in stream:
            for i, text in enumerate(batch_outputs):
                completions[i]["generation"] += text
            yield completions.copy()


def sample_top_p(probs, p):
    """
    执行 Top-p (Nucleus) 采样, 从概率分布中采样下一个词。
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = (probs_sum - probs_sort > p)
    probs_sort[mask] = 0.0

    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token_sorted_idx = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, index=next_token_sorted_idx)

    return next_token