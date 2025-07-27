from typing import Optional
import torch, logging
from typing import Optional, TypedDict, Generator
from .executor.model_executor import ModelExecutor
from .utils.file_interface import get_model_name_from_path
from .kernels.softmax_split import softmax_split

from transformers import AutoTokenizer

# 新增导入
from .quantization.quant_manager import QuantizationType

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: list[str]  # not required
    logprobs: list[float]  # not required


# 保持采样函数不变
@torch.inference_mode()
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


class GenerateStreamText:
    """
    支持量化的GenerateStreamText类
    """

    def __init__(
        self,
        checkpoints_dir: str,
        tokenizer_path: str,
        max_gpu_num_blocks=None,
        max_seq_len=1024,
        compiled_model=False,
        device="cuda",
        quantization: Optional[str] = None,  # 新增参数
    ):
        self.checkpoints_dir = checkpoints_dir
        self.quantization = quantization  # 存储量化类型

        # 创建ModelExecutor时传入量化参数
        self.model_executor = ModelExecutor.build(
            checkpoints_dir=checkpoints_dir,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            compiled_model=compiled_model,
            device=device,
            quantization=quantization,  # 新增参数
        )
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.model_config = self.model_executor.model_config
        self.device = device

        # 记录量化信息
        if self.quantization and self.quantization != QuantizationType.NONE:
            logger.info(f"使用量化推理: {self.quantization}")

    def load_tokenizer(self, pretrained_model_name_or_path):
        model_name = get_model_name_from_path(pretrained_model_name_or_path)

        if "llava" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, use_fast=False, trust_remote_code=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, use_fast=True, trust_remote_code=True
            )

        return tokenizer

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt_tokens: list[list[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
        device="cuda",
    ) -> Generator[tuple[list[str], Optional[list[float]]], None, None]:
        """
        基于提供的 prompt_tokens, 使用语言生成模型逐个生成 token, 并在生成时立即输出。
        支持量化模型推理。
        """
        bsz = len(prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.model_config.max_seq_len
        total_len = min(self.model_config.max_seq_len, max_gen_len + max_prompt_len)
        actual_prompt_lens = torch.tensor(
            [len(t) for t in prompt_tokens], dtype=torch.long, device=device
        )
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        # 预分配tokens张量
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        input_text_mask = tokens != pad_id
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        prev_pos = 0
        last_yielded_pos = [
            len(prompt_tokens[i]) if not echo else 0 for i in range(bsz)
        ]

        # 填充提示词到 tokens 张量
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        b_req_idx = torch.arange(bsz, device=self.device)
        all_select_index_list = []
        prefill_select_index, _ = self.model_executor.prefill_alloc_kv_cache(
            max_prompt_len, actual_prompt_lens, b_req_idx
        )
        all_select_index_list.append(prefill_select_index)

        input_ids = tokens[:, :max_prompt_len]  # [batch_size, seq_len]
        for cur_pos in range(max_prompt_len, total_len):
            input_ids = tokens[:, prev_pos:cur_pos]
            batch_size, seq_len = input_ids.shape
            position_ids = (
                torch.arange(prev_pos, prev_pos + seq_len, device=input_ids.device)
                .unsqueeze(0)  # shape: [1, seq_len]
                .repeat(batch_size, 1)  # shape: [batch_size, seq_len], 不分配额外内存
            )

            # 使用量化模型进行前向推理
            logits = self.model_executor.forward(input_ids, position_ids)
            decode_select_index = self.model_executor.decode_alloc_kv_cache(bsz)
            all_select_index_list.append(decode_select_index)

            if temperature > 0:
                probs = softmax_split(logits[:, -1] / temperature)
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
            prev_pos = cur_pos

            # 为整个批次收集输出
            batch_outputs = []
            for i in range(bsz):
                start = last_yielded_pos[i]
                end = cur_pos + 1
                if start < end:
                    token = tokens[i, start:end].tolist()
                    text = self.tokenizer.decode(
                        token, skip_special_tokens=True
                    )
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
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        echo: bool = False,
    ) -> Generator[list[CompletionPrediction], None, None]:
        if max_gen_len is None:
            max_gen_len = self.model_config.max_seq_len - 1

        prompt_tokens = [
            self.tokenizer.encode(x, add_special_tokens=True) for x in prompts
        ]

        stream = self.generate_stream(
            prompt_tokens=prompt_tokens,
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