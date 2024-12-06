from typing import Optional
import torch
import time
from pathlib import Path
import json,logging

from typing import List, Literal, Optional, Tuple, TypedDict, Generator
import torch.nn.functional as F 
from torch.profiler import record_function
from transformers import AutoTokenizer

from .executor.model_executor import ModelExecutor
from .models.llama import LlamaModel  # 确保这些类已正确定义和导入
from .models.model_config import LlamaConfig
from .utils.file_interface import get_model_name_from_path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required

Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

class GenerateText:
    """
    GenerateText 类用于加载LLaMA模型并执行迭代式生成式推理 (文本生成)。
    """

    def __init__(self, 
        checkpoints_dir: str,
        tokenizer_path: str,
        max_gpu_num_blocks = None,
        max_seq_len = 1024,
        load_model = True,
        triton_weight = True,
        compiled_model = False,
        device="cuda",
    ):
        self.checkpoints_dir = checkpoints_dir
        self.compiled_model = compiled_model

        self.model_executor = ModelExecutor.build(
            checkpoints_dir = checkpoints_dir,
            load_model = load_model,
            max_gpu_num_blocks = max_gpu_num_blocks,
            max_seq_len = max_seq_len,
            triton_weight = triton_weight,
            device = device
        )
        self.model_config = self.model_executor.model_config
        self.tokenizer = self.load_tokenizer(tokenizer_path)
    
    def load_tokenizer(self, pretrained_model_name_or_path):
        model_name = get_model_name_from_path(pretrained_model_name_or_path)

        if 'llava' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=True)
        
        return tokenizer
    
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = True,
        echo: bool = False,
        device = "cuda"
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        基于提供的提示词 (prompts) 使用语言生成模型生成文本序列。
        
        参数：
            prompt_tokens (List[List[int]]): 提示词的 token 序列，每个提示词是一个整数列表, 即 input_ids。
            max_gen_len (int): 最大生成序列长度。
            temperature (float, 可选): 控制采样随机性的温度值，默认 0.6。
            top_p (float, 可选): nucleus 采样的概率阈值，默认 0.9。
            logprobs (bool, 可选): 是否计算每个生成 token 的 log 概率，默认 False。
            echo (bool, 可选): 是否在输出中包含提示词，默认 False。
        返回：
            Tuple[List[List[int]], Optional[List[List[float]]]]: 
                生成的 token 序列和（可选）对应的 log 概率。
        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        model_config = self.model_config
        bsz = len(prompt_tokens) # 批量大小

        # 计算 batch 提示词的最小和最大长度
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= model_config.max_seq_len
        total_len = min(model_config.max_seq_len, max_gen_len + max_prompt_len)

        # batch 输入使用填充技术. 定义填充 token id; 并初始化输入 batch tokens 张量，默认填充为 pad_id
        # NOTE: 示例的 tokens 尺寸是 torch.Size([4, 85], 也是模型的输入张量尺寸.
        # NOTE: pad_id 的作用是用于填充序列使得 batch 中的请求 seq 达到统一长度(max(seq_len))。
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device = device)
        # 将每个提示词 ids 拷贝到模型输入张量 tokens 中
        for seq_id, token_ids in enumerate(prompt_tokens):
            tokens[seq_id, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device = device)
        
        # 生成一个布尔张量，它的值为 True 的位置表示输入序列的实际内容（即非填充部分），False 表示待填充位置。形状为 (batch_size, total_len)
        input_text_mask = tokens != pad_id
        eos_reached = torch.tensor([False] * bsz, device = device) # 记录是否到达结束 token

        if logprobs:
            token_logprobs, _ = torch.zeros_like(tokens, dtype=torch.float)
        else:
            token_logprobs = None
            
        prev_pos = 0 # 初始化上一次生成的位置

        # 如果最短提示长度已达总长度，直接计算 logprobs
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        # 创建 CUDA 事件用于测量时间
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        token_count = 0 # 累计生成 token 数
        for cur_pos in range(min_prompt_len, total_len):
            input_ids = tokens[:, prev_pos: cur_pos] # 当前输入 token ids, decode 阶段 input_ids shape is [4, 1]
            logits, select_index = self.model_executor.forward(input_ids, prev_pos) # 模型执行器的前向推理
            
            # 根据 temperature 和 top_p 进行采样
            if temperature > 0:
                # NOTE: logits[:, -1] 表示选择的是最后一个位置（seq_len 维度的最后一项）对应的 logits，形状变为 [batch_size, vocab_size]
                # NOTE: 在生成模型中的 prefill 阶段，我们只关心当前生成的最后一个 token 的分布。
                # NOTE: temperature 作用是 调整 logits 的分布，用于控制采样的随机性。
                # NOTE: temperature < 1.0，分布会变得更加陡峭，更倾向于选择高概率的 token。temperature > 1.0，分布会变得更加平坦，增加随机性。
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1) # torch.softma 将 logits 转换为概率分布。
                # NOTE: 使用核采样方法，从高概率的候选 token 中选择下一个 token 索引. top_p 控制采样范围（候选 token 的概率累积值）。
                next_token = sample_top_p(probs, top_p) # next_token 形状为 [batch_size, 1]
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1) # 调整为一维, shape is batch_size
            
            # 如果当前位置属于输入文本部分，保留原始 token，否则替换为生成的 token
            # NOTE: input_text_mask[:, cur_pos]：获取掩码中当前列的布尔值，表示每个序列在当前位置是否为实际输入词元。
            # NOTE: tokens[:, cur_pos]：获取 tokens 中当前列的值。next_token：包含当前生成的词元 ID。
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token) # 可以保护提示词部分不被模型生成值覆盖，毕竟每个请求的长度不一
            tokens[:, cur_pos] = next_token
            
            # 如果需要 logprobs，计算当前生成 token 的 log 概率
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            # eos_reached 是一个布尔张量，记录每个序列是否到达了终止状态, 形状为 [batch_size, 1]。
            # NOTE: ～input_text_mask[:, cur_pos] 标记当前生成位置是否是模型生成的部分（非输入部分）。True 表示当前列是待生成的部分。False 表示当前列是输入部分。
            # NOTE: next_token == self.tokenizer.eos_token_id 表示检测当前生成的 next_token 是否等于 eos_token_id，即模型生成了终止标记。
            # NOTE: & 表示按位与操作，确保当前位置是非输入部分且生成了终止标记。
            # NOTE: 使用 |= 按位或更新，表示如果某个序列已经到达 eos_token_id，则保持 True 状态，不会被后续重置为 False。
            eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_token_id)
            prev_pos = cur_pos # 记录当前生成的位置

            # 累加生成的 tokens 数量（假设 batch_size 为 tokens 的第一个维度）
            batch_size = tokens.size(0)
            token_count += batch_size

            if all(eos_reached): # 如果所有样本均到达结束 token，停止生成
                break
        
        # 记录结束事件 
        end_event.record()        
        torch.cuda.synchronize() # 同步 CUDA 流以确保事件被记录
        
        # 计算总的运行时间和吞吐量
        elapsed_time_sec = start_event.elapsed_time(end_event) / 1000.0
        tokens_per_second = token_count / elapsed_time_sec if elapsed_time_sec > 0 else float('inf')
        logger.info(f"Decode stage Batch inference time: {elapsed_time_sec * 1000:.4f} ms")
        logger.info(f"Decode stage tokens per second : {tokens_per_second:.2f} tokens/s")
        
		# 处理生成的 tokens 和对应的对数概率，提取最终的输出序列。
        out_tokens, out_logprobs = self.process_output_tokens(tokens, prompt_tokens, max_gen_len, 
									logprobs, echo, self.tokenizer.eos_token_id, token_logprobs)
        # 减少 kv cache 内存管理器的引用计数
        self.model_executor.kv_mem_manager.release_ref(select_index)
        
        return out_tokens, out_logprobs
    
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
        device = "cuda",
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model_config.max_seq_len - 1

        # 使用 encode 方法获取整数 token ID 列表. prompt_tokens 是整数列表
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=True) for x in prompts]   

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens = prompt_tokens,
            max_gen_len = max_gen_len,
            temperature = temperature,
            top_p = top_p,
            logprobs = logprobs,
            echo = echo,
            device = device,
        )

        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model_config.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]

    def process_output_tokens(
        self,
        tokens: torch.Tensor,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        logprobs: bool,
        echo: bool,
        eos_token_id,
        token_logprobs: Optional[torch.Tensor] = None
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        处理生成的 tokens 和对应的对数概率，提取最终的输出序列。

        参数：	
            tokens (torch.Tensor): 生成的 tokens 张量，形状为 [batch_size, total_len]。
            prompt_tokens (List[List[int]]): 提示 tokens 的列表，每个元素是一个整数列表。
            max_gen_len (int): 最大生成长度。
            logprobs (bool): 是否计算并返回对数概率。
            echo (bool): 是否在输出中包含提示 tokens。
            eos_token_id: tokenizer.eos_token_id 对象。
            token_logprobs (Optional[torch.Tensor]): 生成的 tokens 的对数概率张量，形状与 tokens 相同。
        返回：
            Tuple[List[List[int]], Optional[List[List[float]]]]: 处理后的 tokens 列表和对数概率列表（如果需要）。
        """
        # eos_token_id = tokenizer.eos_token_id
        out_tokens, out_logprobs = [], [] # 初始化两个空列表，分别用于存储输出的 tokens（out_tokens）和对应的对数概率（out_logprobs）。

        if logprobs:
            # 将对数概率张量转换为列表
            token_logprobs = token_logprobs.cpu().tolist()

        for i, out_tokens in enumerate(tokens.tolist()): # 将 tokens 转换为列表
            prompt_len = len(prompt_tokens[i])
            # 根据是否需要在输出中包含提示词，确定起始位置
            start_idx = 0 if echo else prompt_len
            end_idx = prompt_len + max_gen_len

            # 截取从起始位置到最大生成长度的 tokens
            generated_toks = out_tokens[start_idx:end_idx]
            probs = None
            if logprobs: # 如果需要，对应地截取对数概率
                probs = token_logprobs[i][start_idx:end_idx]
            
            # 检查是否存在结束符，若存在则截断到结束符之前
            if eos_token_id in generated_toks:
                eos_idx = generated_toks.index(eos_token_id)
                generated_toks = generated_toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None

            out_tokens.append(generated_toks)
            out_logprobs.append(probs)

        return (out_tokens, out_logprobs if logprobs else None)
    
def sample_top_p(probs, p):
    """
    执行 Top-p (Nucleus) 采样, 从概率分布中采样下一个词。
    参数：
        probs (torch.Tensor): 概率分布张量，形状为 `[batch_size, vocab_size]`。
        p (float): 累积概率阈值，取值范围在 0 到 1 之间。
    返回：
        torch.Tensor: 采样得到的词索引，形状为 `[batch_size, 1]`。

    说明：
        Top-p 采样算法: 选择概率累积和超过阈值 p 的最小集合，将这些词的概率重新归一化后进行采样。
    """
    # 对概率分布进行降序排序。probs_sort: 排序后的概率值，形状与 probs 相同。probs_idx: 排序后的索引，用于映射回原始词汇表。
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # 计算排序后概率的累积和. 返回的 probs_sum 是累积概率分布。
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 保留累积概率未超过阈值 p 的词汇的概率，其余词汇的概率被置为 0.0。
    mask = probs_sum - probs_sort > p # 创建掩码，对于每个位置，计算累积概率（不包括当前词）是否超过阈值 p。
    probs_sort[mask] = 0.0 # 将累积概率超过阈值 p 的词的概率置零。

    # 对剩余的概率重新归一化, 确保总和为 1。
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # 从重新归一化的概率分布中采样下一个词. 返回的 next_token 是采样得到的词在排序后概率分布中的索引。
    next_token_sorted_idx = torch.multinomial(probs_sort, num_samples=1)
    # 在 probs_idx 的最后一维（dim=-1）中，使用 next_token_sorted_idx 作为索引，提取对应的值。沿着 dim=1（列）进行索引提取
    # NOTE: torch.gather 函数按照给定的索引张量 index，从输入张量中收集 (获取) 数据，并返回一个与索引张量形状一致的张量。
    next_token = torch.gather(probs_idx, -1, index = next_token_sorted_idx)
    
    return next_token # 返回采样得到的下一个词的索引