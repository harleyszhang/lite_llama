from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Literal, Optional, Tuple, TypedDict
from llama import LlamaConfig, Llama  # 确保这些类已正确定义和导入
import torch.nn.functional as F 

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


def convert_hf_to_triton(hf_sd, model_args):
    """
    将 Hugging Face 模型的权重字典转换为自定义模型的权重字典。

    参数:
        hf_sd (dict): Hugging Face 模型的状态字典。
        model_args (LlamaConfig): 自定义模型的配置参数。

    返回:
        dict: 转换后的状态字典。
    """
    mapping = {
        "tok_embeddings.weight": "embed_tokens.weight",
        "norm.weight": "norm_weight", 
        "output.weight": "lm_head.weight",
    }

    layers = {
        # key 是原始权重值, value 是自定义模型结构权重参数
        "layers.{i}.attention.wq.weight": "layers.{i}.attention.wq.weight",
        "layers.{i}.attention.wk.weight": "layers.{i}.attention.wk.weight",
        "layers.{i}.attention.wv.weight": "layers.{i}.attention.wv.weight",
        "layers.{i}.attention.wo.weight": "layers.{i}.attention.wo.weight",
        "layers.{i}.feed_forward.w1.weight": "layers.{i}.feed_forward.gate_proj.weight",
        "layers.{i}.feed_forward.w3.weight": "layers.{i}.feed_forward.up_proj.weight",
        "layers.{i}.feed_forward.w2.weight": "layers.{i}.feed_forward.down_proj.weight",
        "layers.{i}.attention_norm.weight": "layers.{i}.attention_norm.weight",
        "layers.{i}.ffn_norm.weight": "layers.{i}.ffn_norm.weight",
    }

    # 根据 Transformer 层数量生成映射
    for i in range(model_args.n_layers):
        for hf_key, custom_key in layers.items():
            mapped_key = hf_key.format(i=i) # hf 权重参数字典 key
            custom_mapped_key = custom_key.format(i=i) # 自定义模型权重参数字典 key
            mapping[mapped_key] = custom_mapped_key

    # 创建新的状态字典
    new_sd = {}
    for hf_key, tensor in tqdm(hf_sd.items(), desc="Mapping weights"):
        custom_key = mapping.get(hf_key, None)
        if custom_key is not None:
            new_sd[custom_key] = tensor
        else:
            # 如果某些权重不需要映射，可以选择忽略或处理
            pass  # 忽略未映射的权重
    
    torch.save(new_sd, "/gemini/code/Llama-3.2-1B-Instruct/my_weight/my_llama3.2-1B.pth")

    return new_sd

class GenerateText:
    """
    GenerateText 类用于加载LLaMA模型并执行迭代式生成式推理（文本生成）。
    """

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str, triton_weight=True):
        """
        构建LLaMAInfer实例, 加载模型和分词器。

        参数:
            checkpoints_dir (str): 模型检查点目录路径。
            tokenizer_path (str): 分词器模型文件路径。
            load_model (bool): 是否加载模型权重。
            max_seq_len (int): 最大序列长度。
            max_batch_size (int): 最大批处理大小。
            device (str): 设备类型（'cuda'或'cpu'）。

        返回:
            LLaMAInfer: 初始化后的LLaMAInfer实例。
        """
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')

            checkpoint = torch.load(ckpt_path, map_location="cuda")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        else:
            checkpoint = None

        # 读取模型参数
        params_path = Path(checkpoints_dir) / "config.json"
        assert params_path.exists(), f"params.json not found in {checkpoints_dir}"
        with open(params_path, "r") as f:
            params = json.load(f)

        model_args: LlamaConfig = LlamaConfig(
            params,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
        )

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 设置默认张量类型
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        if load_model:
            if triton_weight:
                state_dict = checkpoint
                print("Load Triton weight directly!")
            else:
                state_dict = convert_hf_to_triton(checkpoint, model_args) # 转换权重名称
        else:
            state_dict = None

        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # 初始化自定义的 Llama 模型
        model = Llama(model_args).to(device)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            if 'rope.freqs' in checkpoint:
                del checkpoint['rope.freqs']  # 删除检查点中未匹配的键（例如rope.freqs）
            # 使用转换后的 state_dict 加载模型
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        return GenerateText(model, tokenizer, model_args)

    def __init__(self, model: Llama, tokenizer: AutoTokenizer, model_args: LlamaConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        args = self.args
        bsz = len(prompt_tokens)
        assert bsz <= args.max_batch_size, (bsz, args.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= args.max_seq_len
        total_len = min(args.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")

        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)

            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_token_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_token_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_token_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
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
            max_gen_len = self.model.args.max_seq_len - 1

        # 使用 encode 方法获取整数 token ID 列表
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=True) for x in prompts]
    
        # 调试输出，确保 prompt_tokens 是整数列表
        for idx, tokens_list in enumerate(prompt_tokens):
            print(f"Prompt {idx} tokens: {tokens_list}")
            print(f"Type of tokens_list: {type(tokens_list)}")
            if tokens_list:
                print(f"Type of first token: {type(tokens_list[0])}")
            else:
                print("Tokens list is empty.")

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
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
            max_gen_len = self.model.args.max_seq_len - 1
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


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


if __name__ == '__main__':
    torch.manual_seed(0)

    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision: 
        """
    ]

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    model = LLaMAInfer.build(
        checkpoints_dir='/gemini/code/pytriton_llama/',
        tokenizer_path='/gemini/code/Llama-3.2-1B-Instruct/',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
        triton_weight=True
    )
    prompts = ["Consider you what services he has done", "who are you?"]
    output_tokens, output_texts = model.text_completion(prompts, max_gen_len=64)
    output_texts = output_texts[0].replace("<|begin_of_text|>", "")
    print(output_texts)