from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from transformers import AutoTokenizer
from llama import ModelArgs, Llama  # 确保这些类已正确定义和导入


class LLaMAInfer:
    """
    LLaMAInfer类用于加载LLaMA模型并执行推理（文本生成）。
    """
    def __init__(self, model: Llama, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        """
        初始化LLaMAInfer实例。

        参数:
            model (Llama): 加载的LLaMA模型实例。
            tokenizer (SentencePieceProcessor): 加载的SentencePiece分词器实例。
            model_args (ModelArgs): 模型配置参数。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @classmethod
    def convert_hf_to_triton(cls, hf_sd, model_args):
        """
        将 Hugging Face 模型的权重字典转换为自定义模型的权重字典。

        参数:
            hf_sd (dict): Hugging Face 模型的状态字典。
            model_args (ModelArgs): 自定义模型的配置参数。

        返回:
            dict: 转换后的状态字典。
        """
        mapping = {
            "norm.weight": "norm_weight", 
            "output.weight": "lm_head_weight",
            "tok_embeddings.weight": "embed_tokens.weight",
        }

        layers = {
            "layers.{i}.attention.wq.weight": "layers.{i}.attention.q_proj_weight",
            "layers.{i}.attention.wk.weight": "layers.{i}.attention.k_proj_weight",
            "layers.{i}.attention.wv.weight": "layers.{i}.attention.v_proj_weight",
            "layers.{i}.attention.wo.weight": "layers.{i}.attention.o_proj_weight",
            "layers.{i}.feed_forward.w1.weight": "layers.{i}.mlp.gate_proj_weight",
            "layers.{i}.feed_forward.w3.weight": "layers.{i}.mlp.up_proj_weight",
            "layers.{i}.feed_forward.w2.weight": "layers.{i}.mlp.down_proj_weight",
            "layers.{i}.attention_norm.weight": "layers.{i}.input_layernorm_weight",
            "layers.{i}.ffn_norm.weight": "layers.{i}.post_attention_weight",
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

        return new_sd

    @classmethod
    def build(cls, checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
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

            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()

            print(checkpoint.keys())
        else:
            checkpoint = None

        # 读取模型参数
        params_path = Path(checkpoints_dir) / "params.json"
        assert params_path.exists(), f"params.json not found in {checkpoints_dir}"
        with open(params_path, "r") as f:
            params = json.load(f)

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 设置默认张量类型
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        if load_model:
            state_dict = cls.convert_hf_to_triton(checkpoint, model_args) # 转换权重名称
        else:
            state_dict = None

        # 初始化自定义的 Llama 模型
        model = Llama(model_args).to(device)

        print(model)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            if 'rope.freqs' in checkpoint:
                del checkpoint['rope.freqs']  # 删除检查点中未匹配的键（例如rope.freqs）
            # 使用转换后的 state_dict 加载模型
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        return cls(model, tokenizer, model_args)

    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        """
        根据给定的提示生成文本。

        参数:
            prompts (list[str]): 输入的文本提示列表。
            temperature (float): 温度参数，用于控制生成的随机性。
            top_p (float): Top-p采样的累积概率阈值。
            max_gen_len (Optional[int]): 最大生成长度。

        返回:
            tuple: 包含生成的token列表和生成的文本列表。
        """
        
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]

        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)
        
        eos_reached = torch.tensor([False] * batch_size, device=self.args.device)
        prompt_tokens_mask = tokens != pad_id  # True if the token is a prompt token, False otherwise
        cur_iterator = tqdm(range(max_prompt_len, total_len), desc="Generating tokens")
        
        for cur_pos in cur_iterator:
            with torch.no_grad():
                # 假设模型的 forward 方法接受输入 tokens 和当前的位置
                logits = self.model.forward(tokens[:, :cur_pos], cur_pos)

            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if eos_reached.all():
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)
    
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

    model = LLaMAInfer.build(
        checkpoints_dir='/gemini/code/Llama-3.2-1B-Instruct/original/',
        tokenizer_path='/gemini/code/Llama-3.2-1B-Instruct/',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    out_tokens, out_texts = model.text_completion(prompts, max_gen_len=64)
    assert len(out_texts) == len(prompts), "生成的文本数量与提示数量不一致"
    for i in range(len(out_texts)):
        print(f'生成文本 {i+1}: {out_texts[i]}')
        print('-' * 50)
