from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
import torch
from tqdm.auto import tqdm
import json, sys, os
from pathlib import Path

# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lite_llama.models.qwen2 import Qwen2Model, Qwen2Config
from lite_llama.executor.model_executor import ModelExecutor
from lite_llama.executor.weight_convert import convert_qwen2_hf_to_litellama


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


def load_config_from_json(json_file_path: str, device: str = "cuda") -> Qwen2Config:
    with open(json_file_path, "r") as f:
        config_dict = json.load(f)

    # 假设 Qwen2Config 可以通过关键字参数初始化
    config = Qwen2Config(
        hidden_size=config_dict.get("hidden_size", 128),
        num_heads=config_dict.get("num_heads", 8),
        num_kv_heads=config_dict.get("num_kv_heads", 8),
        intermediate_size=config_dict.get("intermediate_size", 512),
        num_layers=config_dict.get("num_layers", 2),
        vocab_size=config_dict.get("vocab_size", 1000),
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
    )
    return config


def load_original_model(model_name_or_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = Qwen2ForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.to(device)
    hf_sd = model.state_dict()

    return model, tokenizer, hf_sd


def load_custom_model(model_dir: str, model_config: Qwen2Config, device: str = "cuda"):
    # 找到 checkpoint 文件
    checkpoints = sorted(Path(model_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {model_dir}"
    ckpt_path = checkpoints[0]
    state_dict = torch.load(ckpt_path, map_location=device)

    # 初始化自定义模型
    model = Qwen2Model(model_config).to(device)

    # Convert model to float16 if on CUDA
    if device == "cuda":
        model = model.half()

    # Load state_dict
    model.load_state_dict(state_dict, strict=True)

    return model


class Qwen2ModelInferTest:
    def __init__(
        self,
        checkpoints_dir: str,
        tokenizer_path: str,
        max_batch_size=32,
        max_seq_len=2048,
        load_model: bool = True,
        triton_weight: bool = True,
        device: str = "cuda",
    ):
        self.model_executor = ModelExecutor.build(
            checkpoints_dir=checkpoints_dir,
            tokenizer_path=tokenizer_path,  # Fixed
            load_model=load_model,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            triton_weight=triton_weight,
            device=device,
        )

    def prefill_stage_compare(
        self,
        original_model,
        model_executor,
        tokenizer,
        input_text: str,
        device: str = "cuda",
    ):
        """Prefill stage comparison, including hidden states."""
        print(
            "\n############################ [Starting Prefill stage comparison] #################################"
        )
        # Prepare input
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        # Original model output
        with torch.no_grad():
            original_outputs = original_model(
                **inputs, output_hidden_states=True, return_dict=True
            )
        original_logits = original_outputs.logits  # [B, S, V]

        # Custom model output
        tokens = inputs["input_ids"]  # [B, S]
        with torch.no_grad():
            custom_outputs, _ = model_executor.forward(tokens, prev_pos=0)
        custom_logits = custom_outputs  # [B, S, V]

        # Compare logits
        if original_logits.shape != custom_logits.shape:
            print(
                f"Logits shape mismatch: original {original_logits.shape}, custom {custom_logits.shape}"
            )
            return None

        # Compare hidden states
        original_hidden_states = original_outputs.hidden_states  # Tuple of [B, S, D]
        custom_hidden_states = (
            model_executor.model.hidden_states
        )  # Assuming list of [B, S, D]

        if len(custom_hidden_states) != len(original_hidden_states):
            print(
                f"Number of hidden states layers mismatch: custom {len(custom_hidden_states)}, original {len(original_hidden_states)}"
            )
            return None

        print(
            f"model_executor.model.hidden_states number: {len(custom_hidden_states)}, original_outputs.hidden_states number: {len(original_hidden_states)} "
        )

        for index in tqdm(range(len(custom_hidden_states)), desc="Comparing layers"):
            custom_layer_output = custom_hidden_states[index]
            original_layer_output = original_hidden_states[index]

            if custom_layer_output.shape != original_layer_output.shape:
                print(
                    f"Layer {index} shape mismatch: custom {custom_layer_output.shape}, original {original_layer_output.shape}"
                )
                continue

            difference = (
                torch.abs(custom_layer_output - original_layer_output).mean().item()
            )
            print(f"Difference at layer {index}: {difference}")

        # Compare logits
        logits_diff = torch.abs(original_logits - custom_logits).mean().item()
        print(f"Prefill stage model Logits difference: {logits_diff}")

        # Sampling next token
        original_next_token_logits = original_logits[
            :, -1, :
        ]  # [B, V] Get logits for last token
        probs = torch.softmax(original_next_token_logits, dim=-1)  # [B, V]
        # Sample next token
        next_token_id = torch.argmax(probs, dim=-1)  # [B]

        # Decode
        transformers_generated_text = tokenizer.decode(next_token_id[0])
        print("Generated token:", transformers_generated_text)

        return transformers_generated_text

    def decode_stage_compare(
        self,
        original_model,
        model_executor,
        tokenizer,
        input_text: str,
        device: str = "cuda",
    ):
        """
        Decode stage comparison: step-by-step comparison of outputs.
        """
        # Prepare input
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]  # [B, S]
        attention_mask = inputs.get("attention_mask", None)

        # Set generation parameters
        max_new_tokens = 10
        original_model.eval()
        model_executor.model.eval()

        # 初始化生成的 tokens
        original_generated = input_ids
        custom_next_token = input_ids
        original_next_token = input_ids

        # 初始化 past_key_values 为 None
        past_key_values = None
        for step in tqdm(range(max_new_tokens), desc="Decoding steps"):
            # 1. Original model generates next token
            with torch.no_grad():
                original_outputs = original_model(
                    original_next_token,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                )
                original_logits = original_outputs.logits[:, -1, :]  # [B, V]

                temperature = 0.6  # Apply temperature # custom_outputs_logits: [B, V]
                probs = torch.softmax(original_logits / temperature, dim=-1)  # [B, V]
                original_next_token = sample_top_p(
                    probs, p=0.9
                )  # Sample next token [B, 1]
                # original_next_token = torch.argmax(original_logits, dim=-1, keepdim=True)  # [B, 1]

            # 2. Custom model generates next token
            with torch.no_grad():
                custom_outputs_logits, _ = model_executor.forward(
                    custom_next_token, prev_pos=original_generated.shape[1] - 1
                )
                # 确保 custom_outputs_logits 是 [B, V]
                if custom_outputs_logits.dim() == 3:
                    custom_logits = custom_outputs_logits[:, -1, :]  # [B, V]
                elif custom_outputs_logits.dim() == 2:
                    custom_logits = custom_outputs_logits  # [B, V]
                else:
                    raise ValueError(
                        f"Unexpected custom_outputs_logits dimensions: {custom_outputs_logits.dim()}"
                    )

                temperature = 0.6  # Apply temperature # custom_outputs_logits: [B, V]
                probs = torch.softmax(custom_logits / temperature, dim=-1)  # [B, V]
                custom_next_token = sample_top_p(
                    probs, p=0.9
                )  # Sample next token [B, 1]

            # Compare hidden states
            original_hidden_states = original_outputs.hidden_states
            custom_hidden_states = model_executor.model.hidden_states

            if len(custom_hidden_states) != len(original_hidden_states):
                print(
                    f"Layer count mismatch: custom {len(custom_hidden_states)}, original {len(original_hidden_states)}"
                )
                break

            print(
                f"============== Step {step + 1}: Layer Compares: ===================="
            )
            for index in tqdm(
                range(len(custom_hidden_states)),
                desc=f"Step {step + 1} Layer Comparison",
            ):
                custom_layer_output = custom_hidden_states[index]
                original_layer_output = original_hidden_states[index]

                if custom_layer_output.shape != original_layer_output.shape:
                    print(
                        f"Step {step + 1} Layer {index} shape mismatch: custom {custom_layer_output.shape}, original {original_layer_output.shape}"
                    )
                    continue

                difference = (
                    torch.abs(custom_layer_output - original_layer_output).mean().item()
                )
                print(f"Step {step + 1} Difference at layer {index}: {difference}")

            # Compare logits
            logits_diff = (
                torch.abs(original_logits - custom_outputs_logits).mean().item()
            )
            print(
                f"=========== Step {step + 1}: Logits difference is: {logits_diff} ================"
            )

            # 生成下一个 token, 模型内部已经集成了过去的 kv cache
            original_generated = torch.cat(
                [original_generated, original_next_token], dim=-1
            )
            past_key_values = original_outputs.past_key_values  # 更新 past_key_values

            # Update attention_mask if necessary
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            device=device,
                            dtype=attention_mask.dtype,
                        ),
                    ],
                    dim=-1,
                )

        print("Decode stage comparison completed.")

    def compare_models(
        self, original_model, tokenizer, input_text: str, device: str = "cuda"
    ):
        prefill_output_token = self.prefill_stage_compare(
            original_model, self.model_executor, tokenizer, input_text, device
        )
        self.decode_stage_compare(
            original_model, self.model_executor, tokenizer, input_text, device
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define model config path
    original_model_path = "/gemini/pretrain/Qwen2.5-3B"
    my_model_path = "/gemini/code/llm_weights/Qwen2.5-3B/"
    json_file_path = os.path.join(original_model_path, "config.json")  # JSON 文件的路径

    # Load config
    model_config = load_config_from_json(json_file_path, device)  # Load config

    # Load original model and tokenizer
    original_model, tokenizer, hf_sd = load_original_model(original_model_path, device)

    # Initialize ModelExecutor with correct paths
    # hf_sd = original_model.state_dict()
    # custom_model, _ = convert_qwen2_hf_to_litellama(checkpoints_dir, hf_sd, model_config, device=device)
    qwen2_test = Qwen2ModelInferTest(
        checkpoints_dir=my_model_path,
        tokenizer_path=original_model_path,  # Assuming tokenizer is at original_model_path
        max_batch_size=64,
        max_seq_len=2048,
        device=device,
    )

    # Test text
    test_text = "Once upon a time in a distant land,"

    # Compare models
    qwen2_test.compare_models(original_model, tokenizer, test_text, device)
