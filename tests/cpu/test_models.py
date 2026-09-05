"""Small model logits compared with the same Hugging Face weights on CPU."""

import pytest
import torch
import transformers

from rapid_llm.executor.model_runner import ModelRunner


@pytest.mark.parametrize("family", ["Qwen2", "Qwen3", "Qwen3Moe", "DeepseekV2"])
@torch.inference_mode()
def test_model_prefill_and_cached_decode(tmp_path, family):
    torch.manual_seed(19)
    config_cls = getattr(transformers, f"{family}Config")
    model_cls = getattr(transformers, f"{family}ForCausalLM")
    options = {
        "vocab_size": 64,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 32,
    }
    if family == "Qwen3Moe":
        options.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128)
    elif family == "DeepseekV2":
        options.update(
            num_key_value_heads=4,
            num_experts_per_tok=2,
            n_routed_experts=4,
            n_shared_experts=1,
            moe_intermediate_size=128,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_nope_head_dim=32,
            qk_rope_head_dim=64,
            v_head_dim=32,
            n_group=2,
            topk_group=1,
            first_k_dense_replace=1,
        )
    config = config_cls(**options)
    reference = model_cls(config).to(torch.bfloat16).eval()
    reference.save_pretrained(tmp_path)
    runner = ModelRunner.build(str(tmp_path), 16, max_gpu_num_blocks=64, device="cpu")
    ids = torch.tensor([[1, 4, 6]])
    runner.prefill_alloc_kv_cache(
        3, torch.tensor([3], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)
    )
    actual = runner.model(ids, torch.arange(3)[None], runner.atten_info)
    expected = reference(ids).logits
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.012, rtol=0.04)
    runner.decode_alloc_kv_cache(1)
    actual = runner.model(torch.tensor([[8]]), torch.tensor([[3]]), runner.atten_info)
    expected = reference(torch.tensor([[1, 4, 6, 8]])).logits[:, -1:]
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.012, rtol=0.04)
