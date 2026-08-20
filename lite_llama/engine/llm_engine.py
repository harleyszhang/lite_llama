"""Unified generation engine.

This module replaces the four near-duplicate generators that previously lived in
``generate.py``, ``generate_stream.py``, ``llava_generate_stream.py`` and
``generete_with_probs.py`` — each of which reimplemented the same prefill/decode
loop, KV-cache allocation, EOS tracking and detokenisation with small drifts.

There is now exactly one loop, in :meth:`LLMEngine.generate`. It always streams
(yields incremental text per step); :meth:`LLMEngine.generate_text` is the thin
blocking wrapper that joins the stream. Multimodal inputs travel through the same
loop: they are consumed once during prefill and ignored thereafter, because the
vision tokens are already resident in the KV cache.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from ..executor.model_executor import ModelExecutor
from ..utils.file_interface import get_model_name_from_path
from .sampler import Sampler, SamplingParams

# Repetition detection: loops that only vary in numbers ("... for 13 years ... "
# "... for 11 years ...") are still loops, so digits are normalised away before
# comparing the trailing window against earlier text.
_DIGIT_RE = re.compile(r"\d+")

# Run the repetition check every N decode steps — cheap enough to be free, and
# the detection needs a few steps of slack anyway to observe a full loop cycle.
_REPEAT_CHECK_INTERVAL = 8


def detect_repetition(text: str, window: int = 128, min_reps: int = 3) -> bool:
    """Whether ``text`` degenerated into a repeating template loop.

    Takes the last ``window`` characters (digits normalised to a placeholder)
    and reports ``True`` when that exact block already occurred at least
    ``min_reps`` times earlier in the text. Normal prose essentially never
    repeats a 128-character block verbatim three times, while base-model
    repetition loops do it within one or two loop cycles — including loops
    whose only variation is a counting number, thanks to the normalisation.
    """
    if len(text) < window * (min_reps + 1):
        return False
    norm = _DIGIT_RE.sub("<n>", text)
    recent = norm[-window:]
    return norm[:-window].count(recent) >= min_reps


class LLMEngine:
    """Loads a model and generates text from tokenised prompts.

    Args:
        checkpoints_dir: Directory with ``config.json`` and a ``*.pth`` checkpoint.
        tokenizer_path: Tokenizer location; defaults to ``checkpoints_dir``.
        max_seq_len: Context bound; also caps the KV cache.
        max_gpu_num_blocks: Manual KV-cache size in tokens; profiled when ``None``.
        device: Torch device string.
    """

    def __init__(
        self,
        checkpoints_dir: str,
        tokenizer_path: str | None = None,
        max_seq_len: int = 2048,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool = False,
    ) -> None:
        self.device = device
        self.model_path = checkpoints_dir
        self.executor = ModelExecutor.build(
            checkpoints_dir=checkpoints_dir,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            device=device,
            use_cuda_graph=use_cuda_graph,
        )
        if use_cuda_graph:
            self.executor.enable_cuda_graph()
        self.tokenizer = self._load_tokenizer(tokenizer_path or checkpoints_dir)
        self.sampler = Sampler()
        self.max_seq_len = self.executor.max_seq_len
        # Full stop-token set (tokenizer EOS + everything generation_config
        # declares). Populated by generate(); records "eos" / "repeat" / "length"
        # per sequence so callers can explain why decoding ended.
        self._stop_ids = self._load_stop_token_ids(checkpoints_dir, self.tokenizer)
        self._stop_ids_gpu = torch.tensor(
            sorted(self._stop_ids), dtype=torch.long, device=self.device
        )
        self.last_stop_reasons: list[str] | None = None

    @staticmethod
    def _load_tokenizer(path: str) -> AutoTokenizer:
        # LLaVA ships a slow tokenizer whose fast variant changes special-token
        # handling, so it must load with use_fast=False.
        use_fast = "llava" not in get_model_name_from_path(path).lower()
        return AutoTokenizer.from_pretrained(path, use_fast=use_fast, trust_remote_code=True)

    @staticmethod
    def _load_stop_token_ids(model_path: str, tokenizer: AutoTokenizer) -> set[int]:
        """Stop ids from the tokenizer *and* ``generation_config.json``.

        HuggingFace checkpoints declare the full stop set — possibly a list,
        e.g. Qwen's ``[151645, 151643]`` — in ``generation_config.json``, while
        the tokenizer only knows one ``eos_token_id``. Instruct models usually
        terminate on a different special token than the tokenizer default
        (``<|im_end|>`` vs ``<|endoftext|>``), so without the generation config
        a finished request would keep decoding until ``max_gen_len``.
        """
        ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            ids.add(int(tokenizer.eos_token_id))
        gen_cfg_path = Path(model_path) / "generation_config.json"
        if gen_cfg_path.is_file():
            try:
                with open(gen_cfg_path) as f:
                    gen_cfg = json.load(f)
                eos = gen_cfg.get("eos_token_id")
                if isinstance(eos, int):
                    ids.add(eos)
                elif isinstance(eos, list):
                    ids.update(int(e) for e in eos)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass  # a broken generation_config must not block inference
        return ids

    @property
    def _pad_id(self) -> int:
        tok = self.tokenizer
        return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: list[list[int]],
        params: SamplingParams,
        position_ids: torch.Tensor | None = None,
        multi_modal_inputs: dict[str, Any] | None = None,
    ) -> Iterator[list[str]]:
        """Run prefill + incremental decode, yielding new text per batch each step.

        Args:
            prompt_token_ids: One token-id list per sequence in the batch.
            params: Sampling configuration.
            position_ids: Optional explicit positions (used by the mrope path for
                multimodal prompts). Defaults to a plain arange per step.
            multi_modal_inputs: Processor outputs consumed during prefill only.

        Yields:
            A list (one entry per sequence) of the text produced *this* step.
        """
        # Each generate() call is an independent batch: reset the paged-KV
        # allocator so a long-lived engine does not leak cache rows.
        self.executor.kv_mem_manager.free_all()

        batch_size = len(prompt_token_ids)
        prompt_lens = [len(ids) for ids in prompt_token_ids]
        max_prompt_len = max(prompt_lens)
        if max_prompt_len > self.max_seq_len:
            raise ValueError(
                f"prompt length {max_prompt_len} exceeds max_seq_len {self.max_seq_len}"
            )

        max_gen_len = params.max_gen_len or (self.max_seq_len - max_prompt_len)
        total_len = min(self.max_seq_len, max_prompt_len + max_gen_len)

        pad_id = self._pad_id
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.device)
        for i, ids in enumerate(prompt_token_ids):
            tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        # True where the position holds a real prompt token (never overwritten).
        prompt_mask = tokens != pad_id
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        # Number of characters already emitted per sequence, so streaming yields
        # deltas of the *decoded string* rather than of the token span.
        emitted_upto = [0] * batch_size
        b_req_idx = torch.arange(batch_size, device=self.device)
        prompt_len_tensor = torch.tensor(prompt_lens, dtype=torch.long, device=self.device)
        allocated_indices = [
            self.executor.prefill_alloc_kv_cache(max_prompt_len, prompt_len_tensor, b_req_idx)
        ]
        step = 0
        stop_reasons: list[str | None] = [None] * batch_size

        prev_pos = 0
        prompt_lens_gpu = torch.tensor(prompt_lens, dtype=torch.long, device=self.device).unsqueeze(
            -1
        )  # [batch, 1] for per-sequence position broadcasting
        # With mrope (Qwen3-VL), a vision block advances the position by only
        # max(h, w) while contributing h*w tokens, so the last prompt position is
        # *below* the token count. Decode must continue from that last mrope
        # position — the per-sequence gap to the plain token index is computed here
        # once and added to every decode step (mirrors HF's rope_deltas).
        position_deltas = None
        if position_ids is not None and position_ids.ndim == 3:
            last_index = prompt_lens_gpu.view(-1) - 1
            last_positions = position_ids[
                :, torch.arange(batch_size, device=self.device), last_index
            ]  # [3, batch] — position components of each sequence's last prompt token
            position_deltas = (
                last_positions.max(dim=0).values + 1 - prompt_lens_gpu.view(-1)
            )  # [batch]
        for cur_pos in range(max_prompt_len, total_len):
            input_ids = tokens[:, prev_pos:cur_pos]
            step_positions = self._step_positions(
                position_ids,
                input_ids,
                prev_pos=prev_pos,
                is_prefill=prev_pos == 0,
                prompt_lens=prompt_lens_gpu,
                position_deltas=position_deltas,
            )

            logits = self.executor.forward(
                input_ids,
                step_positions,
                multi_modal_inputs if prev_pos == 0 else None,
            )
            allocated_indices.append(self.executor.decode_alloc_kv_cache(batch_size))

            # After prefill, the last real token differs per sequence when prompt
            # lengths differ. Pick each sequence's own last position; for decode
            # steps seq_len == 1, so the [:, -1] path in the sampler is already right.
            if prev_pos == 0:
                last_index = prompt_lens_gpu.view(-1) - 1
                logits = logits[torch.arange(batch_size, device=self.device), last_index]
            # Generated span so far (prompt excluded) for repetition_penalty.
            generated = (
                [tokens[i, plen:cur_pos] for i, plen in enumerate(prompt_lens)]
                if params.repetition_penalty != 1.0
                else None
            )
            next_token = self.sampler.sample(logits, params, generated).reshape(-1)
            # Only fill positions that are not part of the original prompt.
            fill = ~prompt_mask[:, cur_pos]
            tokens[:, cur_pos] = torch.where(fill, next_token, tokens[:, cur_pos])
            hit_stop = fill & torch.isin(next_token, self._stop_ids_gpu)
            for i in range(batch_size):
                if bool(hit_stop[i]):
                    eos_reached[i] = True
                    if stop_reasons[i] is None:
                        stop_reasons[i] = "eos"

            # Circuit breaker: base models (and any model driven into a corner
            # by low-temperature sampling) can loop on a template forever
            # without ever emitting a stop token. Once the trailing text block
            # has already been produced several times, further tokens add no
            # information — stop the sequence.
            if (
                params.stop_on_repeat
                and step % _REPEAT_CHECK_INTERVAL == _REPEAT_CHECK_INTERVAL - 1
            ):
                for i in range(batch_size):
                    if bool(eos_reached[i]):
                        continue
                    gen_text = self.tokenizer.decode(
                        tokens[i, prompt_lens[i] : cur_pos + 1].tolist(),
                        skip_special_tokens=True,
                    )
                    if detect_repetition(gen_text):
                        eos_reached[i] = True
                        stop_reasons[i] = "repeat"

            yield self._decode_step(tokens, prompt_lens, emitted_upto, cur_pos, eos_reached)

            step += 1
            prev_pos = cur_pos
            if bool(eos_reached.all()):
                break

        # Release every cache row reserved across prefill + decode.
        self.executor.kv_mem_manager.release_ref(torch.cat(allocated_indices))
        # Sequences that never hit a stop condition ran into the length cap.
        self.last_stop_reasons = [r or "length" for r in stop_reasons]

    def _step_positions(
        self,
        position_ids: torch.Tensor | None,
        input_ids: torch.Tensor,
        prev_pos: int,
        is_prefill: bool,
        prompt_lens: torch.Tensor,
        position_deltas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return position ids for this step.

        For multimodal prefill the caller supplies mrope positions directly. For
        text prefill a single ``arange`` is fine because padded tokens are masked
        out by ``b_seq_len`` inside the attention kernel. Decode is the delicate
        case: each sequence has its own current position ``prompt_lens[i] + k``,
        so the offset must be applied per row rather than shared across the batch.
        For mrope, ``position_deltas[i]`` (see :meth:`generate`) shifts each row to
        continue from its last multimodal position instead of its token count.
        """
        if position_ids is not None and is_prefill:
            return position_ids

        _, seq_len = input_ids.shape
        if is_prefill:
            # Prefill: uniform arange, ok because padding is masked by b_seq_len.
            return (
                torch.arange(prev_pos, prev_pos + seq_len, device=self.device)
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )

        # Decode: seq_len is 1 and each sequence is at its own step count relative
        # to its own prompt length.
        step_offset = prev_pos - int(prompt_lens.max().item())
        positions = prompt_lens + step_offset  # [batch, 1]
        if position_deltas is not None:
            positions = positions + position_deltas.view(-1, 1)
        return positions

    def _decode_step(
        self,
        tokens: torch.Tensor,
        prompt_lens: list[int],
        emitted_upto: list[int],
        cur_pos: int,
        finished: torch.Tensor,
    ) -> list[str]:
        """Return the newly produced text for each sequence.

        Decoding must cover the whole generated span and diff against what was
        already emitted, never a single token in isolation. SentencePiece
        tokenizers (LLaMA, LLaVA, Vicuna) encode a leading space as the ``▁``
        marker and ``decode()`` strips it for a one-token input, so per-token
        decoding silently concatenates words: "A large black dog" comes back as
        "Alargeblackdog".

        Sequences already finished (stop token or repetition breaker) emit no
        further text: in a batch, the engine keeps stepping until *every*
        sequence is done, and without this guard a finished sequence would
        keep appending the tokens it no longer meaningfully generates.
        """
        outputs = []
        for i, prompt_len in enumerate(prompt_lens):
            if bool(finished[i]):
                outputs.append("")
                continue
            end = cur_pos + 1
            # Decode everything generated so far for this sequence...
            full = self.tokenizer.decode(
                tokens[i, prompt_len:end].tolist(), skip_special_tokens=True
            )
            # ...then emit only the part the caller has not seen yet.
            already = emitted_upto[i]
            outputs.append(full[already:])
            emitted_upto[i] = len(full)
        return outputs

    def generate_text(
        self,
        prompt_token_ids: list[list[int]],
        params: SamplingParams,
        position_ids: torch.Tensor | None = None,
        multi_modal_inputs: dict[str, Any] | None = None,
    ) -> list[str]:
        """Blocking wrapper around :meth:`generate` that returns full completions."""
        completions = ["" for _ in prompt_token_ids]
        for step in self.generate(prompt_token_ids, params, position_ids, multi_modal_inputs):
            for i, text in enumerate(step):
                completions[i] += text
        return completions
