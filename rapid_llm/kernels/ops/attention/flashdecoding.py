"""FlashDecoding: single-token decode attention against the paged KV cache.

Two stages — partial attention per KV partition, then a logsumexp-combine
— so history length is split across blocks instead of serially walked,
with scales applied inside the kernel.

Usage:
    out = flash_decoding(q, k_cache, v_cache, qk_scale,
                         b_req_tokens_table, b_req_idx, b_seq_len,
                         max_actual_seq_len)
"""

import os

import torch
import triton
import triton.language as tl

from ..quantization.w8a16 import FP8_E4M3_BIT_TRICK_SCALE, dequant_fp8e4m3

#: Stage-1 inner block. ``PARTITION_SIZE`` must be a multiple of it (asserted below).
_BLOCK_N_SIZE = 16

#: Fixed split used before the adaptive policy (kept reachable for A/B and as the
#: ``LITE_LLAMA_SPLITKV=fixed`` fallback).
_FIXED_PARTITION_SIZE = 128


def _splitkv_mode() -> str:
    """Read ``LITE_LLAMA_SPLITKV``: ``adaptive`` (default) | ``fixed`` | an int.

    An explicit integer pins ``PARTITION_SIZE`` to that value (rounded up to a
    ``_BLOCK_N_SIZE`` multiple), which is what the on/off benchmark sweeps against.
    """
    raw = os.environ.get("LITE_LLAMA_SPLITKV", "adaptive").strip().lower()
    return raw or "adaptive"


def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def adaptive_partition_size(
    batch: int,
    num_heads: int,
    max_seq_len: int,
    num_sms: int,
    *,
    blocks_per_sm: int = 16,
    baseline: int = 128,
    min_partition: int = 32,
) -> int:
    """Pick a KV partition size from the decode shape (O8 split-kv adaptivity).

    The stage-1 grid is ``(batch, num_heads, num_partitions)`` with **one warp**
    per program. On sm_86 an SM hosts at most ``blocks_per_sm`` (=16) concurrent
    blocks, so one full wave is ``num_sms * blocks_per_sm`` blocks (1152 on A10).
    ``batch * num_heads`` is the parallelism floor the shape already provides;
    splitting the KV history supplies the rest.

    The policy fixes the one case a fixed ``baseline`` measurably gets wrong:
    when ``batch * heads * seq/baseline`` lands **below one wave** — batch=1
    short/medium context, the single-request chat decode — the grid underfills
    the SMs, so the history is split finer (down to ``min_partition`` tokens per
    partition, which bounds the stage-2 combine) to raise occupancy. Measured on
    A10: batch=1 seq=512 goes 128 -> 32 (128 -> 512 blocks), worth **1.8x**;
    batch=1 seq=2048 goes 128 -> 64, worth ~1.13x.

    Everywhere else the policy returns ``baseline`` untouched. Coarsening an
    *overfilled* grid (large batch / long context) was measured and **dropped**:
    its sign flips with ``(base, seq)`` — it helped batch=4 seq=8192 (~1.04x) but
    hurt batch=64 seq=2048 (~0.95x) — and every magnitude sat inside ±5% noise,
    so it bought no reliable win while risking regressions. Keeping the baseline
    makes those cells exactly 1.0x. See ``docs/release-v0.12.0.md``.

    A pure function of Python ints, so under a captured decode graph — where
    ``batch`` and ``max_seq_len`` are baked in per bucket — it is deterministic
    and graph-safe. Output is exact for any partition size (online-softmax
    combine), so this only moves speed, never numerics.
    """
    if max_seq_len <= 0:
        return baseline
    base = max(1, batch * num_heads)
    wave = max(1, num_sms * blocks_per_sm)
    blocks_at_baseline = base * -(-max_seq_len // baseline)
    if blocks_at_baseline >= wave:
        # Already fills the GPU (or overfills it): the baseline is the measured
        # best-or-neutral choice, so do not perturb it.
        return max(_BLOCK_N_SIZE, -(-baseline // _BLOCK_N_SIZE) * _BLOCK_N_SIZE)

    needed = -(-wave // base)  # partitions to reach one wave
    max_parts = max(1, -(-max_seq_len // min_partition))
    num_parts = max(1, min(needed, max_parts))
    part = -(-max_seq_len // num_parts)  # ceil(seq / num_parts)
    part = -(-part // _BLOCK_N_SIZE) * _BLOCK_N_SIZE
    return max(min_partition, part)


def _resolve_partition_size(
    batch: int, num_heads: int, max_seq_len: int, device: torch.device
) -> int:
    """Resolve ``PARTITION_SIZE`` for one decode call per ``LITE_LLAMA_SPLITKV``."""
    mode = _splitkv_mode()
    if mode == "fixed":
        return _FIXED_PARTITION_SIZE
    if mode not in ("adaptive", "auto"):
        try:
            pinned = int(mode)
        except ValueError:
            return _FIXED_PARTITION_SIZE
        return max(_BLOCK_N_SIZE, -(-pinned // _BLOCK_N_SIZE) * _BLOCK_N_SIZE)
    return adaptive_partition_size(batch, num_heads, max_seq_len, _num_sms(device))


@triton.jit
def _flash_decoding_stage1_kernel(
    Q,
    K,
    V,
    qk_scale,
    k_scale,  # fp8 KV cache 反量化标量(bit-trick 的 2**8 补偿已由调用方折入)
    v_scale,
    b_req_tokens_table,
    B_Req_Idx,
    B_Seqlen,
    num_kv_groups,  # group of kv heads
    Mid_O,
    Mid_O_LogExpSum,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    q_bs_stride,
    q_heads_stride,
    q_dim_stride,  # Q 的 strides
    k_bs_stride,
    k_heads_stride,
    k_dim_stride,  # K 的 strides
    v_bs_stride,
    v_heads_stride,
    v_dim_stride,  # V 的 strides
    mido_batch_stride,
    mido_heads_stride,
    mido_partitions_stride,
    mido_dim_stride,
    mido_les_batch_stride,
    mido_les_heads_stride,
    mido_les_partitions_stride,
    BLOCK_SEQ: tl.constexpr,  # 默认 128
    BLOCK_N: tl.constexpr,  # 默认 32
    BLOCK_DMODEL: tl.constexpr,
    KV_FP8: tl.constexpr,  # KV cache 以 e4m3 字节存储(uint8 容器)
):
    """Flash Attention Stage1 Triton Kernel"""
    # 获取当前程序的 block 在各个维度上的索引
    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    seq_block_pid = tl.program_id(2)
    kv_head_pid = head_pid // num_kv_groups

    # 计算当前批次的起始位置
    cur_batch_seq_len = tl.load(B_Seqlen + batch_pid)
    # 该批次行属于哪个请求(KV cache 槽位): 批次内位置与槽位号无关,
    # 必须经 b_req_idx 转译后再去索引 token 映射表
    cur_req_idx = tl.load(B_Req_Idx + batch_pid)
    req_table_offset = b_req_tokens_table + stride_req_to_tokens_b * cur_req_idx

    # 计算当前分区的起始和结束索引
    cur_batch_partition_start_index = seq_block_pid * BLOCK_SEQ
    cur_batch_partition_end_index = tl.minimum(
        cur_batch_seq_len, cur_batch_partition_start_index + BLOCK_SEQ
    )

    # 计算需要处理的块数
    num_blocks = tl.where(
        cur_batch_partition_end_index - cur_batch_partition_start_index <= 0,
        0,
        (cur_batch_partition_end_index - cur_batch_partition_start_index + BLOCK_N - 1) // BLOCK_N,
    )

    # 初始化偏移向量
    offs_n = cur_batch_partition_start_index + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_d = tl.arange(0, BLOCK_DMODEL)  # [BLOCK_DMODEL]

    # 计算 Q K 的偏移量
    q_offs = batch_pid * q_bs_stride + head_pid * q_heads_stride + offs_d * q_dim_stride
    k_offs = kv_head_pid * k_heads_stride + offs_d[None, :] * k_dim_stride

    q_ptrs = Q + q_offs  # 获取 Q 指针
    q = tl.load(q_ptrs)  # # 加载 Q 向量 [BLOCK_DMODEL]

    # 初始化归一化项和累加器
    d_i = 0.0  # 标量 # 使用小的正数而不是0
    m_i = -float("inf")  # 标量
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)  # [BLOCK_DMODEL]

    # 迭代处理每个块
    for start_n in range(0, num_blocks, 1):
        # k 位置索引计算
        offs_n_new = offs_n + start_n * BLOCK_N  # [BLOCK_N]
        k_loc = tl.load(
            req_table_offset + offs_n_new,
            mask=offs_n_new < cur_batch_partition_end_index,
            other=0.0,
        )
        k_ptrs = k_loc[:, None] * k_bs_stride + k_offs

        k_mask = offs_n_new < cur_batch_partition_end_index  # [BLOCK_N]

        k = tl.load(K + k_ptrs, mask=k_mask[:, None], other=0.0)
        v = tl.load(V + k_ptrs, mask=k_mask[:, None], other=0.0)

        if KV_FP8:
            # e4m3 字节经 bit surgery 直接升到 fp32(欠 2**8, 已折入 scale)。
            # 屏蔽行的字节为 0 -> 0.0, 与 fp16 路径的 other=0.0 语义一致。
            k = dequant_fp8e4m3(k).to(tl.float32) * k_scale
            v = dequant_fp8e4m3(v).to(tl.float32) * v_scale

        # 计算 qk^T。逐元素乘积必须升到 fp32 再累加: prefill 路径走
        # ``tl.dot`` 的 fp32 累加器, 若这里保持 fp16 乘积, 64 项点积的舍入误差
        # (~1e-2) 会逐层放大成 logits 的 ~5e-2 噪声, 足以让 greedy argmax 在
        # 边缘 token 上翻转并滑入重复吸引子。
        qk = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1)  # [BLOCK_N]
        qk *= qk_scale
        qk = tl.where(k_mask, qk, float("-inf"))  # [BLOCK_N]

        # 更新最大值项和 qk 项
        current_max = tl.max(qk)  # 标量
        m_ij = tl.maximum(m_i, current_max)  # 标量
        p = tl.exp(qk - m_ij)  # [BLOCK_N]

        # 更新归一化项
        alpha = tl.exp(m_i - m_ij)
        d_i = alpha * d_i + tl.sum(p, axis=0)

        # 更新 attention 输出累加器 (p 已是 fp32, v 升 fp32 与 qk 同理)
        acc = alpha * acc + tl.sum(p[:, None] * v.to(tl.float32), axis=0)  # [BLOCK_DMODEL]
        # acc = acc * alpha + tl.dot(p, v)  # [BLOCK_DMODEL]

        # 更新归一化器
        m_i = m_ij

    # 计算存储的偏移量
    off_mid_o = (
        batch_pid * mido_batch_stride
        + head_pid * mido_heads_stride
        + seq_block_pid * mido_partitions_stride
        + offs_d * mido_dim_stride
    )

    off_mid_o_les = (
        batch_pid * mido_les_batch_stride
        + head_pid * mido_les_heads_stride
        + seq_block_pid * mido_les_partitions_stride
    )

    # 本分区完全落在该行序列长度之外时不写: mid_o 只按实际分区数分配,
    # 越界分区没有属于它的存储。用 0 次迭代的循环表达, 因为 triton 里
    # ``if`` 包住 ``tl.store`` 会把整个 store 变成谓词化写入。
    need_store = tl.where(num_blocks == 0, 0, 1)
    for _ in range(0, need_store, 1):
        tl.store(Mid_O + off_mid_o, acc / d_i)
        tl.store(Mid_O_LogExpSum + off_mid_o_les, m_i + tl.log(d_i))


@torch.no_grad()
def flash_decode_stage1(
    q,
    k,
    v,  # Q: [batchs, num_heads, head_dim], K, V: [batchs * seq_len, num_heads, head_dim]
    qk_scale,
    b_req_tokens_table,
    b_req_idx,
    b_seq_len,
    max_actual_seq_len,  # 最大的实际序列长度
    mid_o,
    mid_o_logexpsum,
    PARTITION_SIZE,
    k_scale=1.0,  # fp8 KV cache 的 K 反量化标量(含 2**8 补偿)
    v_scale=1.0,
    kv_fp8=False,
):
    """
    # Mid_O: [batchs, num_heads, cdiv(seq_len, PARTITION_SIZE), head_dim],
    # Mid_O_LogExpSum: [batchs, num_heads, cdiv(seq_len, PARTITION_SIZE)]
    """
    BLOCK_N_SIZE = 16

    # BLOCK_DMODEL = q.shape[-1]
    assert PARTITION_SIZE % BLOCK_N_SIZE == 0, "PARTITION_SIZE 必须是 BLOCK_N_SIZE 的倍数"

    batchs, num_heads, head_dim = (
        q.shape
    )  # decode 阶段 q 张量的 seq_len = 1, 这里的 batchs 实际就是 batch_size

    # grid 配置的并行度比 flashattention1-2 多了 kv cache seq 维度。z 维必须与
    # ``flash_decoding`` 里 mid_o 的分区数完全一致: 多启动一个 block 会让它算出
    # 一个 mid_o 里不存在的分区偏移, 目前只靠 num_blocks == 0 的不写才没有越界。
    grid = (
        batchs,
        num_heads,
        triton.cdiv(max_actual_seq_len, PARTITION_SIZE),
    )
    num_kv_groups = q.shape[1] // k.shape[1]  # num_q_heads // num_k_heads

    _flash_decoding_stage1_kernel[grid](
        q,
        k,
        v,
        qk_scale,
        k_scale,
        v_scale,
        b_req_tokens_table,
        b_req_idx,
        b_seq_len,
        num_kv_groups,  # kv 组数量
        mid_o,
        mid_o_logexpsum,
        *b_req_tokens_table.stride(),
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        BLOCK_SEQ=PARTITION_SIZE,
        BLOCK_N=BLOCK_N_SIZE,
        BLOCK_DMODEL=head_dim,
        KV_FP8=kv_fp8,
        num_warps=1,
        num_stages=2,
    )


@triton.jit
def _flash_decoding_stage2_kernel(
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    Ouput,  # attention 输出首地址
    mido_batch_stride,
    mido_heads_stride,
    mido_partitions_stride,
    mido_dim_stride,
    mido_les_batch_stride,
    mido_les_heads_stride,
    mido_les_partitions_stride,
    o_bs_stride,
    o_heads_stride,
    o_dim_stride,
    B_Seqlen,  # [batch] 每行的历史长度, 决定该行要归约多少个分区
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,  # type: ignore
):
    """Reduction (online softmax)"""
    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    cur_batch_seq_len = tl.load(B_Seqlen + batch_pid)

    # 初始化偏移
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # 最后一个维度 stride 为 1 可省略, 如 mido_dim_stride
    offs_part_v = batch_pid * mido_batch_stride + head_pid * mido_heads_stride + offs_d

    offs_part_max = batch_pid * mido_les_batch_stride + head_pid * mido_les_heads_stride

    part_v_ptrs = Mid_O + offs_part_v
    part_max_ptrs = Mid_O_LogExpSum + offs_part_max

    # Reduce kv 分块相关变量值. num_partitions 是 kv 分块数量
    d_i = 0.0
    m_i = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    num_partitions = (cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    for block_seq_n in range(0, num_partitions, 1):
        part_v = tl.load(part_v_ptrs + block_seq_n * mido_partitions_stride)
        part_max = tl.load(part_max_ptrs + block_seq_n)  # mido_les_partitions_stride = 1

        # -- 更新局部最大值 -- #
        m_ij = tl.maximum(part_max, m_i)
        # -- 计算 alpha = exp(m{j-1} - m{j}) 值 -- #
        alpha = tl.exp(m_i - m_ij)

        # -- 更新归一化项和 attention 输出累加器 -- #
        p = tl.exp(part_max - m_ij)
        acc = alpha * acc + p * part_v

        # alpha * d_i: 缩放 d_i, p * weight: 当前元素的指数值 * 权重
        d_i = alpha * d_i + p

        # 更新 max 值和指针偏移
        m_i = m_ij

    # -- 更新 attention 输出累加器 -- #
    offs_out = batch_pid * o_bs_stride + head_pid * o_heads_stride + offs_d * o_dim_stride
    # 长度为 0 的行没有分区可归约, acc 和 d_i 都还是初值; 除数兜底成 1.0 让它
    # 写出零向量, 而不是把 NaN 播进 o_proj 和之后的每一层。
    tl.store(Ouput + offs_out, acc / tl.where(d_i > 0.0, d_i, 1.0))


@torch.no_grad()
def flash_decode_stage2(
    mid_o,
    mid_o_logexpsum,  # 存储每个批次、每个头、每个分区的中间分数输出及 log(sum(exp(scores)))
    atten_output,  # attention 输出首地址
    b_seq_len,  # kv cache 在 seq_len 维度的长度向量
    PARTITION_SIZE,
):
    batchs, num_heads, HEAD_DIM = mid_o.shape[0], mid_o.shape[1], mid_o.shape[-1]
    grid = (batchs, num_heads)

    _flash_decoding_stage2_kernel[grid](
        mid_o,  # [batch, head, seq_block_num, head_dim]
        mid_o_logexpsum,  # [batch, head, seq_block_num]
        atten_output,  # attention 输出首地址
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        *atten_output.stride(),
        b_seq_len,
        BLOCK_DMODEL=HEAD_DIM,
        BLOCK_SEQ=PARTITION_SIZE,  # type: ignore
        num_warps=4,
        num_stages=2,
    )


@torch.no_grad()
def flash_decoding(
    q,  # q 查询向量，形状为 [bsz, num_head, head_dim]
    k_cache,
    v_cache,  # 键/值向量缓存，形状为 [max_tokens, kv_num_head, head_dim]
    qk_scale,
    b_req_tokens_table,
    b_req_idx,  # which cache slot each batch row belongs to
    b_seq_len,  # start locations and sequence lengths for kv cache in a batch
    max_actual_seq_len,
    k_scale: float = 1.0,  # fp8 KV cache 的逐张量反量化标量(vLLM kv_scale 语义)
    v_scale: float = 1.0,
):
    """Decode attention for one token per sequence.

    Args:
        q: ``[batch, num_heads, head_dim]`` — decode has ``seq_len == 1``.
        k_cache: ``[max_tokens, num_kv_heads, head_dim]`` paged key cache.
        v_cache: Value cache, same shape as ``k_cache``.
        qk_scale: Softmax scale, ``1 / sqrt(head_dim)``.
        b_req_tokens_table: ``[max_requests, max_seq_len]`` position-to-cache-row map.
        b_req_idx: ``[batch]`` request/slot id owning each batch row. Batch order
            is not slot order once requests join and leave a running batch, so
            this is what makes the lookup correct rather than coincidental.
        b_seq_len: ``[batch]`` history length per row, including this step's token.
        max_actual_seq_len: Longest row, which sizes the partition grid.
        k_scale: Dequantisation scale of an fp8 key cache; ignored for fp16.
        v_scale: Same for the value cache.
    """
    # q.view(-1, num_heads, head_dim)
    assert q.shape[-1] == k_cache.shape[-1] == v_cache.shape[-1]
    batchs, num_heads, head_dim = q.shape  # decode 阶段 q 的 seq_len = 1,

    # O8: split the KV history by the decode shape rather than a fixed 128.
    # batch=1 long context splits finer to fill the SMs; large batch splits
    # coarser because batch*heads already saturates and the stage-2 combine is
    # pure overhead. Exact for any partition size, and a pure function of Python
    # ints so a captured graph bakes in one deterministic value per bucket.
    PARTITION_SIZE = _resolve_partition_size(batchs, num_heads, max_actual_seq_len, q.device)

    kv_fp8 = k_cache.dtype == torch.uint8
    if kv_fp8:
        # dequant_fp8e4m3 的 bit-trick 输出欠 2**8; 折进 scale, kernel 内免补偿
        k_scale = k_scale * FP8_E4M3_BIT_TRICK_SCALE
        v_scale = v_scale * FP8_E4M3_BIT_TRICK_SCALE

    # 最大可用分区数量计算
    max_num_partitions = (max_actual_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE

    # mid_o: 存储每个批次、每个头、每个分区的中间输出
    mid_o = torch.empty(
        (batchs, num_heads, max_num_partitions, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    # 存储每个批次、每个头、每个分区的 log(sum(exp(scores)))，用于后续 decode_stage2 的归一化
    mid_o_logexpsum = torch.empty(
        (batchs, num_heads, max_num_partitions), dtype=torch.float32, device=q.device
    )

    # decode stage 1: attention in partitions
    flash_decode_stage1(
        q,
        k_cache,
        v_cache,
        qk_scale,
        b_req_tokens_table,
        b_req_idx,
        b_seq_len,
        max_actual_seq_len,
        mid_o,
        mid_o_logexpsum,
        PARTITION_SIZE,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_fp8=kv_fp8,
    )

    # decode stage 2: reduction among partitions
    atten_output = torch.empty_like(q)

    flash_decode_stage2(mid_o, mid_o_logexpsum, atten_output, b_seq_len, PARTITION_SIZE)

    return atten_output
