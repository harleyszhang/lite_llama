# Quantisation matrix — 2×H100 80GB, 2026-09-01

Everything below was measured on one machine in one day: 2× NVIDIA H100 80GB HBM3
(sm90, 3352 GB/s, 989 TFLOP/s dense tensor core, 50 MiB L2), torch 2.13.0+cu130,
triton 3.7.1, python 3.14.7, `v0.8.0-32-g4f256d7`. Neither `deep_gemm` nor
`flashinfer` is installed, so every kernel row is the native Triton path.

Models, from `$LITE_LLAMA_MODELZOO`:

| Code | Checkpoint | Role |
|---|---|---|
| `M_TINY` | Qwen2.5-0.5B-Instruct | fast axis sweeps, golden baseline |
| `M_MAIN` | Qwen3-4B-Thinking-2507 | the offline and online matrices |
| `M_MOE` | Qwen3-30B-A3B-Instruct-2507 | fp8 fused MoE end to end |

Two accuracy references, and the distinction matters for reading any row:

- **golden** — greedy agreement against this engine at bf16/eager/TP1, recorded in
  Phase 0 (`tests/golden/data/`). Isolates *quantisation* error, because engine and
  tokeniser are held fixed.
- **HF fp16** — agreement against `transformers` at fp16. Reported once, on the bf16
  row, where it comes out at **0.348**. That is the *engine's* divergence from
  HuggingFace at matched precision — paged attention, a different softmax reduction
  order, fused RMSNorm — and it is why the golden reference exists. A quantisation
  scheme measured against HF would be charged for all of that as well.

## 1. Kernel level

### 1.1 Quantised dense GEMM

Full table in
[`bench_quant_gemm_h100_20260901.json`](bench_quant_gemm_h100_20260901.json), over
`M ∈ {1, 8, 32, 128, 512, 2048}` × the real Qwen3-4B and Qwen3-30B-A3B projection
shapes (48 cells), run with `LITE_LLAMA_AUTOTUNE=0`.

**cuBLAS bf16 is fastest in 44 of 48 cells.** The four exceptions are the useful
part: all are `qwen3-4b/gate_up` (N=19456, K=2560 — 100 MB in bf16, the largest
weight here) at M ≤ 128.

| projection (µs) | bf16 | fp8 W8A16 | fp8 W8A8 | int8 W8A8 | int4 awq | nvfp4 |
|---|---|---|---|---|---|---|
| gate_up, M=1 | 48.8 | **45.8 (1.06×)** | **36.1 (1.35×)** | **34.8 (1.40×)** | **44.4 (1.10×)** | 117.9 |
| gate_up, M=32 | 50.2 | 65.1 | **38.8 (1.29×)** | **37.7 (1.33×)** | 66.6 | 124.2 |
| gate_up, M=2048 | **304.0** | 1125.4 | 513.6 | 313.2 | 1008.6 | 2267.9 |
| qkv, M=1 | **21.7** | 28.2 | 24.0 | 22.7 | 22.2 | 49.0 |
| qkv, M=2048 | **89.1** | 370.0 | 166.1 | 103.8 | 334.8 | 755.3 |

At `gate_up` M=1 **four of the five formats beat bf16**, including int4; by M=2048 on
the same shape none do and nvfp4 is 7.5× behind.

The rule those four cells expose: **quantisation wins where bf16 is genuinely
bandwidth-bound, and nowhere else.** At M=1 the bf16 row's share of peak HBM spans
10.4% (`qwen3-30b-a3b/down`, a 3 MB weight) to 60.9% (`qwen3-4b/gate_up`), and the
wins sit at the top of that range. Below ~50% the kernel is not waiting on memory, so
deleting weight bytes deletes nothing and the dequant is pure addition. Predict from
the bf16 `%bw` column, not from the compression ratio — the compression ratio is what
an earlier draft of this section used, and it produced the exactly-backwards claim
that nvfp4's 3.6× fewer bytes would make it fastest at decode. It is the slowest row
in all 48 cells.

Two more readings, per regime:

- **Decode (M ≤ 32)**, memory-bound in principle. `moved` is *not* the ranking; the
  dequant rate is. Where int4 reaches parity on `qkv` (22.2 vs 21.7 µs) it is two
  different limits meeting — bf16 streams 31 MB at 43% of peak and is
  bandwidth-bound, int4 streams 7.9 MB at 11.9% and is unpack-bound — so the parity
  does not transfer to another shape.
- **Prefill (M ≥ 512)**, compute-bound. **No quantised row wins any cell.** cuBLAS
  reaches 73% of tensor-core peak while the Triton rows stay pinned by their unpack
  loops (1.17× for int8 up to 8.5× for nvfp4). fp8 W8A8 reaches 39% of peak — it is
  the only format that *could* win here on arithmetic and it does not.

### 1.1b The int4 heuristic defect `--tune` found

Of the five quantised dense kernels, **only `w4a16_matmul` consults `ConfigStore`**;
fp8 W8A8, fp8/int8 W8A16 and NVFP4 compute their launch config unconditionally, so
`bench_quant_gemm.py --tune` reports them as having no consumer instead of writing
entries no kernel would read. (v0.5's changelog claims autotune covers "量化 GEMM";
for the dense path that is one kernel of five.)

On that kernel the sweep found a heuristic defect, not a per-shape tuning
opportunity. The `m ≤ 32` branch used `GROUP_M=1, num_stages=2`; the *same* 16×64 tile
with `GROUP_M=8, num_stages=4` won at **all 16** `m ≤ 32` store keys (two geometries ×
four projections × the M16/M32 buckets) by 9.0–41.5%, with the tile held fixed so
those two knobs are the only variables. `GROUP_M=1` groups nothing, so consecutive
programs walk the grid row-major and share no weight tile in L2. Because the win was
uniform it belongs in the kernel fallback rather than a shape-keyed cache: it now
ships without a tuning run, and it is what moved M=1 int4 on `qkv` from 34.0 to
22.2 µs — i.e. the parity above is a consequence of this fix, not a property of int4.

After the fix, per-shape tuning still has plenty left: 29 of 32 keys improve on the
corrected heuristic, by 9.7–46.0%. Only three report "heuristic already best" — the M16
keys of `qwen3-4b/qkv`, `qwen3-4b/gate_up` and `qwen3-30b-a3b/qkv`. Elsewhere the
winners are *narrower* than the heuristic at decode (16×32 or 64×32 across M16/M32) and
much wider at prefill (128×64 through 128×256 at M512) — a spread a three-branch
fallback cannot cover, which is why this is the one dense kernel worth caching.
Caveat: a bucket entry is chosen for the *total* over the token counts sharing it, so
one width inside a bucket can regress while the entry is still a net win.
Spot-checked on `qwen3-30b-a3b/qkv` and `qwen3-4b/qkv`, both widths improved on both
keys (t512 +0.7% / +12.2%, t2048 +25.5% / +24.3%), so no regression was observed — but
a decode-only deployment should still narrow `--tokens` to the widths it serves.

### 1.2 Fused MoE

Qwen3-30B-A3B geometry (E=128, top_k=8, h=2048, i=768), µs, run with
`LITE_LLAMA_AUTOTUNE=0` so these are the heuristic tiles a user gets with no tuning
cache. From
[`bench_fused_moe_h100_20260901.json`](bench_fused_moe_h100_20260901.json):

| tokens | fp16 | fp8 W8A16 | fp8 W8A8 | int8 | int4 | `moe_align` only | fp16 @ old BLOCK_K=32 |
|---|---|---|---|---|---|---|---|
| 1 | 360.7 | 365.2 | 481.1 | 364.3 | 366.2 | 188.2 | 359.6 |
| 8 | 363.7 | 368.5 | 481.5 | 367.3 | 367.2 | 186.6 | 363.8 |
| 64 | 531.7 | 439.1 | 484.8 | **398.9** | 630.0 | 180.9 | 713.4 |
| 512 | 583.4 | 615.5 | **477.8** | 576.9 | 691.8 | 182.0 | 753.9 |
| 4096 | 1573.2 | 2301.0 | **1469.4** | 2096.7 | 2598.7 | 217.2 | 1755.5 |

Three regimes, and no format wins in more than one:

- **≤8 tokens**: fp16 and all three weight-only formats land inside 1.5% while
  reading 4× different weight bytes, because `moe_align_block_size` alone is ~188 µs
  — over half the layer, identical for every format, and the largest single target on
  this path. W8A8 is the only distinguishable row and it is 33% *worse*: two extra
  quantisation launches on an already launch-bound layer.
- **64 tokens**: weight-only wins. int8 by 25.0%, W8A16 fp8 by 17.4%. int4 does not
  — 18.5% slower than fp16, lowest GB/s in the table while reading the fewest bytes,
  so it is unpack-bound, not traffic-bound.
- **≥512 tokens**: the ordering inverts. W8A8 fp8 is fastest (18.1% under fp16 at
  512, 6.6% at 4096, 210 TFLOP/s — the highest arithmetic rate anywhere in this
  round) while W8A16 fp8 and int8 become outright regressions against plain fp16
  (46% and 33% slower at 4096).

A caveat the TFLOP/s column hides: Triton emits Hopper fp8 `wgmma` only from
`BLOCK_M ≥ 64`, which the heuristic reaches above 64 tokens. The t1/t8/t64 W8A8 rows
widen both e4m3 operands to an fp16 `mma.sync` and are not measuring the fp8 tensor
cores at all — consistent with the win starting at 512.

### 1.3 A defect the baseline column found

`_launch_config` returned `BLOCK_K = 128 if quant_mode else 32`. The tile sweep
(`bench_fused_moe.py --tune`) found **no** winning fp16 config with `BLOCK_K` below
64 at any token count; the narrow tile cost 25.5% at t64, 22.6% at t512, 10.4% at
t4096. It only ever depressed the *unquantised baseline*, which is why no test caught
it and why at t512 W8A16 fp8 used to read as an 18% win instead of the 5.5% loss
above. Fixed: all modes get 128. The last column keeps the old tile as a regression
guard — the two fp16 columns converging means it came back.

The same sweep, persisted to `ConfigStore`, improved **13 of 15** store keys
(largest: fp16 at the M512 bucket, 2502.8 → 1694.1 µs, +32.3%; fp8 W8A8 at M512 and
int4 at M512 were the two the heuristic already had right). The cache is per-device
and not committed; run `--tune` on a new GPU.

## 2. Offline matrix — `M_MAIN` (Qwen3-4B-Thinking-2507)

batch 4, 64 generated tokens, KV capacity profiled rather than fixed, so the memory
columns are the measurement.
[A](bench_quant_main_A_h100_20260901.json) ·
[B/C](bench_quant_main_BC_h100_20260901.json) ·
[D/E](bench_quant_main_DE_h100_20260901.json) ·
[F](bench_quant_main_F_h100_20260901.json) ·
[G](bench_quant_main_G_h100_20260901.json)

The five `int4` rows were re-measured after the `w4a16` tile fix of §1.1b, with the
same commands and `LITE_LLAMA_AUTOTUNE=0`, and it is those numbers that are in the
table: [BC](bench_quant_main_int4fix_BC_h100_20260901.json) ·
[DE](bench_quant_main_int4fix_DE_h100_20260901.json) ·
[F](bench_quant_main_int4fix_F_h100_20260901.json). The pre-fix values are kept in
the row notes below the table rather than deleted — the delta is the fix's e2e size,
and it is the only reason this whole section is not a single consistent run.

| config | TTFT ms | TPOT ms | TPS | model GB | KV tok | golden | prefix |
|---|---|---|---|---|---|---|---|
| HF fp16 (reference) | 35.4 | 56.47 | 71.3 | — | — | — | — |
| **A** bf16+tp1+graph | 22.0 | 4.77 | 793.0 | 7.49 | 447,830 | **1.000** | 1.000 |
| **A** bf16+tp1+eager | 21.6 | 22.23 | 180.0 | 7.49 | 458,752 | 1.000 | 1.000 |
| **B** fp8+tp1+eager | 32.0 | 32.17 | 124.3 | 4.11 | 476,672 | 0.659 | 0.617 |
| **B** int8+tp1+eager | 25.1 | 25.32 | 158.0 | 4.11 | 476,653 | 0.907 | 0.822 |
| **B** int4+tp1+eager | 25.7 | 26.12 | 153.1 | 2.63 | 488,320 | 0.157 | 0.139 |
| **B** smoothquant+tp1+eager | 29.3 | 29.43 | 135.9 | 4.11 | 476,667 | 0.181 | 0.051 |
| **B** nvfp4+tp1+eager | 25.3 | 25.75 | 155.3 | 2.63 | 488,420 | 0.249 | 0.233 |
| **C** fp8+tp1+graph | 32.0 | 5.60 | 664.3 | 4.11 | 465,750 | 0.659 | 0.617 |
| **C** int8+tp1+graph | 25.4 | 7.11 | 540.5 | 4.11 | 465,730 | 0.907 | 0.822 |
| **C** int4+tp1+graph | 24.8 | 6.97 | 551.3 | 2.63 | 477,398 | 0.157 | 0.139 |
| **C** smoothquant+tp1+graph | 29.0 | 5.26 | 710.0 | 4.11 | 465,745 | 0.181 | 0.051 |
| **C** nvfp4+tp1+graph | 26.2 | 13.66 | 288.7 | 2.63 | 477,497 | 0.249 | 0.233 |
| **D** fp8+tp2+eager | 38.2 | 41.11 | 97.4 | 2.06 | 977,678 | 0.676 | 0.608 |
| **D** int4+tp2+eager | 31.4 | 34.32 | 116.7 | 1.31 | 989,653 | 0.160 | 0.139 |
| **E** fp8+tp2+graph | 40.0 | 5.89 | 622.2 | 2.06 | 955,832 | 0.676 | 0.608 |
| **E** int4+tp2+graph | 33.8 | 6.43 | 583.1 | 1.31 | 967,808 | 0.160 | 0.139 |
| **F** fp8+dp2+graph | — | — | 1275.3 | — | — | 0.659 | 0.617 |
| **F** int4+dp2+graph | — | — | 1057.6 | — | — | 0.157 | 0.139 |
| **G** bf16+kvfp8+tp1+graph | 26.1 | 5.47 | 690.2 | 7.49 | 895,692 | 0.718 | 0.703 |
| **G** fp8+kvfp8+tp1+graph | 37.2 | 6.34 | 586.0 | 4.11 | 931,503 | 0.617 | 0.574 |

`model GB` on a tp2 row is **per rank** (`note: rank 0 shard`); D/E and the fp8 tp1
baseline use continuous batching on both sides so a tp1↔tp2 comparison differs in
parallelism only, not in scheduler.

What the grid says:

- **CUDA graphs are the largest single win, and they are orthogonal to
  quantisation.** 4.4× on bf16 (180 → 793 TPS), 5.3× on fp8, 5.2× on smoothquant,
  3.6× on int4, 3.4× on int8, 1.9× on nvfp4. Every scheme captured, including the
  TP2 rows. That the *quantised* paths capture at all is the Phase 2/3/5 result:
  their launch configs are shape-dependent, and a host synchronisation anywhere in
  them would have made the layer uncapturable. The spread across schemes is itself
  informative — a graph removes launch overhead, so the scheme that gains least
  (nvfp4, 1.9×) is the one whose kernel does the most real work per launch. int4 used
  to be in that sentence too, at 2.8×; the tile fix of §1.1b moved it to 3.6×, which
  is what "the kernel was doing avoidable work per launch" looks like from here.
- **What the `w4a16` tile fix bought end to end.** Pre-fix → post-fix, same commands:
  C int4+tp1+graph 419.0 → 551.3 TPS (TPOT 9.28 → 6.97 ms), E int4+tp2+graph 471.9 →
  583.1 (8.06 → 6.43), F int4+dp2 816.8 → 1057.6, and the eager rows barely move
  (B 149.7 → 153.1, D 112.3 → 116.7) because eager decode is dominated by launch
  overhead the tile cannot touch. Accuracy is bit-identical across the change
  (0.157/0.139 before and after), as a tile-config change must be. int4 goes from the
  slowest graph row to mid-table, and it is still slower than bf16.
- **No quantisation scheme is faster than bf16 at this size.** bf16+graph is the
  fastest row in the table. At 4B the weights already fit and decode is
  launch-bound, so a weight-only format pays its unpacking cost against no traffic
  it needed to save. Quantisation here buys **memory**: int4 cuts the weights to a
  third (7.49 → 2.63 GB) and fp8+tp2 to 27% per rank (7.49 → 2.06 GB), and the
  freed memory becomes KV capacity — 447,830 → 967,808 tokens, 2.2×.
- **TP is a capacity feature at this size, not a speed feature.** fp8+tp2+graph is
  *slower* than fp8+tp1+graph (622 vs 644 TPS with the scheduler held fixed): the
  per-step all-reduce costs more than the second card's compute buys at 4B. It
  doubles KV capacity, which is the reason to use it.
- **DP scales where TP does not.** fp8+dp2 reaches 1275 TPS against 664 at tp1 —
  1.92×, near-linear, because the replicas share nothing per step. int4+dp2 is 1058
  against 551, 1.92× as well.
- **Accuracy ranks int8 (0.907) > fp8 (0.659) > nvfp4 (0.249) > smoothquant (0.181)
  > int4 (0.157)**, and the spread is far larger than the speed spread. On a
  *reasoning* checkpoint one diverged token reroutes the rest of the chain, so these
  are closer to "did the completion survive intact" than to a per-token error rate.
  int8 is the only row that could be called nearly lossless. smoothquant is the worst
  combination in the table — a 0.181 match rate but only a 0.051 *prefix* rate, so it
  diverges very early and then stays wrong — and it is a runtime-quantised path with
  no calibration data, which is the likely cause.
- **KV fp8 costs 0.28 accuracy on bf16** (1.000 → 0.718) and buys 2.0× KV capacity
  (447,830 → 895,692). See §4.

### `M_MOE` (Qwen3-30B-A3B-Instruct-2507) — group H

batch 4, 32 generated tokens, `--max-seq-len 2048`. No golden baseline exists for
this checkpoint, so accuracy columns are empty by construction, not by omission.
[fp8](bench_quant_moe_fp8_h100_20260901.json) ·
[bf16](bench_quant_moe_bf16_h100_20260901.json)

| config | model GB | peak GB | KV tok | TTFT ms | TPOT ms | TPS |
|---|---|---|---|---|---|---|
| fp8+tp1+graph | 29.11 | 68.97 | 420,348 | 73.1 | 11.11 | 306.5 |
| fp8+tp2+graph | 14.59 (rank 0) | 68.16 | 1,141,756 | 88.2 | 10.96 | 298.9 |
| bf16+tp2+graph | 28.45 (rank 0) | 68.94 | 855,078 | 76.4 | 11.17 | 302.7 |

All three land within 2.5% of each other on throughput, and TPOT is flat at ~11 ms.
This is the kernel table's decode regime showing up end to end: at 4 concurrent
sequences the MoE layer is launch-bound behind `moe_align_block_size`, and fp8-A8's
kernel-level 33% decode penalty is diluted by the rest of the model into no
measurable difference. **fp8-A8 MoE is not a decode optimisation.** What it does buy
is capacity: fp8+tp1 fits the 57 GB bf16 checkpoint on *one* card at 29.11 GB, and
fp8+tp2 reaches 1.14M KV tokens — 2.7× the tp1 figure and 1.34× bf16+tp2. The
prefill gain measured at the kernel (18% at 512 tokens) needs a prefill-heavy
workload to appear; this cell does not have one.

### Companion run — the *native* FP8 checkpoint

The rows above quantise the bf16 checkpoint at runtime. The released
`Qwen3-30B-A3B-Instruct-2507-FP8` checkpoint (fp8-e4m3 + 128×128 block scales,
`quant_method: fp8`) is served by the W8A16 path instead — block-scale weights do not
match the per-channel layout `w8a8_fp8` expects — and was measured on 2026-09-01 over
the full axis matrix (tp1/tp2 × graph/eager × kv auto/fp8, plus dp2, with a golden
baseline recorded for it): tp1+graph lands at TPOT 13.16 ms / 285.9 TPS, tp2+graph at
12.76 ms / 290.3 TPS with 2.7× the KV capacity, and CUDA graph is worth 4.8× at this
size. Full table and accuracy columns in
[`quantization.md` § Qwen3-30B-A3B-Instruct-2507-FP8](../quantization.md), raw JSON in
[`bench_quant_Qwen3-30B-A3B-FP8_20260901.json`](bench_quant_Qwen3-30B-A3B-FP8_20260901.json)
(+ `-dp` for the data-parallel row).

## 3. Online matrix — `M_MAIN`, `lite-llama serve`

64 max tokens, `max_seq_len` 1024, concurrency 1/8/32 over `POST /v1/completions`,
`temperature=0`. [tp](bench_serving_main_tp_h100_20260901.json) ·
[dp](bench_serving_main_dp_h100_20260901.json)

The three `int4` rows are post-§1.1b re-measurements, same commands
([tp](bench_serving_main_int4fix_tp_h100_20260901.json) ·
[dp](bench_serving_main_int4fix_dp_h100_20260901.json)); the other six are the
original run.

| config | conc | TTFT mean | TTFT p99 | TPOT | TPS | batch | in-wave dup | offline | dup batch |
|---|---|---|---|---|---|---|---|---|---|
| bf16+tp1 | 1 | 24.2 | 24.2 | 4.70 | 199.7 | 1.000 | — | 1.000 | 1.000 |
| bf16+tp1 | 8 | 46.4 | 47.3 | 5.50 | 1294.3 | 0.883 | — | 1.000 | 1.000 |
| bf16+tp1 | 32 | 179.2 | 187.2 | 7.28 | 3148.2 | 0.890 | 0.766 | 1.000 | 1.000 |
| bf16+tp2 | 32 | 222.3 | 249.3 | 7.91 | 2730.4 | 0.760 | 1.000 | 1.000 | 1.000 |
| fp8+tp1 | 32 | 202.7 | 217.2 | 7.89 | 2803.0 | 0.546 | 0.766 | 1.000 | 1.000 |
| fp8+tp2 | 32 | 147.2 | 160.0 | 8.54 | 2879.8 | 0.633 | 1.000 | 1.000 | 1.000 |
| int4+tp1 | 32 | 143.6 | 157.7 | 9.75 | 2585.0 | 1.000 | 1.000 | 1.000 | 1.000 |
| int4+tp2 | 32 | 232.0 | 253.3 | 9.12 | 2440.5 | 1.000 | 1.000 | 1.000 | 1.000 |
| bf16+dp2 | 32 | 144.7 | 161.8 | 5.45 | 3969.2 | 0.883 | 1.000 | 1.000 | 1.000 |
| fp8+dp2 | 32 | 146.6 | 159.7 | 6.26 | 3633.2 | 0.569 | 0.281 | 1.000 | 1.000 |
| int4+dp2 | 32 | 130.6 | 148.1 | 7.72 | 3226.1 | 1.000 | 1.000 | 1.000 | 1.000 |

Every request completed in every cell (`completed == issued`, 9 configs × 3
concurrencies).

- **The server matches offline generation exactly.** `offline` — the served
  completion against `LLM.generate` on the same prompt at `temperature=0` — is
  **1.000 in all 9 configs**. Continuous batching does not change the answer.
- **TTFT p99 tracks the mean within 15% at concurrency 32 in every cell**, now
  including int4+tp1. Pre-fix that row was the exception at 30% (391.7 against
  302.1) and the slowest cell in the table at 1720 TPS; the `w4a16` tile fix took it
  to 157.7/143.6 and 2585 TPS, i.e. the tail was the decode kernel failing to absorb
  a full batch, not a scheduling artefact. int4+dp2 likewise went 2654.9 → 3226.1.
- **int4+tp2 is the one cell the re-measurement made *worse*** (2676.4 → 2440.5 TPS,
  TTFT 124.4 → 232.0) and it is reported as measured. TPOT improved (9.79 → 9.12), so
  the steady state is faster and the regression is in first-token latency, where two
  ranks admitting a 32-request wave is the least repeatable thing in this table —
  pre-fix, tp2 also beat tp1 by 2.4× on TTFT, an ordering nothing in the model
  explains. Treat single-run TTFT under concurrency 32 as ±50% here; TPOT and TPS are
  the columns that reproduced.
- **DP is the throughput axis online too**: bf16+dp2 at 3969 TPS is 1.26× bf16+tp1,
  and at concurrency 32 the three dp2 configs land in a 131-147 ms TTFT band
  regardless of scheme — the router, not the model, is setting first-token latency
  there.

### A negative result: batch invariance does not hold

`batch` is the served completion at concurrency N against the same prompt served
alone; `in-wave dup` is two identical prompts inside one wave against each other.
Both fall below 1.000 — as low as 0.546 (fp8+tp1) and 0.281 (fp8+dp2). Neither is a
bug, and the evidence for that is the last column.

`dup batch` submits the same prompt 32× in **one** `engine.generate` call: all
copies are queued before the first step, all the same length, so they necessarily
share every batch. It is **1.000 in all 9 configs**. Concurrent requests therefore
cannot see each other's state — if they could, the copies would diverge here too.

What remains is batch-size-dependent arithmetic: a GEMM picks its tile from M, a
captured graph pads to a bucket, a split-K reduction changes summation order. A
1e-3 logit shift is enough to flip a bf16 `argmax` tie, and on a reasoning
checkpoint one flipped token rewrites the rest of the completion — which is why the
rates are 0.55-0.89 rather than 0.999. An earlier draft of this benchmark treated
`in-wave dup < 1.000` as proof of a state leak; that was wrong, because the
scheduler admits from `_waiting` under `max_num_seqs` and `max_num_batched_tokens`
and HTTP arrivals are asynchronous, so two copies in one wave are not guaranteed to
share a batch at all. The duplicate-batch control group is what closed it.

## 4. fp8 KV cache

[`kv_fp8_error_qwen3-4b_20260901.json`](kv_fp8_error_qwen3-4b_20260901.json).
Thresholds set before measuring: any layer with `amax > 448`, **or** a token match
rate below 0.98, triggers calibration.

One gate fired, and the follow-up says calibration is not the fix:

| Probe | Result |
|---|---|
| clipping, 36 layers, 47.1M values | `max_amax` **294.0** (`layers.0...k`), 0 values clipped — gate **not** tripped |
| greedy agreement, 4 prompts × 128 tokens | **0.316** (162/512), first divergence at token 11-74 — gate **tripped** |
| control (`auto` vs `auto`, same seed) | **1.000** — the harness itself is deterministic, so the divergence is the dtype |
| subnormal fraction | up to **0.567** (`layers.0...v`) |
| oracle per-tensor headroom | mean **1.030×**, best layer 1.185× |
| GSM8K, 500 questions | 0.192 → 0.164, Δ **−0.028** against an unpaired stderr of **0.024** |

`scale=1.0` is not clipping — nothing reaches 448, and the *oracle* scale (the best a
perfect per-tensor calibration could pick) is only 1.03× away on average. So
calibration has almost nothing to recover: the error is e4m3's 3-bit mantissa spread
over every cached token, and 57% of the values in the worst layer land in the
subnormal range where the format's relative precision is worse still. **Phase 4b was
therefore not executed**, and that is a measured decision, not a skipped step.

The honest cost, on two independent references: 0.316 greedy agreement on the KV
script's own probe set, and 0.703 golden prefix agreement on the offline matrix's
longer prompts (row G, 1.000 → 0.718 match / 0.703 prefix). The benefit is 2.0× KV
capacity (447,830 → 895,692 tokens). GSM8K cannot resolve the difference at n=500 —
−2.8 points against a ±2.4-point stderr — so the task-accuracy question stays open
and is recorded as open. fp8 KV is a capacity/accuracy trade with those numbers
attached; it stays off by default.

## 5. Failures and non-results, recorded rather than dropped

| Item | Status |
|---|---|
| `M_MOE_INT4` (Qwen3-30B-A3B Int4 W4A16) | `ValueError: unsupported quant_method 'compressed-tensors'` at `quantization/__init__.py:105`. The checkpoint uses the compressed-tensors serialisation, which has no reader here; adding one was out of scope. Excluded from every matrix, recorded in Phase 0. |
| fp4 tensor-core (A4) GEMM | Not attempted. sm90 has no fp4 MMA and Triton has no fp4 dtype. NVFP4 is weight-only: it buys bytes, not arithmetic. |
| NVFP4 MoE experts | Not implemented; `NVFP4Config.get_quant_method` raises on a MoE layer rather than silently falling back. |
| `deepgemm/fp8_gemm_nt`, flashinfer rows | Still `GoldenRecord(verified=False)`. Neither library is installed; the rows are registered and filtered out, visible in `explain()`. |
| `tests/kernels/test_w4a16_accuracy.py::[w4a16_problem2]` | **Was** a failure on clean `HEAD`; fixed here, in the test rather than the kernel. The case is M16/N1024/K2048, where the largest output reaches `abs(ref) = 139.8` and one fp16 ULP is already 0.125 — so the flat `max_diff < 0.1` bound demanded better-than-representable accuracy. The measured `max_diff` is exactly 0.1250, i.e. one rounding step, and it is *identical* under the old and new tiles (checked by forcing both configs), so it was never a tile or kernel defect. The bound is now `max(0.1, 2 * abs(ref).max() * 2**-10)`, which is 2 fp16 ULP at the output's own magnitude. All 4 shapes pass; the other 3 passed under both bounds. |
| nvfp4 TPOT at tp1+graph (13.66 ms) | The worst decode latency of any captured row, by 2× — int4+graph, its nearest format, is 6.97 ms. The 16-element block dequant is more work per byte than int4's, and the format saves no bytes over it. Nobody has swept nvfp4's tiles: `nvfp4_matmul` does not consult `ConfigStore`, so the `w4a16` fix of §1.1b has no counterpart here, and this number should be read as "the first tile that worked" rather than as the format's floor. |

## Reproducing

```bash
export LITE_LLAMA_MODELZOO=/mnt/otto-temp/modelzoo_with_full_weights

# kernel
LITE_LLAMA_AUTOTUNE=0 python benchmarks/kernels/bench_fused_moe.py --json out.json
python benchmarks/kernels/bench_fused_moe.py --tune          # writes ConfigStore
LITE_LLAMA_AUTOTUNE=0 python benchmarks/kernels/bench_quant_gemm.py --json out.json
python benchmarks/kernels/bench_quant_gemm.py --tune       # w4a16 only; see §1.1b

# offline
python benchmarks/bench_quant.py --model-dir $LITE_LLAMA_MODELZOO/Qwen3/Qwen3-4B-Thinking-2507 \
    --schemes fp8 int4 --tp 1 2 --engine continuous --cuda-graph --no-cuda-graph --skip-hf

# online
python benchmarks/bench_serving.py --model-dir $LITE_LLAMA_MODELZOO/Qwen3/Qwen3-4B-Thinking-2507 \
    --schemes bf16 fp8 int4 --tp 1 2 --concurrency 1 8 32 --max-tokens 64 --max-seq-len 1024

# kv fp8 error
python scripts/quant_kv_error.py --model-dir $LITE_LLAMA_MODELZOO/Qwen3/Qwen3-4B-Thinking-2507
```
