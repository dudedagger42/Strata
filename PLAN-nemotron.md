# PLAN: Nemotron 3.5 Bolt 74B-A9B (`nemotron_h_moe`) as a Strata family

Branch `nemotron-bolt`, private fork. Target: Windows 11, RTX 4070 Laptop 8 GB, 64 GB RAM, i9-13950HX (AVX2), CUDA 12.8.
Goal: beat llama.cpp (`--n-cpu-moe` / `-ot exps=CPU`) on decode and prefill tok/s with identical greedy tokens.

Status of every statement below: read from the GGUF dumps and from the source. Nothing was built or run.
Speeds are estimates from byte counts, marked "est."; they are not measurements.

## 0. Findings that change the plan

1. **The vendored llama.cpp cannot load this file (static analysis, unconfirmed).** `llama-model.cpp:1263` reads
   `expert_count` as one scalar (496) and `nemotron-h.cpp` creates `ffn_gate_inp {n_embd, n_expert}`;
   `check_tensor_dims` throws on the 240/304/368/432-expert layers. Use the vendored code for the math only. The
   runnable oracle is LM Studio's llama-server 2.62.1. Our loader takes E per layer from the tensor shape.
2. **Vendored ggml has no CUDA backend** (`strata-ggml/ggml/src` has `ggml-cpu` only). The CPU `ssm_conv` / `ssm_scan`
   in `ggml-cpu/ops.cpp:9703, 9773` are the only Mamba2 reference.
3. **Split 3 is not an MTP block.** Split 2 has 39 of 40 `ffn_up_exps` and `ffn_up_shexp`; the two missing tensors are
   `blk.87.ffn_up_exps.weight` [1024, 2560, 368] and `blk.87.ffn_up_shexp.weight`. Q5_K + Q8_0 would be 686.6 MB,
   which matches the 686 MB file. No `nextn` keys in the metadata. Confirm with `gguf_dump.py` when the download ends.
   The loader finds tensors by name across shards, so either outcome needs no special case; `nextn.*` tensors or a
   block past 88 are skipped (as llama.cpp does without `--mtp`).
4. **All dense weights do not fit in 8 GB.** Dense is 6676 MiB before state, KV, scratch and the CUDA context
   (section 3). The output head goes to the CPU by default.
5. **Strata's CPU expert pool cannot be reused as is.** `ExpertPool` jobs call `native_gu_rows` (SiLU(gate) * up).
   Nemotron experts are up-only with ReLU^2. The row kernels underneath are reusable; the pool is new code.
6. **CPU kernel coverage:** multi-token rows exist for Q4_K (`kq256_rows`) and IQ4_XS (`iq256_rows`). Q5_K and Q6_K
   have single-token AVX2 dots only (`strata::quant::dot_kernel`, `src/quant/x86/dot_avx2.cpp`). 31 of 40 layers have
   Q5_K up, so decode (nt = 1) is covered today and CPU prefill needs two new multi-token kernels.
7. **GPU kernel coverage is complete:** `native_mmvq` takes Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS.

## 1. GGUF layout and hparams

### Split loading
`strata::GgufModel::open(any_shard)` (`include/strata/artifact/gguf_reader.hpp:579`, `gguf_split.hpp`) already opens
all shards, validates `split.count / split.no / split.tensors.count`, refuses a missing shard, and `find(name, &shard)`
returns the tensor and its shard. Shard 1 has metadata and 0 tensors: supported (`meta()` is shard 0). No reader change.
`read_value` keeps numeric arrays up to 1024 items, so the 88-entry per-layer arrays arrive whole.

### hparams: new `include/strata/nemotron/hparams.hpp` (header-only, no GPU, unit-tested)
Do not use `core/geometry.hpp` (compiled for Flash-Next). Read from `meta()`:

| key (`nemotron_h_moe.` prefix) | value | use |
|---|---|---|
| `block_count` | 88 | layers |
| `embedding_length` | 4096 | n_embd |
| `feed_forward_length[88]` | 0 or 1536..2560 | > 0 means MoE layer |
| `attention.head_count_kv[88]` | 2 at 7,16,25,36,47,58,69,78 | > 0 means attention layer |
| `attention.head_count` | 32 | q heads |
| `attention.key_length / value_length` | 128 | head dim |
| `attention.layer_norm_rms_epsilon` | 1e-5 | every RMS norm |
| `expert_feed_forward_length[88]` | per layer | expert FFN width (check against tensor shape) |
| `expert_used_count[88]` | 9 on MoE layers | top-k |
| `expert_count` | 496 | max only; **per-layer E = `ffn_gate_inp.weight` shape[1]** |
| `expert_gating_func` | 2 | sigmoid; refuse anything else |
| `expert_weights_norm`, `expert_weights_scale` | true, 5.0 | routing weights |
| `expert_shared_feed_forward_length` | 5376 | shared expert |
| `moe_latent_size` | 1024 | expert input and output width |
| `ssm.inner_size / state_size / group_count / time_step_rank / conv_kernel` | 8192 / 128 / 8 / 128 / 4 | Mamba2 |
| `tokenizer.ggml.*`, `tokenizer.chat_template` | gpt2, pre `tekken`, eos 11, bos 1, add_bos false | server side |

Layer kind, as the reference decides it: recurrent if `head_count_kv == 0 && ff == 0`; attention if `ff == 0`; else MoE.
Counts: 40 Mamba2, 8 attention, 40 MoE. `rope.dimension_count = 128` is ignored (section 4). `f_attention_scale` is
absent, so the scale is 1/sqrt(128).

### Tensors per layer kind (shape as GGUF prints it: [n_in, n_out, (E)])
- All layers: `attn_norm.weight` [4096] F32. One pre-norm per block: `x += f(rms(x) * attn_norm)`.
- Mamba2: `ssm_in` [4096, 18560] Q6_K (24) or Q8_0 (16); `ssm_conv1d.weight` [4, 10240] F32, `.bias` [10240];
  `ssm_dt.bias` [128]; `ssm_a`, `ssm_d` [1, 128]; `ssm_norm.weight` [1024, 8]; `ssm_out` [8192, 4096] Q6_K (22) or Q8_0 (18).
- Attention: `attn_q` [4096, 4096] Q8_0, `attn_k`, `attn_v` [4096, 256] Q6_K, `attn_output` [4096, 4096] Q8_0. No bias.
- MoE: `ffn_gate_inp` [4096, E] F32; `exp_probs_b.bias` [E] F32; `ffn_latent_down` [4096, 1024] Q8_0;
  `ffn_latent_up` [1024, 4096] Q8_0; `ffn_up_exps` [1024, ff, E]; `ffn_down_exps` [ff, 1024, E];
  `ffn_up_shexp` [4096, 5376] (Q5_K 8, Q6_K 16, Q8_0 15, one in split 3); `ffn_down_shexp` [5376, 4096] (Q4_K 1, Q5_K 15, Q6_K 16, Q8_0 8).
- Global: `token_embd` and `output` [4096, 131072] Q8_0 (544 MiB each), `output_norm` [4096] F32.

Expert count per layer: 240 x 12, 304 x 6, 368 x 9, 432 x 10, 496 x 3 = 13,824 experts, 39.7 GiB.
Expert types: up Q5_K 31 (+ layer 87 assumed), Q6_K 4, Q4_K 2, IQ4_XS 2; down IQ4_XS 16, Q5_K 14, Q6_K 6, Q4_K 4.
One expert is 2.2 to 3.7 MiB. One token reads 9 x 40 experts = **1059 MiB**.

Expert `e` of a layer is a contiguous slice, no repack: up rows at `data + e * ff * row_bytes(type, 1024)` (ff rows of
1024 values, 4 K-blocks each); down rows at `data + e * 1024 * row_bytes(type, ff)` (1024 rows of ff values; every ff
is a multiple of 256). **Experts stay in the file mapping. No copy to anonymous memory** (a copy would double 39.7 GiB).

Loader type gate (refuse at load, plain words): matrices need `native_mmvq_supported`; vectors F32; expert types in
{12, 13, 14, 23}; `ff % 256 == 0`; `moe_latent_size % 256 == 0`.

## 2. Mamba2 layers

### Exact math, per token (from `mamba-base.cpp:153` and `ops.cpp:9773`)
```
h      = rms(x, eps 1e-5) * attn_norm
zxBCdt = W_in h                       # 18560 = z[0:8192] | xBC[8192:18432] | dt[18432:18560]
win    = [conv_state(3 x 10240), xBC] # conv_state holds the 3 previous RAW xBC rows
c[j]   = silu( sum_k win[k][j] * w[k + 4*j] + bias[j] )         # j < 10240, k < 4; float sum, k ascending
conv_state = last 3 rows of win
xs = c[0:8192] as [128 heads][64];  B = c[8192:9216] as [8][128];  C = c[9216:10240] as [8][128]
dt_h = softplus(dt[h] + dt_bias[h])   # softplus(v) = v if v > 20 else log(1 + exp(v)); no other clamp
dA_h = exp(dt_h * ssm_a[h])           # ssm_a used as stored
g    = h / 16                         # 128 heads over 8 groups
S[h][i][:] = S[h][i][:] * dA_h + B[g][:] * (xs[h][i] * dt_h)    # state [128][64][128] f32
y[h][i]    = dot(S[h][i][:], C[g][:]) + xs[h][i] * ssm_d[h]     # D skip
y = silu(z) * y                       # ggml_swiglu_split(z, y): silu on z
y = per group of 1024 channels: y / sqrt(mean(y^2) + 1e-5) * ssm_norm[c][g]
x += W_out y
```
State per layer: conv 3 x 10240 f32 (0.12 MiB), SSM 128 x 64 x 128 f32 (4.0 MiB). 40 layers: 165 MiB, on the GPU.

### Kernels (new file `engine/src/nemotron/nem_kernels.cuh`)
- `k_mamba_conv(xBC, conv_state, w, b, out, T)`: one thread per (t, channel); window rows before the chunk come from
  `conv_state`. Writes the new `conv_state` (last 3 raw rows) in the same launch. Serves decode (T = 1) and prompt.
- `k_mamba_dt(dt, dt_bias, a, dt_sp, dA, T)`: 128 x T softplus and exp.
- `k_mamba_scan(S, xs, B, C, dt_sp, dA, D, y, T)`: grid = 8192 rows (h, i), block = 128 threads (one per state
  element), loop t = 0..T-1 inside the block: update s in a register, block-reduce `s * C` to `y[t][h][i]`, add the D
  term. The scan is sequential in t and parallel over 8192 rows. Decode is T = 1.
  T = 512 is 537 M state updates per layer: small next to the 60 MiB `ssm_in` read.
- `k_mamba_gate_norm(z, y, w, out, T)`: silu(z) * y, grouped RMS (8 blocks of 1024 per token).
- Reuse: `k_rms` / `k_add_rms` pattern from `q36_engine.cu:172-225` (copied, not included: that file is one
  translation unit), `native_quantize_q8_1` + `native_mmvq` for `ssm_in` and `ssm_out` in decode,
  `strata/prefill/gemm.hpp` for T > 8.
- Not reusable: `fused_gdn.hpp` / `native_gdn.hpp` (gated DeltaNet, different recurrence).

Summation order in `k_mamba_scan` differs from ggml's AVX2 order (4 accumulators of 8). The reference build
(section 7, M1) uses ggml's order; the GPU kernel is checked against it at 1e-5 relative.

### Prompt cache
SSM state cannot be rolled back. v1 keeps two host-RAM snapshots (165 MiB each): end of the last prompt and end of
the last reply. A request that extends either resumes; anything else re-reads from token 0. Model on q36's
`ckpt_save / ckpt_resume` (`q36_engine.cu:2296-2329`).

## 3. Latent MoE, placement and the VRAM budget

### Math, per token (from `nemotron-h.cpp:285` and `llama-graph.cpp:1993`)
```
h = rms(x) * attn_norm
logits = W_gate_inp h                   # F32, E outputs
p      = sigmoid(logits)
sel    = top-9 of (p + exp_probs_b)     # bias for selection only
w_k    = p[sel_k] / max(sum_k p[sel_k], 6.103515625e-5) * 5.0
a      = W_latent_down h                # 1024
m      = sum_k w_k * W_down[sel_k] ( relu(W_up[sel_k] a)^2 )     # 1024, summed in selection order
s      = W_down_shexp ( relu(W_up_shexp h)^2 )                   # full width, no gate
x     += W_latent_up m + s
```
Open check in M1: `ggml_argsort_top_k` tie order, and that the reference sums experts in top-k order.

### Decode: experts on the CPU (decision)
Per MoE layer:
1. GPU, one launch group: rms + q8_1 image; router `k_gemv_f32` (copy of `q36_engine.cu:269`); new `k_route_sig9`
   (sigmoid, bias, top-9, normalise, x5; one warp, E <= 496; modelled on `k_route_group`, `q36_kernels.cuh:306`,
   which is hard-coded to 256/top-8); `native_mmvq` for `latent_down`. Async copy of {9 ids, 9 weights, 1024 floats}
   to pinned host memory.
2. GPU continues with the shared expert (up mmvq, new `k_relu_sq`, down mmvq) while the CPU runs the experts.
3. CPU: `nem_cpu_experts.cpp` (new). Quantise `a` once to Q8_K (`q8k_quant_avx2`, 1168 bytes). For each of 9 experts:
   up rows -> `relu^2` -> quantise h (ff values) to Q8_K -> down rows -> `m += w_k * out`. Rows come from
   `kq256_rows` (Q4_K), `iq256_rows` (IQ4_XS), `strata::quant::dot_kernel(Q5_K | Q6_K, Arith::Avx2)`. These are
   bit-exact with ggml-cpu's AVX2 `vec_dot`, which is what `-ot exps=CPU` runs in the baseline.
   Own small barrier pool (rows split over workers, each output row written by one thread, so the result does not
   depend on the thread count). Take core selection from `kernels/cpu/pool_plan.hpp` (8P + 16E -> 13 workers is its
   measured rule) and `--cpu-threads`. Do not modify `ExpertPool`: Flash-Next's default path must stay byte-identical.
4. Upload `m` (1024 floats), `native_mmvq` for `latent_up`, add both branches to x.

One host round trip per MoE layer, 40 per token. Spin on the copy event as `moe_hybrid_cpu` does (`q36_engine.cu:3306`).
Decode CUDA graphs are captured per segment between CPU hops (41 segments); eager first, graphs in M3.

Rejected for decode: streaming experts to the GPU. 1059 MiB per token over PCIe (est. 8-12 GB/s) is 90+ ms.
No VRAM is left for a hot-expert cache (q36's slot tier); revisit only if the fit function finds > 1 GiB spare.

### Prefill: two paths (decision)
- `--prefill-experts cpu`: same pool, entries grouped per expert, up to 8 tokens per weight read. Needs new
  `q5k256_rows` and `q6k256_rows` in `src/kernels/cpu/kq_avx2.cpp` (same pattern as `q4k_dot`, parity test against
  `strata::quant::dot_ref`). Exact against the baseline. Default until the stream path passes the token gate.
  Also always used for chunks of <= 32 tokens (q36's `hybrid_t` convention).
- `--prefill-experts gpu` (stream): chunk T = 256 or 512. Route the chunk on the GPU, group entries per expert
  (`k_route_count` / `k_route_scan`, `q36_engine.cu:893-897`), then feed experts in waves through a pinned staging
  buffer (memcpy from the mapping -> `cudaMemcpyAsync` -> `native_mmvq` per expert over its token columns, up then
  `k_relu_sq` then down). Wave planning: see `core/moe_wave_plan.hpp` and q36's `stage[2]` double buffer. A full chunk
  touches most of 39.7 GiB, so est. 3.5-5 s per chunk: about 100-145 tok/s at T = 512, 50-70 at T = 256.
  Activations are q8_1 here and Q8_K on the CPU, so hidden states differ at quantisation-noise level from the
  baseline. It becomes the default only if section 6's gate passes with it.
  Later optimisation: adapt `prefill/moe_mmq.hpp` to up-only experts instead of per-expert `native_mmvq`.

### VRAM budget (MiB; computed from tensor shapes and block sizes)
| item | size | where |
|---|---|---|
| Mamba2 x 40: `ssm_in` 2660, `ssm_out` 1190, F32 vectors 10 | 3860 | GPU |
| MoE dense x 40: shared up 748, shared down 683, latent down + up 340 | 1771 | GPU |
| Routers (F32, 13,824 x 4096) + selection bias | 216 | GPU |
| Attention x 8 (q, o 17.0 each; k, v 0.8 each) | 285 | GPU |
| SSM state 160 + conv state 5 | 165 | GPU |
| KV fp16, 8 layers x 512 values: 8 KiB per token | 64 at 8k, 256 at 32k | GPU |
| Decode scratch (activations, q8_1 images, pinned mirrors) | 32 | GPU |
| CUDA context + cuBLAS (measure at M2) | ~350 | GPU |
| **Core, 8k context** | **6743** | |
| Prefill GEMM image (largest matrix as f16: 145) + chunk activations | ~210 | GPU, optional |
| Expert staging 2 x 192 | 384 | GPU, stream prefill only |
| Output head Q8_0 | 544 | **CPU by default** |
| `token_embd` Q8_0 | 0 | CPU: dequantise one row per token |

Core + prefill extras = 7337 MiB. With the head on the GPU the core alone is 7287 MiB.

Fit function `include/strata/nemotron/fit.hpp` (header-only, unit-tested like `qwen35/hybrid_fit.hpp`): take
`cudaMemGetInfo` free, subtract a 300 MiB reserve, then apply in this order until it fits:
1. Output head on the CPU (-544). CPU Q8_0 rows over all cores plus argmax; est. +15-18 ms per token.
2. Stream prefill with one 128 MiB stage, or `--prefill-experts cpu` (-256 to -384).
3. Prefill dense through `native_mmvq` in 8-column batches instead of the f16 GEMM image (-145).
4. Context 8k -> 4k (-32).
5. Routers on the CPU (-216, +216 MiB read per token).
6. `ssm_in` of the Q8_0 Mamba layers on the CPU, last layer first (-77 each, one more round trip each).
Still over: refuse with the numbers. Never run into WDDM shared memory.
If free VRAM is >= 7.6 GiB the head stays on the GPU and prefill experts run on the CPU (decode-first profile).
`--head gpu|cpu|auto` overrides. Print the chosen placement in one INFO line.

Every dense byte moved to the CPU costs about 8x more time than on the GPU (est. 30 vs 256 GB/s), so the order
above is by fewest round trips, then fewest bytes.

## 4. Attention layers

Reference: `build_attention_layer` (`nemotron-h.cpp:266`), `llama_model_rope_type` returns `LLAMA_ROPE_TYPE_NONE`
for `NEMOTRON_H_MOE` (`llama-model.cpp:2922`), and `build_attn` is called with no positions. **No RoPE, no q/k norm,
no gate, no bias.**
```
h = rms(x) * attn_norm;  q = W_q h (32 x 128);  k = W_k h (2 x 128);  v = W_v h (2 x 128)
q head j reads KV head j / 16;  att = softmax(q . k / sqrt(128)) v;  x += W_o concat(att)
```
KV cache: fp16 by default (llama.cpp's default K/V type is F16, so this matches the baseline), 8 KiB per token.
Reuse q36's design: split-K over 256-cell chunks with a log-sum-exp merge, causal batch for the prompt, one query for
decode, and its `--kv` plumbing (`kv_tok_bytes`, `k_kv_f16`). The q36 kernels cannot be called as they are: they sit
in the anonymous namespace of `q36_engine.cu`, assume head dim 256 (`load()` refuses anything else; `fwht256_row`),
and carry RoPE and the gate. Copy the fp32 and fp16 variants into `nem_kernels.cuh` with HD = 128, HQ = 32, HKV = 2
as template parameters and delete the RoPE, norm and gate arguments. Skip fp8 and q4 KV in v1.

## 5. Build, family registration, serve protocol, terminal

### Engine files (all new unless marked)
- `engine/include/strata/artifact/model_family.hpp` (edit): `ModelFamily::NemotronHMoe`, arch `"nemotron_h_moe"`,
  title "NVIDIA Nemotron 3.5 Bolt". `model_family_runs` stays false for it (own engine path, as Qwen3.6).
- `engine/include/strata/nemotron/{hparams.hpp, fit.hpp, engine.hpp}`; `engine.hpp` declares `strata::nemotron::run`.
- `engine/src/nemotron/nem_engine.cu` (loader, Engine, decode, prefill, serve), `nem_kernels.cuh`,
  `nem_cpu_experts.{h,cpp}`, `nem_main.cpp` (5 lines, as `q36_main.cpp`), `nem_ref.cpp` (CPU reference, M1),
  `nem_metadata_test.cpp`, `nem_fit_test.cpp`, `nem_cpu_experts_test.cpp`.
- `engine/cmake/strata_gpu_engine.cmake` (edit, after the q36 block at line 571), behind
  `option(STRATA_NEMOTRON "..." OFF)` so default builds are unchanged:
  ```
  add_library(strata_nemotron STATIC src/nemotron/nem_engine.cu src/nemotron/nem_cpu_experts.cpp)
  target_link_libraries(strata_nemotron PRIVATE strata_prefill strata_kernels strata_kernels_cpu strata_warnings <gpu runtime>)
  add_executable(strata-nemotron src/nemotron/nem_main.cpp);  strata_utf8_manifest(strata-nemotron)
  target_link_libraries(strata PRIVATE strata_nemotron);  target_compile_definitions(strata PRIVATE STRATA_HAVE_NEMOTRON=1)
  ```
  `strata-nemref` (no CUDA) and the three tests go in `engine/CMakeLists.txt` beside `q36_metadata_test` (line 420).
  CUDA only in v1; HIP and SYCL untouched.
- `engine/src/program/generate.cpp:3808` (edit): next to the Qwen35Moe switch, `#if STRATA_HAVE_NEMOTRON` ->
  `return strata::nemotron::run(argc, argv)`.

### Engine CLI and serve protocol
Same flags as q36 where they apply: `--model | --native <any shard>`, `--serve`, `--max-context`, `--chunk`,
`--tokens IDS --n N --force IDS --dump-full-logits F`, `--bench-pp N`, `--bench-tg N`, `--kv fp16|fp32`,
`--cpu-threads`, `--no-graph`; new: `--head`, `--prefill-experts`, `--dump-layers F`. Unknown flags in `--serve` are
accepted and ignored, as q36 does (`q36_engine.cu:5405`), so `server.py` can pass its usual set.
Protocol copied from `q36_engine.cu:4990-5330`: `READY <ctx> stop`; `GEN <max_new> [keys] <ids>`; `PP <done> <fresh>
<ms> <tok/s>`; `T <id>`; `DONE ...` with the same field count; `STOP` from a reader thread; `INFO` lines. Sampling
through `strata/qwen35/sampling.hpp`. Not in v1: MTP, batching (`BGEN`), vision (`GENI`), peer device.

### Server (`terminal/serve`)
- New `nemotron_kind.py` modelled on `hy4_kind.py` / `inkling_kind.py`: `ARCH = "nemotron_h_moe"`,
  `EXE_PREFIX = "strata-nemotron"`, `is_nemotron(cfg)`, `engine_args(cfg, exe)`, `end_ids` (eos 11 plus the
  template's end tokens), `refuse` (logprobs, logit_bias, images in v1).
- `server.py` hooks at the same places hy4 and inkling have them: engine kind flags (line 764), `engine_args`
  dispatch (3540), stop ids and literal tags (4141), catalog names (4391), tokenizer bootstrap (8540).
- Tokenizer: `terminal/tools/strata_tokenizer.py` gets a `"tekken"` kind in `pre_tokenizer_kind` with the regex of
  `LLAMA_VOCAB_PRE_TYPE_TEKKEN` (`llama-vocab.cpp`) and `ignore_merges = True` (as the o200k kind). `add_bos` comes
  from the GGUF (false). The tokenizer folder (`vocab.json`, `merges.txt`, `token_type.json`, `tokenizer.json` with
  `pre` and `special_ids`, `chat_template.jinja`) is written from the GGUF by `Tokenizer.from_gguf` on first start.
- Chat template: the GGUF's Jinja uses macros. Check that the server's `ChatTemplate` renders it; compare rendered
  ids with llama-server's `/apply-template` + `/tokenize` on three conversations.

### Terminal (`terminal/strata_cli`, `terminal/strata_control`)
- `strata_cli/launch/adopt.py`: add `"nemotron-bolt"` to `FAMILIES`; `model_files` -> `find_shards`; a `plan()`
  branch that writes `{"arch": "nemotron_h_moe", "exe": <strata-nemotron>, "args": ["--native", shard1,
  "--max-context", ctx, "--kv", "fp16"], "tokenizer": <data_dir>/packs/nemotron-bolt-ud-q4_k_xl/tokenizer}`.
- `strata_control/models.py`: accept the arch in `plan_import` (line 979) through a new `nemotron_types.py` that
  mirrors the loader's type gate (as `q36_types.py` does, with a test keeping the two equal).
- `strata_control/nemotron_fit.py`: refusal words below 8 GB VRAM or 64 GB RAM; the 3-shard completeness check.
- `terminal/setup.py` catalog: family entry marked local files only (no download source).
- Start: `strata launch --family nemotron-bolt --model ud-q4_k_xl --model-path <folder with the 3 shards>
  --data-dir B:/Claude/strata-nemotron-data --context 8k --yes --no-tui`. The separate data dir keeps its config,
  tokenizer and logs away from the Qwen install. Until there is a packaged component, `exe` is the build's absolute
  path. Tests: `terminal/tests/terminal` gets plan and types tests; nothing needs a GPU.

## 6. Verification and benchmarks

### Oracle
`C:/Users/gsbul/.lmstudio/extensions/backends/llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.62.1/llama-server.exe`, run
from its own folder (it loads the DLLs beside it), with LM Studio closed: two 50 GB mappings do not fit in 64 GB.
Driver: `terminal/tools/nemotron/parity.py`, modelled on `terminal/tools/qwen35/parity.py`, `kl.py` and
`q36_lgcmp.py` (check `B:/Claude/strata-nemotron-tools/ref_llama.py` first; reuse it if it already drives the server).

### Correctness
1. 10 fixed prompts, checked in as token ids (`parity_prompts.ids`): from `/tokenize`, so tokenizer differences
   cannot hide in the engine test. Mix: 3 chat-templated, 2 code, 2 long (about 1500 tokens, crosses chunk
   boundaries), 1 non-English, 1 single token, 1 with tool tags.
2. Reference run: `/completion` with `prompt` as the id array, `n_predict 64`, `temperature 0`, `top_k 1`,
   `n_probs 10`, `cache_prompt false`. Save the ids and the top-10 logprobs of every step.
3. Ours: `strata-nemotron --tokens IDS --n 64 --dump-full-logits F`. The tool turns logits into log-softmax.
4. Gates:
   - A (M1, CPU reference vs llama-server `-ngl 0`): 640 of 640 tokens equal; first-token top-10 ids equal and
     |delta logprob| <= 1e-3.
   - B (engine, CPU experts, vs llama-server `-ngl <max> -ot exps=CPU`): 640 of 640 tokens equal; first-token top-1
     equal on all 10; top-10 |delta logprob| <= 0.05, reported per prompt.
   - C (stream prefill): same as B. If it fails, stream stays opt-in.
   - Any mismatch is reported with the reference's top-1 to top-2 gap at that step, then the run continues
     teacher-forced (`--force`) so later steps are still compared. A near-tie is reported, not waived.
5. Per-layer check for debugging: `--dump-layers` from `strata-nemref` and from the engine, compared by
   `compare_dumps.py` (relative error per layer, first layer above 1e-3 named).
6. Unit level, no model needed: Mamba step vs a scalar transcription of `ops.cpp:9773`; conv; grouped norm; top-9
   routing with ties; Q5_K / Q6_K multi-token rows vs `dot_ref`; `nem_cpu_experts` at 1 thread vs N threads (bitwise).

### Benchmark
- Same machine state for both: AC power, same context (8192), same prompts, model warm in the page cache (one
  discarded warm-up run each), the other engine not loaded. Record `nvidia-smi` dedicated memory; a run that spills
  into shared memory is void for either side.
- Baseline, tuned in its favour: best of `--n-cpu-moe 88 -ngl 99`; `-ot exps=CPU` with the highest `-ngl` that
  stays in dedicated VRAM; with and without `-ot "output\.weight=CPU"`; threads 8, 16, 24. Report the best.
- Decode: 256 new tokens after a 512-token prompt, 3 runs, median tok/s. Ours: `--bench-tg 256` and through the server.
- Prefill: 512, 2048 and 6000 token prompts, tok/s from llama-server's `timings.prompt_per_second`; ours
  `--bench-pp N` and the `PP` lines. Report CPU and stream prefill separately.
- Also report: time to first token for a 30-token prompt (short prompts take the CPU path), load time cold and warm,
  peak RAM and VRAM. State what was not run.

## 7. Order of work

| # | milestone | done when | est. |
|---|---|---|---|
| M0 | Split 3 check; family enum; `hparams.hpp` + test; `nemotron_types.py`; tekken tokenizer + test against `/tokenize` | tests pass without a GPU; ids equal on the 10 prompts | 1 d |
| M1 | `strata-nemref`: whole forward on the CPU, one token at a time, ggml-exact quant dots for every matrix, ggml's scan order | gate A | 2 d |
| M2 | `nem_engine.cu` decode: loader (dense read unbuffered and uploaded, experts mapped), Mamba, attention, MoE with `nem_cpu_experts`, head on CPU, `--tokens` | gate B; per-layer dumps within 1e-3 of M1 | 3-4 d |
| M3 | Decode speed: pool cores, pinned hops, shared expert overlapped, segment graphs, fit function; first baseline numbers | `--bench-tg` beats the tuned baseline, or the gap is explained by measured byte rates | 2 d |
| M4 | Prefill, CPU experts: chunked dense (GEMM), scan kernel, causal attention, Q5_K / Q6_K multi-token rows, snapshots | gate B on the long prompts at chunk 64, 256, 512 (same tokens for every chunk size) | 3 d |
| M5 | Stream prefill: waves, staging, fit tiers | gate C; `--bench-pp` vs baseline | 3 d |
| M6 | `--serve`, `nemotron_kind.py`, server hooks, chat template check, adopt / plan, `strata launch` from local files | a chat round trip through the server; `strata connect <app> --check` | 2 d |
| M7 | Full verification and benchmark report (measured numbers, machine named, what was not run) | report written | 1 d |

M1 is slow by design (est. 0.3-0.5 s per token, about 5-7 minutes for the 10 prompts) and is kept as the permanent
reference binary, as `strata-q36ref` is for Qwen3.6.

## 8. Risks and fallbacks

1. **VRAM fit.** Core is 6743 MiB against about 7000 usable; every number above is computed, not measured, and the
   CUDA context is a guess. Fallbacks are the fit tiers in section 3. Free win to check first: drive the panel from
   the Intel iGPU so the 4070 starts near empty. Set the NVIDIA "Prefer No Sysmem Fallback" policy for both engines.
2. **Decode margin may be small.** Both engines read the same 1059 MiB of experts per token on the CPU; ours adds
   544 MiB for the head unless it fits on the GPU. est. 80-90 ms per token (GPU dense about 25-35, experts about 35,
   head about 18), so 11-13 tok/s. The win has to come from finer placement than `-ngl` allows, fewer and cheaper
   CPU-GPU hops, the overlapped shared expert and core pinning. M3 measures CPU byte rate first; if the head costs
   more than it frees, switch to the decode-first profile (head on GPU, CPU prefill).
3. **RAM and the Windows page cache.** Experts 39.7 GiB mapped + head and embedding 1.1 GiB + staging; Windows and
   apps need the rest of 64 GB. Rules: no second copy of the experts; dense tensors read with unbuffered I/O so
   6.5 GiB of pages do not compete (`core/page_cache_drop.hpp`, `core/pinned.hpp`); `PrefetchVirtualMemory` at load;
   raise the working-set minimum so the mapping is not trimmed. If the cache thrashes (decode tok/s drops with disk
   activity), fall back to `VirtualLock` in 1 GiB steps, then to an anonymous copy of the hottest layers only.
   LM Studio must be closed during our runs and the reverse.
4. **Exactness across devices.** CPU experts use Q8_K activations (bit-exact with the baseline); GPU experts use
   q8_1. The stream prefill can flip a near-tie token. Fallback: CPU prefill stays the default and stream is opt-in.
5. **No runnable vendored reference** (finding 1). M1 replaces it. If LM Studio's runtime changes the model's
   handling between versions, pin 2.62.1 in the report.
6. **Tokenizer and template.** `tekken` is new to `strata_tokenizer.py`, and the template uses Jinja macros.
   Both are checked against llama-server in M0 and M6; engine gates use raw ids, so they do not depend on this.
7. **Chunk-size invariance.** The Mamba scan and the K/V cache must give the same tokens at chunk 1, 64, 256, 512.
   M4's gate checks it; a failure points at conv-state handover or the scan's first step.
8. **Routing ties.** Top-9 over sigmoid + bias in F32 on the GPU vs the baseline's GEMM order can flip the ninth
   expert at a tie. Detect with the per-layer dumps (selected ids per layer in `--dump-layers`); if it shows up in the
   gates, compute the router with the baseline's summation order before changing anything else.
