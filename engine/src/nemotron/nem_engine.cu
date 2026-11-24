// src/nemotron/nem_engine.cu - strata-nemotron: NVIDIA Nemotron-H MoE (GGUF architecture nemotron_h_moe, e.g. Nemotron 3.5
// Bolt 74B-A9B) on one GPU with the routed experts in RAM.
//
//   * every block is one of: Mamba2 mixer, attention (no RoPE, GQA), latent MoE; pre-norm (attn_norm) + residual;
//   * GPU: the dense matrices straight from the GGUF blocks (native MMVQ, llama.cpp's arithmetic), the Mamba2 conv and
//     SSM state, the attention KV (f16), the router (sigmoid + selection bias, top-k, normalised, scaled), the latent
//     down/up projections and the shared expert;
//   * CPU: the routed experts (nem_cpu.cpp), read in place from the mapped GGUF, overlapped with the shared expert;
//   * the prompt in chunks through the same driver, decode one token at a time.
//
// Modes: --serve (terminal/serve/server.py's engine protocol: READY / GEN / PP / T / DONE / STOP), --tokens IDS [--n N]
// [--top K] (reference interface: greedy ids and the first steps' top log-probs), --bench-pp N / --bench-tg N.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "nem_cpu.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace K = strata::kernels;
namespace N = strata::nemotron;
using Clock = std::chrono::steady_clock;

#define CK(x)                                                                                                    \
    do {                                                                                                         \
        cudaError_t e_ = (x);                                                                                    \
        if (e_ != cudaSuccess) {                                                                                 \
            std::fprintf(stderr, "strata-nemotron: CUDA %s at %s:%d (%s)\n", cudaGetErrorString(e_), __FILE__, \
                         __LINE__, #x);                                                                          \
            std::exit(1);                                                                                        \
        }                                                                                                        \
    } while (0)

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// ------------------------------------------------------------------------------------------------ kernels
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
template <int NT> __device__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = threadIdx.x < NT / 32 ? sh[threadIdx.x] : 0.f;
    if (w == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}
template <int NT> __device__ float block_max(float v, float* sh) {
    v = warp_max(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = threadIdx.x < NT / 32 ? sh[threadIdx.x] : -INFINITY;
    if (w == 0) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}

// y[t] = x[t] / sqrt(mean(x[t]^2) + eps) * w, rows of n (grid: rows, block 256); ld = row strides
__global__ void k_rms(const float* x, int ldx, const float* w, float* y, int ldy, int n, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = y + (size_t) blockIdx.x * ldy;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) s += xr[i] * xr[i];
    s = block_sum<256>(s, sh);
    const float sc = rsqrtf(s / n + eps);
    for (int i = threadIdx.x; i < n; i += 256) yr[i] = xr[i] * sc * w[i];
}

// grouped RMS norm: rows of n_group groups of gs, weight [n_group][gs] (grid: rows * n_group)
__global__ void k_group_rms(float* y, const float* w, int gs, int n_group, float eps) {
    __shared__ float sh[32];
    const int g = blockIdx.x % n_group;
    float* r = y + (size_t) blockIdx.x * gs;
    float s = 0.f;
    for (int i = threadIdx.x; i < gs; i += 256) s += r[i] * r[i];
    s = block_sum<256>(s, sh);
    const float sc = rsqrtf(s / gs + eps);
    for (int i = threadIdx.x; i < gs; i += 256) r[i] = r[i] * sc * w[g * gs + i];
}

__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
// x += a + b (the MoE block: routed + shared, then the residual)
__global__ void k_add2(float* x, const float* a, const float* b, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (a[i] + b[i]) + x[i];
}
__global__ void k_relu2(float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = fmaxf(x[i], 0.f);
        x[i] = v * v;
    }
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// Mamba2 causal conv over T tokens: zx rows of ld_zx floats, xBC at offset `off`, C channels; state [C][3] (oldest
// first); out [T][C] = silu(sum_j w[c][j] * window[j] + bias[c]).  One thread per channel.
__global__ void k_conv(const float* zx, int ld_zx, int off, int C, int T, float* state, const float* w, const float* bias,
                       float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s0 = state[c * 3 + 0], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3], b = bias[c];
    for (int t = 0; t < T; ++t) {
        const float cur = zx[(size_t) t * ld_zx + off + c];
        float s = 0.f;
        s += s0 * w0;
        s += s1 * w1;
        s += s2 * w2;
        s += cur * w3;
        out[(size_t) t * C + c] = silu(s + b);
        s0 = s1;
        s1 = s2;
        s2 = cur;
    }
    state[c * 3 + 0] = s0;
    state[c * 3 + 1] = s1;
    state[c * 3 + 2] = s2;
}

// Mamba2 SSM scan over T tokens.  One warp per (head, dim): lane l holds states [l*4, l*4+4) of d_state 128.
// xbc [T][d_inner + 2*G*S] (x | B | C), zx [T][ld_zx] (z at 0, dt at dt_off), state [H][P][S].
// y[t][h*P+i] = silu(z) * (sum_s state*C + x*D)
__global__ void __launch_bounds__(256) k_ssm(const float* xbc, int ld_xbc, const float* zx, int ld_zx, int dt_off,
                                             const float* dt_bias, const float* A, const float* D, float* state, float* y,
                                             int H, int P, int G, int T) {
    constexpr int S = 128;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= H * P) return;
    const int h = warp / P, i = warp % P;
    const int g = h / (H / G);
    const int d_inner = H * P;
    float* st = state + (size_t) warp * S + lane * 4;
    float4 s = *reinterpret_cast<float4*>(st);
    const float a = A[h], dd = D[h], db = dt_bias[h];
    for (int t = 0; t < T; ++t) {
        const float* row = xbc + (size_t) t * ld_xbc;
        const float x = row[h * P + i];
        const float sp = softplus(zx[(size_t) t * ld_zx + dt_off + h] + db);
        const float dA = expf(sp * a);
        const float xdt = x * sp;
        const float4 B = *reinterpret_cast<const float4*>(row + d_inner + g * S + lane * 4);
        const float4 C = *reinterpret_cast<const float4*>(row + d_inner + G * S + g * S + lane * 4);
        s.x = s.x * dA + B.x * xdt;
        s.y = s.y * dA + B.y * xdt;
        s.z = s.z * dA + B.z * xdt;
        s.w = s.w * dA + B.w * xdt;
        float acc = s.x * C.x + s.y * C.y + s.z * C.z + s.w * C.w;
        acc = warp_sum(acc);
        if (lane == 0) {
            const float z = zx[(size_t) t * ld_zx + h * P + i];
            const float v = acc + x * dd;
            y[(size_t) t * d_inner + h * P + i] = silu(z) * v;
        }
    }
    *reinterpret_cast<float4*>(st) = s;
}

// k, v rows [T][kvd] -> the cache (f16) at cells pos0 .. pos0+T-1
__global__ void k_kv_put(const float* k, const float* v, __half* kc, __half* vc, int kvd, int T, int pos0) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * kvd) return;
    kc[(int64_t) pos0 * kvd + i] = __float2half(k[i]);
    vc[(int64_t) pos0 * kvd + i] = __float2half(v[i]);
}

// causal attention for queries t0 .. t0+nq-1 of a batch whose first token is at cell pos0.  Block per (head, query),
// 128 threads (head dim 128).  sc: scratch [nq*H][ctx].
__global__ void __launch_bounds__(128) k_attn(const float* q, const __half* kc, const __half* vc, float* out, float* sc,
                                              int H, int HKV, int pos0, int t0, int ctx, float scale) {
    constexpr int HD = 128;
    __shared__ float qs[HD];
    __shared__ float sh[32];
    const int h = blockIdx.x % H, qi = blockIdx.x / H;
    const int t = t0 + qi;
    const int kvh = h / (H / HKV);
    const int kvd = HKV * HD;
    const int np = pos0 + t + 1;
    qs[threadIdx.x] = q[(size_t) t * H * HD + h * HD + threadIdx.x];
    __syncthreads();
    float* s = sc + (size_t) blockIdx.x * ctx;
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < np; p += 128) {
        const __half2* kr = reinterpret_cast<const __half2*>(kc + (size_t) p * kvd + kvh * HD);
        float d = 0.f;
#pragma unroll 8
        for (int j = 0; j < HD / 2; ++j) {
            const float2 kk = __half22float2(kr[j]);
            d += qs[2 * j] * kk.x + qs[2 * j + 1] * kk.y;
        }
        d *= scale;
        s[p] = d;
        mx = fmaxf(mx, d);
    }
    mx = block_max<128>(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < np; p += 128) {
        const float e = expf(s[p] - mx);
        s[p] = e;
        sum += e;
    }
    sum = block_sum<128>(sum, sh);
    __syncthreads();
    float acc = 0.f;
    for (int p = 0; p < np; ++p) acc += s[p] * __half2float(vc[(size_t) p * kvd + kvh * HD + threadIdx.x]);
    out[(size_t) t * H * HD + h * HD + threadIdx.x] = acc / sum;
}

// Long-context attention (the KV in host RAM, streamed to the GPU in pieces): flash-style, online softmax over keys held in
// dk/dv (cells k0 .. k0+n-1), for queries of a batch whose first token is at cell pos0.  Block per (4 queries, KV head,
// key split) = 64 rows of (query, head), 256 threads; split z takes keys [z*sub, (z+1)*sub) of the piece and writes its
// partial max / sum / unnormalized output to pm / pl / pa ([S][T*H], [S][T*H][128]).  k_attn_merge folds the splits into the
// running state st_* (carried across pieces) or, on the last piece, writes out = acc / l.
constexpr int AT_TQ = 4, AT_KT = 64, AT_HD = 128;
constexpr int at_smem(int G) {
    return AT_TQ * G * (AT_HD + 1) * 4 + 2 * AT_KT * (AT_HD + 2) * 2 + AT_TQ * G * (AT_KT + 1) * 4 + 3 * AT_TQ * G * 4;
}
template <int G>
__global__ void __launch_bounds__(256) k_attn_tile(const float* q, const __half* dk, const __half* dv, float* pm, float* pl,
                                                   float* pa, int H, int HKV, int pos0, int T, int k0, int n, int sub,
                                                   float scale) {
    constexpr int R = AT_TQ * G, QS = AT_HD + 1, KS = AT_HD + 2, SS = AT_KT + 1;
    constexpr int RPT = R / 2;   // rows per thread in the output update (2 halves of 128 threads)
    extern __shared__ __align__(16) unsigned char at_sm[];
    float* qs = reinterpret_cast<float*>(at_sm);
    __half* ks = reinterpret_cast<__half*>(qs + R * QS);
    __half* vs = ks + AT_KT * KS;
    float* S = reinterpret_cast<float*>(vs + AT_KT * KS);
    float* rm = S + R * SS;
    float* rl = rm + R;
    float* ra = rl + R;
    const int tid = threadIdx.x, g = blockIdx.y, z = blockIdx.z, kvd = HKV * AT_HD;
    const int tq0 = blockIdx.x * AT_TQ;
    auto row_t = [&](int r) { return tq0 + r / G; };
    auto row_h = [&](int r) { return g * G + r % G; };
    for (int i = tid; i < R * AT_HD; i += 256) {
        const int r = i / AT_HD, c = i % AT_HD, t = row_t(r);
        qs[r * QS + c] = t < T ? q[(size_t) t * H * AT_HD + row_h(r) * AT_HD + c] * scale : 0.f;
    }
    for (int i = tid; i < R; i += 256) rm[i] = -INFINITY, rl[i] = 0.f;
    const int d = tid & 127, half = tid >> 7;
    float acc[RPT];
#pragma unroll
    for (int rr = 0; rr < RPT; ++rr) acc[rr] = 0.f;
    // this split's keys, cut at the last key any query of the block may see
    const int kmax = pos0 + min(tq0 + AT_TQ, T) - 1;
    const int kb = z * sub;
    const int ke = min(min(n, kb + sub), kmax - k0 + 1);
    for (int kt = kb; kt < ke; kt += AT_KT) {
        __syncthreads();
        for (int i = tid; i < AT_KT * (AT_HD / 2); i += 256) {
            const int j = i / (AT_HD / 2), c2 = i % (AT_HD / 2);
            __half2 kv2 = __float2half2_rn(0.f), vv2 = kv2;
            if (kt + j < ke) {
                kv2 = reinterpret_cast<const __half2*>(dk + (size_t) (kt + j) * kvd + g * AT_HD)[c2];
                vv2 = reinterpret_cast<const __half2*>(dv + (size_t) (kt + j) * kvd + g * AT_HD)[c2];
            }
            reinterpret_cast<__half2*>(ks + j * KS)[c2] = kv2;
            reinterpret_cast<__half2*>(vs + j * KS)[c2] = vv2;
        }
        __syncthreads();
        // scores: row r = tid / 4, keys j = tid % 4 + 4 i
        {
            const int r = tid >> 2, j0 = tid & 3;
            const int t = row_t(r), qa = pos0 + t;
            float s[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) s[i] = 0.f;
            for (int c = 0; c < AT_HD; c += 2) {
                const float q0 = qs[r * QS + c], q1 = qs[r * QS + c + 1];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const float2 kf = __half22float2(reinterpret_cast<const __half2*>(ks + (j0 + 4 * i) * KS)[c / 2]);
                    s[i] += q0 * kf.x + q1 * kf.y;
                }
            }
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int j = j0 + 4 * i, key = k0 + kt + j;
                if (t >= T || kt + j >= ke || key > qa) s[i] = -INFINITY;
                mx = fmaxf(mx, s[i]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float mo = rm[r];
            const float mn = fmaxf(mo, mx);
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float p = mn == -INFINITY ? 0.f : __expf(s[i] - mn);
                S[r * SS + j0 + 4 * i] = p;
                sum += p;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            __syncwarp();
            if (j0 == 0) {
                const float a = mn == -INFINITY ? 1.f : __expf(mo - mn);
                ra[r] = a;
                rl[r] = rl[r] * a + sum;
                rm[r] = mn;
            }
        }
        __syncthreads();
#pragma unroll
        for (int rr = 0; rr < RPT; ++rr) acc[rr] *= ra[half * RPT + rr];
        for (int j = 0; j < AT_KT; ++j) {
            const float v = __half2float(vs[j * KS + d]);
#pragma unroll
            for (int rr = 0; rr < RPT; ++rr) acc[rr] += S[(half * RPT + rr) * SS + j] * v;
        }
    }
    __syncthreads();
    const size_t base = (size_t) z * T * H;
#pragma unroll
    for (int rr = 0; rr < RPT; ++rr) {
        const int r = half * RPT + rr, t = row_t(r);
        if (t < T) pa[(base + (size_t) t * H + row_h(r)) * AT_HD + d] = acc[rr];
    }
    for (int i = tid; i < R; i += 256) {
        const int t = row_t(i);
        if (t < T) pm[base + (size_t) t * H + row_h(i)] = rm[i], pl[base + (size_t) t * H + row_h(i)] = rl[i];
    }
}

// fold S split partials (and, unless first, the running state) of each (query, head) row; block per row, 128 threads
__global__ void __launch_bounds__(128) k_attn_merge(const float* pm, const float* pl, const float* pa, float* st_m, float* st_l,
                                                    float* st_acc, float* out, int rows, int S, bool first, bool last) {
    const int r = blockIdx.x, d = threadIdx.x;
    float M = first ? -INFINITY : st_m[r];
    for (int z = 0; z < S; ++z) M = fmaxf(M, pm[(size_t) z * rows + r]);
    float L = 0.f, A = 0.f;
    if (!first && M != -INFINITY) {
        const float w = __expf(st_m[r] - M);
        L = st_l[r] * w;
        A = st_acc[(size_t) r * AT_HD + d] * w;
    }
    for (int z = 0; z < S; ++z) {
        const float m = pm[(size_t) z * rows + r];
        if (m == -INFINITY) continue;
        const float w = __expf(m - M);
        L += pl[(size_t) z * rows + r] * w;
        A += pa[((size_t) z * rows + r) * AT_HD + d] * w;
    }
    if (last) {
        out[(size_t) r * AT_HD + d] = A / L;
    } else {
        __syncthreads();
        st_acc[(size_t) r * AT_HD + d] = A;
        if (d == 0) st_m[r] = M, st_l[r] = L;
    }
}

// The host-KV attention of T queries over cells 0 .. pos+T-1: each piece copied in, split over enough blocks to fill the GPU,
// merged.  Scratch: pm/pl [cap_rows*H], pa [cap_rows*H][128] (cap_rows >= T), st_* the carried state [T*H](x128).
constexpr int AT_CAP_ROWS = 1024;
static void attn_hostkv_gpu(const float* q, float* out, const __half* hk, const __half* hv, __half* dk, __half* dv, int piece,
                            float* pm, float* pl, float* pa, float* sm, float* sl, float* sa, int H, int HKV, int pos, int T,
                            bool sync_copies, cudaStream_t st) {
    const int kvd = HKV * AT_HD, nkeys = pos + T;
    const float scale = 1.0f / std::sqrt((float) AT_HD);
    const int bx = (T + AT_TQ - 1) / AT_TQ, blocks = bx * HKV;
    for (int k0 = 0; k0 < nkeys; k0 += piece) {
        const int n = std::min(piece, nkeys - k0);
        if (sync_copies) {
            CK(cudaMemcpy(dk, hk + (size_t) k0 * kvd, (size_t) n * kvd * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dv, hv + (size_t) k0 * kvd, (size_t) n * kvd * 2, cudaMemcpyHostToDevice));
        } else {
            CK(cudaMemcpyAsync(dk, hk + (size_t) k0 * kvd, (size_t) n * kvd * 2, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(dv, hv + (size_t) k0 * kvd, (size_t) n * kvd * 2, cudaMemcpyHostToDevice, st));
        }
        int S = std::max(1, std::min({(144 + blocks - 1) / blocks, (n + 255) / 256, AT_CAP_ROWS / T}));
        const int sub = ((n + S - 1) / S + AT_KT - 1) / AT_KT * AT_KT;
        S = (n + sub - 1) / sub;
        k_attn_tile<16><<<dim3((unsigned) bx, (unsigned) HKV, (unsigned) S), 256, at_smem(16), st>>>(q, dk, dv, pm, pl, pa, H, HKV,
                                                                                                  pos, T, k0, n, sub, scale);
        k_attn_merge<<<T * H, 128, 0, st>>>(pm, pl, pa, sm, sl, sa, out, T * H, S, k0 == 0, k0 + n >= nkeys);
    }
}

// logits[t][e] = W[e] . x[t] (f32 router; grid (E, T), block 128)
__global__ void k_gemv_f32(const float* W, const float* x, int ldx, float* y, int n_in, int n_out) {
    __shared__ float sh[32];
    const float* w = W + (size_t) blockIdx.x * n_in;
    const float* xr = x + (size_t) blockIdx.y * ldx;
    float s = 0.f;
    for (int i = threadIdx.x; i < n_in; i += 128) s += w[i] * xr[i];
    s = block_sum<128>(s, sh);
    if (threadIdx.x == 0) y[(size_t) blockIdx.y * n_out + blockIdx.x] = s;
}

// sigmoid routing: probs = sigmoid(logits), top-k of probs + bias, weights = probs of those, / sum (clamped), * scale.
// Block per token, 512 threads, E <= 1024.
__global__ void __launch_bounds__(512) k_route(const float* logits, const float* bias, int E, int k, bool norm,
                                               float wscale, int32_t* ids, float* wts) {
    __shared__ float sel[1024], pr[1024];
    __shared__ float bv[16];
    __shared__ int bi[16];
    __shared__ int pick[32];
    const float* lg = logits + (size_t) blockIdx.x * E;
    for (int e = threadIdx.x; e < E; e += 512) {
        const float p = 1.f / (1.f + expf(-lg[e]));
        pr[e] = p;
        sel[e] = p + bias[e];
    }
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        float best = -INFINITY;
        int bidx = 0x7fffffff;
        for (int e = threadIdx.x; e < E; e += 512)
            if (sel[e] > best || (sel[e] == best && e < bidx)) { best = sel[e]; bidx = e; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bidx, o);
            if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
        }
        if (threadIdx.x % 32 == 0) { bv[threadIdx.x / 32] = best; bi[threadIdx.x / 32] = bidx; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = bv[0];
            int id = bi[0];
            for (int w = 1; w < 16; ++w)
                if (bv[w] > b || (bv[w] == b && bi[w] < id)) { b = bv[w]; id = bi[w]; }
            pick[j] = id;
            sel[id] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int j = 0; j < k; ++j) sum += pr[pick[j]];
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int j = 0; j < k; ++j) {
            float w = pr[pick[j]];
            if (norm) w = w / sum;
            if (wscale != 0.f && wscale != 1.f) w = w * wscale;
            ids[(size_t) blockIdx.x * k + j] = pick[j];
            wts[(size_t) blockIdx.x * k + j] = w;
        }
    }
}

// two-stage argmax over n (stage 1: per block; stage 2: one block over the partials)
__global__ void k_argmax(const float* x, int n, float* pv, int32_t* pi) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (x[i] > best || (x[i] == best && i < idx)) { best = x[i]; idx = i; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    if (threadIdx.x % 32 == 0) { bv[threadIdx.x / 32] = best; bi[threadIdx.x / 32] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int) blockDim.x / 32; ++w)
            if (bv[w] > best || (bv[w] == best && bi[w] < idx)) { best = bv[w]; idx = bi[w]; }
        best = bv[0];
        idx = bi[0];
        for (int w = 1; w < (int) blockDim.x / 32; ++w)
            if (bv[w] > best || (bv[w] == best && bi[w] < idx)) { best = bv[w]; idx = bi[w]; }
        pv[blockIdx.x] = best;
        pi[blockIdx.x] = idx;
    }
}
__global__ void k_argmax2(const float* pv, const int32_t* pi, int n, int32_t* out) {
    if (threadIdx.x != 0) return;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = 0; i < n; ++i)
        if (pv[i] > best || (pv[i] == best && pi[i] < idx)) { best = pv[i]; idx = pi[i]; }
    *out = idx;
}

inline int nblk(int64_t n, int b = 256) { return (int) ((n + b - 1) / b); }

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;
    int type = -1;
    int n_in = 0, n_out = 0;
};

enum class Kind { Mamba, Attn, Moe };

struct Layer {
    Kind kind = Kind::Mamba;
    const float* norm = nullptr;
    // Mamba2
    Mat ssm_in, ssm_out;
    const float *conv_w = nullptr, *conv_b = nullptr, *dt_b = nullptr, *A = nullptr, *D = nullptr, *ssm_norm = nullptr;
    // attention
    Mat wq, wk, wv, wo;
    int n_head_kv = 0;
    // MoE
    const float *router = nullptr, *probs_b = nullptr;
    int n_expert = 0, k = 0, n_ff = 0;
    Mat lat_down, lat_up, sh_up, sh_down;
    N::ExpertLayer ex;
};

struct Model {
    std::shared_ptr<strata::GgufModel> file;
    int n_layer = 0, n_embd = 0, n_vocab = 0, n_head = 0, head_dim = 128, n_lat = 0, n_ff_sh = 0;
    int d_conv = 4, d_inner = 0, d_state = 0, n_group = 0, n_ssm_head = 0;
    float eps = 1e-5f, wscale = 1.f;
    bool wnorm = true;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    bool head_cpu = false;              // the head stays in RAM (not enough VRAM): head_host, computed on the CPU
    const uint8_t* head_host = nullptr;
    const uint8_t* embd = nullptr;   // host (mapped file)
    int embd_type = -1;
    size_t embd_row = 0;
    std::vector<int> eos;
    size_t gpu_bytes = 0;
    bool lock_experts = false;
};

bool mmvq_type_ok(int t) { return t == 8 || t == 12 || t == 13 || t == 14 || t == 23 || t == 11 || t == 2 || t == 6; }

void mmvq(const Mat& W, const void* xq, float* y, int ncols, cudaStream_t s) {
    switch (W.type) {
        case 8: K::native_q8_0_mmvq(W.d, xq, y, W.n_in, W.n_out, ncols, s); break;
        case 12: K::native_q4_k_mmvq(W.d, xq, y, W.n_in, W.n_out, ncols, s); break;
        case 13: K::native_q5_k_mmvq(W.d, xq, y, W.n_in, W.n_out, ncols, s); break;
        case 14: K::native_q6_k_mmvq(W.d, xq, y, W.n_in, W.n_out, ncols, s); break;
        case 23: K::native_iq4_xs_mmvq(W.d, xq, y, W.n_in, W.n_out, ncols, s); break;
        case 11: K::native_q3_k_mmvq(W.d, xq, y, W.n_in, W.n_out, ncols, s); break;
        case 2: K::native_q4_0_mmvq(W.d, xq, y, W.n_in, W.n_out, ncols, s); break;
        case 6: K::native_q5_0_mmvq(W.d, xq, y, W.n_in, W.n_out, ncols, s); break;
        default: std::fprintf(stderr, "strata-nemotron: no MMVQ for type %d\n", W.type); std::exit(1);
    }
}

const strata::MetaValue* meta(const strata::GgufModel& gm, const std::string& k) { return gm.meta().get(k); }

int64_t meta_layer_int(const strata::GgufModel& gm, const std::string& k, int l, int64_t dflt) {
    const strata::MetaValue* v = meta(gm, k);
    if (!v) return dflt;
    if (v->type == strata::MetaType::ARRAY) return l < (int) v->items.size() ? (int64_t) v->items[(size_t) l].num() : dflt;
    return (int64_t) v->num();
}

bool load(const std::string& path, Model& m, std::string& err, int head_mode /* -1 auto, 0 GPU, 1 CPU */,
          size_t reserve_mib, bool lock_experts) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    const strata::MetaValue* arch = meta(gm, "general.architecture");
    if (!arch || arch->s != "nemotron_h_moe") {
        err = "strata-nemotron runs nemotron_h_moe GGUF files only (this one is " + (arch ? arch->s : std::string("?")) + ")";
        return false;
    }
    const std::string p = "nemotron_h_moe.";
    auto geti = [&](const std::string& k, int64_t d) { const auto* v = meta(gm, p + k); return v ? (int64_t) v->num() : d; };
    m.n_layer = (int) geti("block_count", 0);
    m.n_embd = (int) geti("embedding_length", 0);
    m.n_head = (int) meta_layer_int(gm, p + "attention.head_count", 0, 0);
    m.head_dim = (int) geti("attention.key_length", 128);
    m.d_conv = (int) geti("ssm.conv_kernel", 4);
    m.d_inner = (int) geti("ssm.inner_size", 0);
    m.d_state = (int) geti("ssm.state_size", 0);
    m.n_group = (int) geti("ssm.group_count", 1);
    m.n_ssm_head = (int) geti("ssm.time_step_rank", 0);
    m.n_lat = (int) geti("moe_latent_size", 0);
    m.n_ff_sh = (int) geti("expert_shared_feed_forward_length", 0);
    if (const auto* v = meta(gm, p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const auto* v = meta(gm, p + "expert_weights_scale")) m.wscale = (float) v->num();
    if (const auto* v = meta(gm, p + "expert_weights_norm")) m.wnorm = v->u != 0;
    if (const auto* v = meta(gm, p + "expert_gating_func"); v && (int) v->num() != 2) {
        err = "only sigmoid expert gating (expert_gating_func 2) is supported";
        return false;
    }
    if (m.head_dim != 128 || m.d_state != 128 || m.d_conv != 4 || m.n_lat <= 0 || m.n_ssm_head <= 0 ||
        m.d_inner % m.n_ssm_head != 0 || m.n_ssm_head % m.n_group != 0) {
        err = "a geometry this engine's kernels do not take (head dim 128, SSM state 128, conv 4, a latent MoE)";
        return false;
    }
    auto find = [&](const std::string& name, size_t& s, bool required = true) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t && required) {
            std::fprintf(stderr, "strata-nemotron: missing tensor %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        if (t->type != 0) {
            std::fprintf(stderr, "strata-nemotron: %s is %s, expected F32\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const size_t nb = (size_t) t->elements() * 4;
        float* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        m.gpu_bytes += nb;
        return d;
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!mmvq_type_ok(w.type)) {
            std::fprintf(stderr, "strata-nemotron: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        m.gpu_bytes += nb;
        w.d = d;
        return w;
    };
    auto L = [](int l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) m.n_layer);
    for (int l = 0; l < m.n_layer; ++l) {
        Layer& y = m.layers[(size_t) l];
        const int64_t kv = meta_layer_int(gm, p + "attention.head_count_kv", l, 0);
        const int64_t ff = meta_layer_int(gm, p + "feed_forward_length", l, 0);
        y.kind = ff > 0 ? Kind::Moe : kv > 0 ? Kind::Attn : Kind::Mamba;
        y.norm = f32(L(l, "attn_norm.weight"));
        if (y.kind == Kind::Mamba) {
            y.ssm_in = mat(L(l, "ssm_in.weight"));
            y.ssm_out = mat(L(l, "ssm_out.weight"));
            y.conv_w = f32(L(l, "ssm_conv1d.weight"));
            y.conv_b = f32(L(l, "ssm_conv1d.bias"));
            y.dt_b = f32(L(l, "ssm_dt.bias"));
            y.A = f32(L(l, "ssm_a"));
            y.D = f32(L(l, "ssm_d"));
            y.ssm_norm = f32(L(l, "ssm_norm.weight"));
        } else if (y.kind == Kind::Attn) {
            y.n_head_kv = (int) kv;
            y.wq = mat(L(l, "attn_q.weight"));
            y.wk = mat(L(l, "attn_k.weight"));
            y.wv = mat(L(l, "attn_v.weight"));
            y.wo = mat(L(l, "attn_output.weight"));
        } else {
            y.k = (int) meta_layer_int(gm, p + "expert_used_count", l, 0);
            y.router = f32(L(l, "ffn_gate_inp.weight"));
            y.probs_b = f32(L(l, "exp_probs_b.bias"));
            y.lat_down = mat(L(l, "ffn_latent_down.weight"));
            y.lat_up = mat(L(l, "ffn_latent_up.weight"));
            y.sh_up = mat(L(l, "ffn_up_shexp.weight"));
            y.sh_down = mat(L(l, "ffn_down_shexp.weight"));
            size_t su = 0, sd = 0;
            const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
            const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
            y.n_ff = (int) tu->shape[1];
            y.n_expert = (int) tu->shape[2];
            if (y.k < 1 || y.k > 32 || y.n_expert > 1024 || (int) tu->shape[0] != m.n_lat || (int) td->shape[1] != m.n_lat) {
                err = L(l, "experts: a shape this engine does not take");
                return false;
            }
            y.ex.n_lat = m.n_lat;
            y.ex.n_ff = y.n_ff;
            y.ex.n_expert = y.n_expert;
            y.ex.up_type = (int) tu->type;
            y.ex.down_type = (int) td->type;
            y.ex.up = gm.shard(su).tensor_data(*tu);
            y.ex.down = gm.shard(sd).tensor_data(*td);
            if (!N::expert_layer_init(y.ex, err)) {
                err = L(l, "experts: ") + err;
                return false;
            }
        }
    }
    m.out_norm = f32("output_norm.weight");
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("output.weight", s);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        const bool fits = fr > nb + (reserve_mib << 20);
        m.head_cpu = head_mode == 1 || (head_mode == -1 && !fits);
        if (m.head_cpu) {
            m.head.type = (int) t->type;
            m.head.n_in = (int) t->shape[0];
            m.head.n_out = (int) t->shape[1];
            m.head_host = gm.shard(s).tensor_data(*t);
            std::fprintf(stderr, "strata-nemotron: the output head (%.0f MiB) stays in RAM: %.0f MiB of VRAM free, %zu MiB kept for "
                         "the state and scratch\n", nb / 1048576.0, fr / 1048576.0, reserve_mib);
        } else {
            m.head = mat("output.weight");
        }
    }
    m.n_vocab = m.head.n_out;
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        if (t->type != 8 && t->type != 0) {
            err = "token_embd must be Q8_0 or F32";
            return false;
        }
        m.embd_type = (int) t->type;
        m.embd_row = t->type == 8 ? (size_t) m.n_embd / 32 * 34 : (size_t) m.n_embd * 4;
        m.embd = gm.shard(s).tensor_data(*t);
    }
    m.n_head = m.n_head > 0 ? m.n_head : 32;
#if defined(_WIN32)
    // the experts into RAM now, in large sequential reads (PrefetchVirtualMemory), not page by page during the first tokens
    if (!std::getenv("STRATA_NEMOTRON_NO_PREFETCH")) {
        const auto p0 = Clock::now();
        std::vector<WIN32_MEMORY_RANGE_ENTRY> rs;
        size_t total = 0;
        for (const Layer& y : m.layers) {
            if (y.kind != Kind::Moe) continue;
            const size_t ub = (size_t) y.n_expert * y.n_ff * y.ex.up_row, db = (size_t) y.n_expert * m.n_lat * y.ex.down_row;
            rs.push_back({(PVOID) y.ex.up, ub});
            rs.push_back({(PVOID) y.ex.down, db});
            total += ub + db;
        }
        if (m.head_cpu) {   // the head read on the CPU every token: the same treatment
            const size_t hb = (size_t) m.head.n_out * (size_t) (m.head.n_in / 32) * 34;
            rs.push_back({(PVOID) m.head_host, hb});
            total += hb;
        }
        const BOOL ok = PrefetchVirtualMemory(GetCurrentProcess(), rs.size(), rs.data(), 0);
        // the call only queues the reads: touch one byte per page so the load returns with the experts resident
        volatile uint8_t sink = 0;
        for (const auto& r : rs)
            for (size_t o = 0; o < r.NumberOfBytes; o += 4096) sink ^= ((const volatile uint8_t*) r.VirtualAddress)[o];
        (void) sink;
        std::fprintf(stderr, "strata-nemotron: %.1f GiB of experts read into RAM in %.1f s (prefetch %s)\n",
                     total / 1073741824.0, ms_since(p0) / 1000.0, ok ? "on" : "refused");
    }
    m.lock_experts = lock_experts;   // the hot experts are locked after the warm-up (Engine::lock_hot)
#endif
    if (const auto* v = meta(gm, "tokenizer.ggml.eos_token_id")) m.eos.push_back((int) v->num());
    if (const auto* v = meta(gm, "tokenizer.ggml.tokens"); v && v->type == strata::MetaType::ARRAY)
        for (size_t i = 0; i < v->items.size() && i < 1000; ++i) {
            const std::string& s = v->items[i].s;
            if ((s == "<|im_end|>" || s == "</s>" || s == "<|endoftext|>" || s == "<|end_of_text|>") &&
                std::find(m.eos.begin(), m.eos.end(), (int) i) == m.eos.end())
                m.eos.push_back((int) i);
        }
    std::fprintf(stderr, "strata-nemotron: %s: %d layers, %.2f GiB on the GPU in %.1f s (experts stay in RAM)\n", path.c_str(),
                 m.n_layer, (double) m.gpu_bytes / (1 << 30), ms_since(t0) / 1000.0);
    return true;
}

// ------------------------------------------------------------------------------------------------ the engine
template <class T> T* dalloc(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}
template <class T> T* halloc(size_t n) {
    T* p = nullptr;
    CK(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocDefault));
    return p;
}

struct Engine {
    Model* m = nullptr;
    int ctx = 8192, chunk = 256, pos = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
    std::unique_ptr<N::SpinPool> pool;
    N::ExpertScratch xs;
    std::vector<uint8_t> head_act;
    // per-layer state
    std::vector<float*> conv_st, ssm_st;
    std::vector<__half*> kc, vc;
    // long context: the KV of each attention layer in pinned host RAM (hk/hv), streamed through dk/dv in pieces
    bool kv_host = false;
    int piece = 32768;
    std::vector<__half*> hk, hv;
    __half *dk = nullptr, *dv = nullptr, *nk = nullptr, *nv = nullptr;
    float *at_m = nullptr, *at_l = nullptr, *at_acc = nullptr, *at_pm = nullptr, *at_pl = nullptr, *at_pa = nullptr;
    float *h_q = nullptr, *h_y = nullptr;
    bool gpu_decode = false;   // --kv-host-gpu-decode: one-token attention on the GPU too (for comparison)
    double attn_ms = 0;        // one-token attention on the CPU
    // scratch (chunk rows)
    float *x = nullptr, *h = nullptr, *zx = nullptr, *xbc = nullptr, *y = nullptr, *o = nullptr, *q = nullptr, *kk = nullptr,
          *vv = nullptr, *sc = nullptr, *lat = nullptr, *lat_o = nullptr, *moe = nullptr, *sh = nullptr, *shh = nullptr,
          *logits = nullptr, *rl = nullptr, *wts = nullptr, *pv = nullptr;
    int32_t *ids = nullptr, *pi = nullptr, *amax = nullptr;
    uint8_t* xq = nullptr;
    int sc_q = 1;   // queries per attention launch (the score scratch holds sc_q * H rows of ctx)
    // pinned host
    float *h_x = nullptr, *h_lat = nullptr, *h_lat_o = nullptr, *h_wts = nullptr, *h_logits = nullptr;
    int32_t *h_ids = nullptr, *h_amax = nullptr;
    double cpu_ms = 0, head_ms = 0, wait_ms = 0;   // routed experts on the CPU, the head on the CPU, waiting for the GPU

    void init(Model& model, int ctx_, int chunk_, int threads, bool kv_host_, int pin = 0) {
        m = &model;
        ctx = ctx_;
        chunk = chunk_;
        kv_host = kv_host_;
        piece = std::min(ctx, piece);
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        pool = std::make_unique<N::SpinPool>(threads, pin);
        const Model& M = *m;
        const int C = M.d_inner + 2 * M.n_group * M.d_state;
        const int dproj = 2 * M.d_inner + 2 * M.n_group * M.d_state + M.n_ssm_head;
        size_t state_bytes = 0, kv_host_bytes = 0;
        for (const Layer& L : M.layers) {
            if (L.kind == Kind::Mamba) {
                conv_st.push_back(dalloc<float>((size_t) C * 3));
                ssm_st.push_back(dalloc<float>((size_t) M.n_ssm_head * (M.d_inner / M.n_ssm_head) * M.d_state));
                kc.push_back(nullptr);
                vc.push_back(nullptr);
                hk.push_back(nullptr);
                hv.push_back(nullptr);
                state_bytes += (size_t) C * 12 + (size_t) M.d_inner * M.d_state * 4;
            } else if (L.kind == Kind::Attn) {
                const size_t kvd = (size_t) L.n_head_kv * M.head_dim;
                conv_st.push_back(nullptr);
                ssm_st.push_back(nullptr);
                if (kv_host) {
                    kc.push_back(nullptr);
                    vc.push_back(nullptr);
                    hk.push_back(halloc<__half>((size_t) ctx * kvd));
                    hv.push_back(halloc<__half>((size_t) ctx * kvd));
                    std::memset(hk.back(), 0, (size_t) ctx * kvd * 2);
                    std::memset(hv.back(), 0, (size_t) ctx * kvd * 2);
                    kv_host_bytes += (size_t) ctx * kvd * 4;
                } else {
                    kc.push_back(dalloc<__half>((size_t) ctx * kvd));
                    vc.push_back(dalloc<__half>((size_t) ctx * kvd));
                    hk.push_back(nullptr);
                    hv.push_back(nullptr);
                    state_bytes += (size_t) ctx * kvd * 4;
                }
            } else {
                conv_st.push_back(nullptr);
                ssm_st.push_back(nullptr);
                kc.push_back(nullptr);
                vc.push_back(nullptr);
                hk.push_back(nullptr);
                hv.push_back(nullptr);
            }
        }
        int max_e = 0, max_k = 0;
        for (const Layer& L : M.layers) max_e = std::max(max_e, L.n_expert), max_k = std::max(max_k, L.k);
        const size_t T = (size_t) chunk;
        x = dalloc<float>(T * M.n_embd);
        h = dalloc<float>(T * M.n_embd);
        zx = dalloc<float>(T * dproj);
        xbc = dalloc<float>(T * C);
        y = dalloc<float>(T * M.d_inner);
        o = dalloc<float>(T * M.n_embd);
        q = dalloc<float>(T * M.n_head * M.head_dim);
        kk = dalloc<float>(T * 1024);
        vv = dalloc<float>(T * 1024);
        sc_q = (int) std::max<int64_t>(1, std::min<int64_t>((int64_t) T, (int64_t) (64ll << 20) / (4ll * M.n_head * ctx)));
        if (kv_host) {
            int kvd_max = 0;
            for (const Layer& L : M.layers)
                if (L.kind == Kind::Attn) kvd_max = std::max(kvd_max, L.n_head_kv * M.head_dim);
            dk = dalloc<__half>((size_t) piece * kvd_max);
            dv = dalloc<__half>((size_t) piece * kvd_max);
            nk = dalloc<__half>(T * kvd_max);
            nv = dalloc<__half>(T * kvd_max);
            at_m = dalloc<float>(T * M.n_head);
            at_l = dalloc<float>(T * M.n_head);
            at_acc = dalloc<float>(T * M.n_head * M.head_dim);
            at_pm = dalloc<float>((size_t) AT_CAP_ROWS * M.n_head);
            at_pl = dalloc<float>((size_t) AT_CAP_ROWS * M.n_head);
            at_pa = dalloc<float>((size_t) AT_CAP_ROWS * M.n_head * M.head_dim);
            h_q = halloc<float>((size_t) M.n_head * M.head_dim);
            h_y = halloc<float>((size_t) M.n_head * M.head_dim);
            CK(cudaFuncSetAttribute(k_attn_tile<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, at_smem(16)));
        } else {
            sc = dalloc<float>((size_t) sc_q * M.n_head * ctx);
        }
        lat = dalloc<float>(T * M.n_lat);
        lat_o = dalloc<float>(T * M.n_lat);
        moe = dalloc<float>(T * M.n_embd);
        sh = dalloc<float>(T * std::max(M.n_ff_sh, M.n_embd));
        shh = dalloc<float>(T * M.n_embd);
        logits = dalloc<float>((size_t) M.n_vocab);
        rl = dalloc<float>(T * std::max(1, max_e));
        wts = dalloc<float>(T * std::max(1, max_k));
        ids = dalloc<int32_t>(T * std::max(1, max_k));
        pv = dalloc<float>(256);
        pi = dalloc<int32_t>(256);
        amax = dalloc<int32_t>(1);
        const int max_in = std::max({M.n_embd, M.d_inner, M.n_ff_sh, M.n_lat, M.n_head * M.head_dim});
        xq = dalloc<uint8_t>(K::native_q8_1_bytes(max_in, 8) + 4096);
        h_x = halloc<float>(T * M.n_embd);
        h_lat = halloc<float>(T * M.n_lat);
        h_lat_o = halloc<float>(T * M.n_lat);
        h_wts = halloc<float>(T * std::max(1, max_k));
        h_ids = halloc<int32_t>(T * std::max(1, max_k));
        h_logits = halloc<float>((size_t) M.n_vocab);
        h_amax = halloc<int32_t>(1);
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        std::fprintf(stderr, "strata-nemotron: context %d, chunk %d, %d CPU threads; state + KV %.0f MiB on the GPU, KV %.0f MiB in RAM; %.0f MiB of VRAM free\n",
                     ctx, chunk, threads, state_bytes / 1048576.0, kv_host_bytes / 1048576.0, fr / 1048576.0);
    }

    void reset() {
        const Model& M = *m;
        const int C = M.d_inner + 2 * M.n_group * M.d_state;
        for (size_t l = 0; l < M.layers.size(); ++l)
            if (conv_st[l]) {
                CK(cudaMemsetAsync(conv_st[l], 0, (size_t) C * 3 * 4, st));
                CK(cudaMemsetAsync(ssm_st[l], 0, (size_t) M.d_inner * M.d_state * 4, st));
            }
        pos = 0;
    }

    // y[T][n_out] = W x[T][n_in] (columns in groups of 8)
    void matmul(const Mat& W, const float* xin, float* yout, int T) {
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int n = std::min(8, T - c0);
            K::native_quantize_q8_1(xin + (size_t) c0 * W.n_in, xq, W.n_in, n, st);
            mmvq(W, xq, yout + (size_t) c0 * W.n_out, n, st);
        }
    }

    void embed(const int32_t* toks, int T) {
        const Model& M = *m;
        for (int t = 0; t < T; ++t) {
            const uint8_t* r = M.embd + (size_t) toks[t] * M.embd_row;
            float* d = h_x + (size_t) t * M.n_embd;
            if (M.embd_type == 0) {
                std::memcpy(d, r, (size_t) M.n_embd * 4);
            } else {
                for (int b = 0; b < M.n_embd / 32; ++b) {
                    const uint8_t* qb = r + (size_t) b * 34;
                    uint16_t hb;
                    std::memcpy(&hb, qb, 2);
                    __half hh;
                    std::memcpy(&hh, &hb, 2);
                    const float dsc = __half2float(hh);
                    for (int j = 0; j < 32; ++j) d[b * 32 + j] = dsc * (float) (int8_t) qb[2 + j];
                }
            }
        }
        CK(cudaMemcpyAsync(x, h_x, (size_t) T * M.n_embd * 4, cudaMemcpyHostToDevice, st));
    }

    // attention with the KV in host RAM: the new rows appended there, then every key streamed through dk/dv in pieces
    void attn_host(size_t l, const Layer& L, int T) {
        const Model& M = *m;
        const int kvd = L.n_head_kv * M.head_dim;
        k_kv_put<<<nblk((int64_t) T * kvd), 256, 0, st>>>(kk, vv, nk, nv, kvd, T, 0);
        CK(cudaMemcpyAsync(hk[l] + (size_t) pos * kvd, nk, (size_t) T * kvd * 2, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(hv[l] + (size_t) pos * kvd, nv, (size_t) T * kvd * 2, cudaMemcpyDeviceToHost, st));
        if (T == 1 && !gpu_decode) {
            // one token: attention on the CPU straight from RAM (no PCIe copy of the whole cache)
            CK(cudaMemcpyAsync(h_q, q, (size_t) M.n_head * M.head_dim * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(ev, st));
            const auto w0 = Clock::now();
            CK(cudaEventSynchronize(ev));
            wait_ms += ms_since(w0);
            const auto c0 = Clock::now();
            N::attn_decode(reinterpret_cast<const uint16_t*>(hk[l]), reinterpret_cast<const uint16_t*>(hv[l]), pos + 1, M.n_head,
                           L.n_head_kv, h_q, h_y, 1.0f / std::sqrt((float) M.head_dim), *pool);
            attn_ms += ms_since(c0);
            CK(cudaMemcpyAsync(y, h_y, (size_t) M.n_head * M.head_dim * 4, cudaMemcpyHostToDevice, st));
            return;
        }
        attn_hostkv_gpu(q, y, hk[l], hv[l], dk, dv, piece, at_pm, at_pl, at_pa, at_m, at_l, at_acc, M.n_head, L.n_head_kv, pos, T,
                        false, st);
    }

    // one chunk of T tokens through every layer; the last token's logits into `logits`
    void forward(const int32_t* toks, int T) {
        const Model& M = *m;
        const int E = M.n_embd;
        embed(toks, T);
        const int C = M.d_inner + 2 * M.n_group * M.d_state;
        const int dproj = 2 * M.d_inner + 2 * M.n_group * M.d_state + M.n_ssm_head;
        for (size_t l = 0; l < M.layers.size(); ++l) {
            const Layer& L = M.layers[l];
            k_rms<<<T, 256, 0, st>>>(x, E, L.norm, h, E, E, M.eps);
            if (L.kind == Kind::Mamba) {
                matmul(L.ssm_in, h, zx, T);
                k_conv<<<nblk(C, 128), 128, 0, st>>>(zx, dproj, M.d_inner, C, T, conv_st[l], L.conv_w, L.conv_b, xbc);
                const int P = M.d_inner / M.n_ssm_head;
                k_ssm<<<nblk((int64_t) M.n_ssm_head * P * 32, 256), 256, 0, st>>>(
                    xbc, C, zx, dproj, 2 * M.d_inner + 2 * M.n_group * M.d_state, L.dt_b, L.A, L.D, ssm_st[l], y,
                    M.n_ssm_head, P, M.n_group, T);
                k_group_rms<<<T * M.n_group, 256, 0, st>>>(y, L.ssm_norm, M.d_inner / M.n_group, M.n_group, M.eps);
                matmul(L.ssm_out, y, o, T);
                k_add<<<nblk((int64_t) T * E), 256, 0, st>>>(x, o, (int64_t) T * E);
            } else if (L.kind == Kind::Attn) {
                const int kvd = L.n_head_kv * M.head_dim;
                matmul(L.wq, h, q, T);
                matmul(L.wk, h, kk, T);
                matmul(L.wv, h, vv, T);
                if (kv_host) {
                    attn_host(l, L, T);
                } else {
                    k_kv_put<<<nblk((int64_t) T * kvd), 256, 0, st>>>(kk, vv, kc[l], vc[l], kvd, T, pos);
                    for (int t0 = 0; t0 < T; t0 += sc_q) {
                        const int nq = std::min(sc_q, T - t0);
                        k_attn<<<nq * M.n_head, 128, 0, st>>>(q, kc[l], vc[l], y, sc, M.n_head, L.n_head_kv, pos, t0, ctx,
                                                              1.0f / std::sqrt((float) M.head_dim));
                    }
                }
                matmul(L.wo, y, o, T);
                k_add<<<nblk((int64_t) T * E), 256, 0, st>>>(x, o, (int64_t) T * E);
            } else {
                // routing + the latent input, to the host
                k_gemv_f32<<<dim3(L.n_expert, T), 128, 0, st>>>(L.router, h, E, rl, E, L.n_expert);
                k_route<<<T, 512, 0, st>>>(rl, L.probs_b, L.n_expert, L.k, M.wnorm, M.wscale, ids, wts);
                matmul(L.lat_down, h, lat, T);
                CK(cudaMemcpyAsync(h_lat, lat, (size_t) T * M.n_lat * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaMemcpyAsync(h_ids, ids, (size_t) T * L.k * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaMemcpyAsync(h_wts, wts, (size_t) T * L.k * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaEventRecord(ev, st));
                // the shared expert on the GPU while the CPU runs the routed ones
                matmul(L.sh_up, h, sh, T);
                k_relu2<<<nblk((int64_t) T * M.n_ff_sh), 256, 0, st>>>(sh, (int64_t) T * M.n_ff_sh);
                matmul(L.sh_down, sh, shh, T);
                const auto w0 = Clock::now();
                CK(cudaEventSynchronize(ev));
                wait_ms += ms_since(w0);
                const auto c0 = Clock::now();
                N::experts_forward(L.ex, *pool, xs, T, L.k, h_lat, h_ids, h_wts, h_lat_o);
                cpu_ms += ms_since(c0);
                if (M.lock_experts && !hot_locked) {
                    auto& cnt = hot[l];
                    if (cnt.empty()) cnt.assign((size_t) L.n_expert, 0);
                    for (int p = 0; p < T * L.k; ++p) cnt[(size_t) h_ids[p]]++;
                }
                CK(cudaMemcpyAsync(lat_o, h_lat_o, (size_t) T * M.n_lat * 4, cudaMemcpyHostToDevice, st));
                matmul(L.lat_up, lat_o, moe, T);
                k_add2<<<nblk((int64_t) T * E), 256, 0, st>>>(x, moe, shh, (int64_t) T * E);
            }
        }
        // the head on the last token only
        k_rms<<<1, 256, 0, st>>>(x + (size_t) (T - 1) * E, E, M.out_norm, h, E, E, M.eps);
        if (M.head_cpu) {
            CK(cudaMemcpyAsync(h_x, h, (size_t) E * 4, cudaMemcpyDeviceToHost, st));
            const auto w0 = Clock::now();
            CK(cudaStreamSynchronize(st));
            wait_ms += ms_since(w0);
            const auto c0 = Clock::now();
            N::dense_cpu(M.head.type, M.head_host, M.head.n_in, M.head.n_out, h_x, h_logits, *pool, head_act);
            head_ms += ms_since(c0);
        } else {
            matmul(M.head, h, logits, 1);
        }
        pos += T;
        if (M.lock_experts && !hot_locked && (seen += T) >= lock_after) lock_hot();
    }

    // --lock-experts: after the first --lock-after tokens (256), VirtualLock the most-used experts (and the CPU head) so Windows trims other
    // programs instead of re-reading them from disk.  Locked pages are charged to the commit limit, so the budget is the
    // commit headroom minus 12 GiB (what CFD and the rest may still allocate), capped by --lock-gib.
    std::map<size_t, std::vector<uint32_t>> hot;
    bool hot_locked = false;
    int64_t seen = 0;
    double lock_gib_cap = 1e9;
    int64_t lock_after = 256;
    void lock_hot() {
        hot_locked = true;
#if defined(_WIN32)
        const Model& M = *m;
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof ms;
        GlobalMemoryStatusEx(&ms);
        const double head_gib = (double) ms.ullAvailPageFile / 1073741824.0;
        double budget = std::min(lock_gib_cap, head_gib - 12.0);
        if (budget < 1.0) {
            std::fprintf(stderr, "strata-nemotron: no experts locked (commit headroom %.1f GiB)\n", head_gib);
            return;
        }
        struct Slice { double c; const uint8_t* p; size_t n; };
        std::vector<Slice> sl;
        if (M.head_cpu) sl.push_back({1e30, M.head_host, (size_t) M.head.n_out * (size_t) (M.head.n_in / 32) * 34});
        for (auto& [l, cnt] : hot) {
            const Layer& L = M.layers[l];
            for (int e = 0; e < L.n_expert; ++e) {
                const double c = cnt[(size_t) e];
                sl.push_back({c, L.ex.up + (size_t) e * L.n_ff * L.ex.up_row, (size_t) L.n_ff * L.ex.up_row});
                sl.push_back({c, L.ex.down + (size_t) e * M.n_lat * L.ex.down_row, (size_t) M.n_lat * L.ex.down_row});
            }
        }
        std::stable_sort(sl.begin(), sl.end(), [](const Slice& a, const Slice& b) { return a.c > b.c; });
        size_t want = 0;
        size_t take = 0;
        for (; take < sl.size() && (double) (want + sl[take].n) / 1073741824.0 <= budget; ++take) want += sl[take].n;
        const SIZE_T extra = (SIZE_T) 1 << 30;
        if (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), want + extra, want + 2 * extra,
                                        QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
            std::fprintf(stderr, "strata-nemotron: no experts locked (working set refused, error %lu)\n", GetLastError());
            return;
        }
        size_t locked = 0;
        double covered = 0, all = 0;
        for (size_t i = 0; i < sl.size(); ++i) {
            if (sl[i].c < 1e29) all += sl[i].c;
            if (i < take && VirtualLock((LPVOID) sl[i].p, sl[i].n)) {
                locked += sl[i].n;
                if (sl[i].c < 1e29) covered += sl[i].c;
            }
        }
        std::fprintf(stderr, "strata-nemotron: %.1f GiB of the most-used experts locked in RAM (budget %.1f GiB of %.1f GiB commit "
                     "headroom; they served %.0f%% of the expert reads so far)\n", locked / 1073741824.0, budget, head_gib,
                     all > 0 ? 100.0 * covered / all : 0.0);
#endif
    }

    /// Reads n tokens in chunks; cb(done) after each chunk (false: stop).  Returns the tokens read.
    template <class F> int64_t process(const int32_t* toks, int64_t n, F&& cb) {
        int64_t done = 0;
        while (done < n) {
            const int T = (int) std::min<int64_t>(chunk, n - done);
            forward(toks + done, T);
            done += T;
            if (!cb(done)) break;
        }
        return done;
    }

    int greedy() {
        if (m->head_cpu) {
            int best = 0;
            for (int i = 1; i < m->n_vocab; ++i)
                if (h_logits[i] > h_logits[best]) best = i;
            return best;
        }
        k_argmax<<<128, 256, 0, st>>>(logits, m->n_vocab, pv, pi);
        k_argmax2<<<1, 32, 0, st>>>(pv, pi, 128, amax);
        CK(cudaMemcpyAsync(h_amax, amax, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return *h_amax;
    }
    void host_logits(std::vector<float>& lg) {
        if (m->head_cpu) {
            lg.assign(h_logits, h_logits + m->n_vocab);
            return;
        }
        CK(cudaMemcpyAsync(h_logits, logits, (size_t) m->n_vocab * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        lg.assign(h_logits, h_logits + m->n_vocab);
    }
};

// ------------------------------------------------------------------------------------------------ sampling
struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 0, last_n = 64;
    uint64_t seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f) {
        std::map<int, int> cnt;
        const size_t from = hist.size() > (size_t) s.last_n && s.last_n > 0 ? hist.size() - (size_t) s.last_n : 0;
        for (size_t i = from; i < hist.size(); ++i) cnt[hist[i]]++;
        for (auto [t, c] : cnt) {
            if (t < 0 || t >= (int) lg.size()) continue;
            float& v = lg[(size_t) t];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + (c > 0 ? s.present : 0.f);
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    const int k = s.top_k > 0 ? std::min<int>(s.top_k, (int) lg.size()) : std::min<int>(1000, (int) lg.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
    idx.resize((size_t) k);
    std::vector<double> p((size_t) k);
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (int i = 0; i < k; ++i) sum += p[(size_t) i] = std::exp((lg[(size_t) idx[(size_t) i]] - mx) / s.temperature);
    int n = k;
    if (s.min_p > 0.f)
        for (int i = 1; i < n; ++i)
            if (p[(size_t) i] / p[0] < s.min_p) { n = i; break; }
    if (s.top_p < 1.f) {
        double c = 0;
        for (int i = 0; i < n; ++i) {
            c += p[(size_t) i] / sum;
            if (c >= s.top_p) { n = i + 1; break; }
        }
    }
    double tot = 0;
    for (int i = 0; i < n; ++i) tot += p[(size_t) i];
    double r = std::uniform_real_distribution<double>(0, tot)(rng);
    for (int i = 0; i < n; ++i) {
        r -= p[(size_t) i];
        if (r <= 0) return idx[(size_t) i];
    }
    return idx[(size_t) (n - 1)];
}

std::vector<int32_t> parse_ids(const char* p) {
    std::vector<int32_t> v;
    while (*p) {
        while (*p == ' ' || *p == ',') ++p;
        if (!*p) break;
        char* e = nullptr;
        const long x = std::strtol(p, &e, 10);
        if (e == p) break;
        v.push_back((int32_t) x);
        p = e;
    }
    return v;
}

// ------------------------------------------------------------------------------------------------ serve
struct Serve {
    Engine& E;
    std::vector<int32_t> live;   // tokens whose effect the state holds
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::vector<float> lg;
    explicit Serve(Engine& e) : E(e) {}

    bool is_eos(int t) const { return std::find(E.m->eos.begin(), E.m->eos.end(), t) != E.m->eos.end(); }

    void request(const std::string& line) {
        char* p = const_cast<char*>(line.c_str()) + 4;
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* s0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(s0, (size_t) (p - s0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(s0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            return;
        }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            return;
        }
        for (int32_t t : ids)
            if (t < 0 || t >= E.m->n_vocab) {
                std::printf("ERR a token id is outside the vocabulary\n");
                std::fflush(stdout);
                return;
            }
        stop.store(false);
        // the held state continues when the prompt extends it (a recurrent state cannot go back: anything else reads afresh)
        size_t reuse = 0;
        if (!live.empty() && live.size() < ids.size() && std::equal(live.begin(), live.end(), ids.begin())) reuse = live.size();
        if (reuse == 0) {
            E.reset();
            live.clear();
        }
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        bool cancelled = false;
        const int64_t got = E.process(ids.data() + reuse, fresh, [&](int64_t done) -> bool {
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                        (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
            if (stop.load() && done < fresh) { cancelled = true; return false; }
            return true;
        });
        live.insert(live.end(), ids.begin() + (long) reuse, ids.begin() + (long) reuse + (long) got);
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            live.clear();   // (the chunk boundary is not a place the next prompt can be matched to cheaply)
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse, (long long) got);
            std::fflush(stdout);
            return;
        }
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist;
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        for (;;) {
            int nx;
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            } else {
                nx = E.greedy();
            }
            std::printf("T %d\n", nx);
            std::fflush(stdout);
            ++produced;
            hist.push_back(nx);
            if (is_eos(nx)) { finish = "stop"; break; }
            if (produced >= max_new) break;
            if (stop.load()) { finish = "cancel"; break; }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.forward(&one, 1);
            live.push_back(nx);
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms, decode_ms, finish,
                    reuse, (long long) fresh);
        std::fflush(stdout);
        std::fprintf(stderr, "strata-nemotron serve: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), %lld "
                     "generated in %.0f ms (%.1f tok/s), %s\n", ids.size(), reuse, (long long) fresh, prompt_ms,
                     prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, finish);
    }

    int run() {
        std::printf("READY %d stop\n", E.ctx);
        std::fflush(stdout);
        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (lines.empty()) return 0;
                line = lines.front();
                lines.pop_front();
            }
            if (line.empty() || line.rfind("BSTOP", 0) == 0 || line.rfind("BYIELD", 0) == 0) continue;
            if (line.rfind("GEN ", 0) == 0) {
                request(line);
            } else {
                std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
                std::fflush(stdout);
            }
        }
    }
};

const char* arg(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return nullptr;
}
bool flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return true;
    return false;
}

}  // namespace

int strata_nemotron_main(int argc, char** argv) {
    if (const char* ba = arg(argc, argv, "--bench-attn")) {
        // one-token CPU attention over N cached keys, 8 layers' worth (no model, no GPU)
        const int nkeys = std::atoi(ba), H = 32, HKV = 2, kvd = HKV * 128;
        const int th = arg(argc, argv, "--threads") ? std::atoi(arg(argc, argv, "--threads")) : 16;
        std::vector<uint16_t> K((size_t) nkeys * kvd * 8), V(K.size());
        for (size_t i = 0; i < K.size(); ++i) K[i] = V[i] = (uint16_t) (0x3000 + (i * 2654435761u >> 20) % 0x800);
        std::vector<float> q(H * 128, 0.01f), out(H * 128);
        N::SpinPool pool(th);
        for (int rep = 0; rep < 3; ++rep) {
            const auto t0 = Clock::now();
            for (int l = 0; l < 8; ++l)
                N::attn_decode(K.data() + (size_t) l * nkeys * kvd, V.data() + (size_t) l * nkeys * kvd, nkeys, H, HKV, q.data(),
                               out.data(), 0.088f, pool);
            std::printf("attention, %d keys x 8 layers, %d threads: %.1f ms\n", nkeys, th, ms_since(t0));
        }
        return 0;
    }
    if (flag(argc, argv, "--selftest-attn")) {
        // the pieced host-KV kernel against k_attn on random data (several pieces, a partial last tile)
        const int H = 32, HKV = 2, HD = 128, kvd = HKV * HD;
        int worst_fail = 0;
        for (int cfg = 0; cfg < 4; ++cfg) {
            const int T = cfg == 0 ? 1 : cfg == 1 ? 7 : cfg == 2 ? 133 : 1, pos = cfg == 0 ? 300 : cfg == 1 ? 250 : cfg == 2 ? 61 : 9000,
                      piece = cfg == 2 ? 1000 : cfg == 3 ? 4096 : 97;
            const int nkeys = pos + T, ctx = nkeys;
            std::vector<float> hq((size_t) T * H * HD);
            std::vector<__half> hkc((size_t) ctx * kvd), hvc((size_t) ctx * kvd);
            uint32_t s = 12345 + cfg;
            auto rnd = [&] { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffff) / 32768.f - 1.f; };
            for (auto& v : hq) v = rnd() * 3.f;
            for (auto& v : hkc) v = __float2half(rnd());
            for (auto& v : hvc) v = __float2half(rnd());
            float *q, *y0, *y1, *sc, *am, *al, *aa;
            __half *kc, *vc, *dk, *dv;
            CK(cudaMalloc(&q, hq.size() * 4));
            CK(cudaMalloc(&y0, hq.size() * 4));
            CK(cudaMalloc(&y1, hq.size() * 4));
            CK(cudaMalloc(&sc, (size_t) T * H * ctx * 4));
            CK(cudaMalloc(&am, (size_t) T * H * 4));
            CK(cudaMalloc(&al, (size_t) T * H * 4));
            CK(cudaMalloc(&aa, (size_t) T * H * HD * 4));
            CK(cudaMalloc(&kc, hkc.size() * 2));
            CK(cudaMalloc(&vc, hvc.size() * 2));
            CK(cudaMalloc(&dk, (size_t) piece * kvd * 2));
            CK(cudaMalloc(&dv, (size_t) piece * kvd * 2));
            CK(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(kc, hkc.data(), hkc.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(vc, hvc.data(), hvc.size() * 2, cudaMemcpyHostToDevice));
            const float scale = 1.0f / std::sqrt((float) HD);
            k_attn<<<T * H, 128>>>(q, kc, vc, y0, sc, H, HKV, pos, 0, ctx, scale);
            CK(cudaFuncSetAttribute(k_attn_tile<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, at_smem(16)));
            float *pm, *pl, *pa;
            CK(cudaMalloc(&pm, (size_t) AT_CAP_ROWS * H * 4));
            CK(cudaMalloc(&pl, (size_t) AT_CAP_ROWS * H * 4));
            CK(cudaMalloc(&pa, (size_t) AT_CAP_ROWS * H * HD * 4));
            attn_hostkv_gpu(q, y1, hkc.data(), hvc.data(), dk, dv, piece, pm, pl, pa, am, al, aa, H, HKV, pos, T, true, 0);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            std::vector<float> a(hq.size()), b(hq.size());
            CK(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost));
            auto cmp = [&](const char* what, const std::vector<float>& u) {
                double mx = 0, ma = 0;
                for (size_t i = 0; i < a.size(); ++i) mx = std::max(mx, (double) std::fabs(a[i] - u[i])), ma = std::max(ma, (double) std::fabs(a[i]));
                std::printf("attn selftest %s T=%d pos=%d piece=%d: max |diff| %.3g (max |y| %.3g)\n", what, T, pos, piece, mx, ma);
                if (!(mx < 1e-4)) worst_fail = 1;
            };
            cmp("gpu", b);
            if (T == 1) {
                N::SpinPool pool(8);
                std::vector<float> c(hq.size());
                N::attn_decode(reinterpret_cast<const uint16_t*>(hkc.data()), reinterpret_cast<const uint16_t*>(hvc.data()), nkeys, H,
                               HKV, hq.data(), c.data(), scale, pool);
                cmp("cpu", c);
            }
            cudaFree(pm), cudaFree(pl), cudaFree(pa);
            for (void* p : {(void*) q, (void*) y0, (void*) y1, (void*) sc, (void*) am, (void*) al, (void*) aa, (void*) kc, (void*) vc,
                            (void*) dk, (void*) dv})
                cudaFree(p);
        }
        std::printf(worst_fail ? "FAIL\n" : "OK\n");
        return worst_fail;
    }
    const char* model = arg(argc, argv, "--native");
    if (!model) model = arg(argc, argv, "--model");
    if (!model || flag(argc, argv, "--help")) {
        std::fprintf(stderr,
                     "strata-nemotron --model <first .gguf shard> [--max-context N] [--chunk N] [--threads N]\n"
                     "  --serve                          the server's engine protocol on stdin/stdout\n"
                     "  --tokens IDS [--n N] [--top K]   greedy reference run (prints gen: and top: lines)\n"
                     "  --bench-pp N --bench-tg N        prompt / decode speed\n");
        return 2;
    }
    int ctx = 8192, chunk = 256;
    int threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    if (const char* v = arg(argc, argv, "--max-context")) ctx = std::atoi(v);
    if (const char* v = arg(argc, argv, "--ctx")) ctx = std::atoi(v);
    if (const char* v = arg(argc, argv, "--chunk")) chunk = std::max(1, std::atoi(v));
    if (const char* v = arg(argc, argv, "--threads")) threads = std::max(1, std::atoi(v));
    int head_mode = -1;
    if (const char* v = arg(argc, argv, "--head")) head_mode = std::strcmp(v, "cpu") == 0 ? 1 : std::strcmp(v, "gpu") == 0 ? 0 : -1;
    // what the engine allocates after the weights: the Mamba state, the KV (f16), the chunk scratch, the CUDA libraries
    // --kv-host (or a context over 16k): the KV in RAM, only a 32k-token staging piece on the GPU
    const bool kv_host = flag(argc, argv, "--kv-host") || (ctx > 16384 && !flag(argc, argv, "--kv-gpu"));
    const size_t reserve = 450 + (kv_host ? 48 : (size_t) ctx * 16 / 1024) + (size_t) chunk * 1;
    Model M;
    std::string err;
    try {
        if (!load(model, M, err, head_mode, reserve, flag(argc, argv, "--lock-experts"))) {
            std::fprintf(stderr, "strata-nemotron: %s\n", err.c_str());
            std::printf("ERR %s\n", err.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-nemotron: %s\n", e.what());
        std::printf("ERR %s\n", e.what());
        return 1;
    }
    Engine E;
    const char* pv = arg(argc, argv, "--pin");
    const int pin = !pv ? 0 : std::strcmp(pv, "p") == 0 ? 1 : std::strcmp(pv, "cores") == 0 ? 2 : 0;
    E.init(M, ctx, chunk, threads, kv_host, pin);
    E.gpu_decode = flag(argc, argv, "--kv-host-gpu-decode");
    if (const char* v = arg(argc, argv, "--spin")) E.pool->spin = std::max(0, std::atoi(v));
    if (const char* v = arg(argc, argv, "--lock-after")) E.lock_after = std::max(1, std::atoi(v));
    if (const char* v = arg(argc, argv, "--lock-gib")) E.lock_gib_cap = std::atof(v);
    E.reset();
    if (flag(argc, argv, "--serve")) {
        Serve sv(E);
        return sv.run();
    }
    auto run_bench = [&](const char* bp) {
        const int npp = std::atoi(bp);
        const int ntg = arg(argc, argv, "--bench-tg") ? std::atoi(arg(argc, argv, "--bench-tg")) : 64;
        const int reps = arg(argc, argv, "--reps") ? std::atoi(arg(argc, argv, "--reps")) : 2;
        std::vector<int32_t> toks((size_t) npp);
        std::mt19937 r(1);
        for (auto& t : toks) t = 1000 + (int) (r() % 50000);
        for (int rep = 0; rep < reps; ++rep) {
            E.reset();
            // --bench-pp-at N: the prompt starts at cell N (prefill timing deep into a long context; the KV before it is filler)
            if (const char* ppat = arg(argc, argv, "--bench-pp-at")) E.pos = std::min(std::atoi(ppat), E.ctx - npp - ntg - 1);
            E.cpu_ms = 0;
            auto t0 = Clock::now();
            E.process(toks.data(), npp, [](int64_t) { return true; });
            int nx = E.greedy();
            const double pp_ms = ms_since(t0), pp_cpu = E.cpu_ms;
            // --bench-pos N: time decode as if N tokens came before (attention over N cells; timing only, the KV is filler)
            if (const char* bpos = arg(argc, argv, "--bench-pos")) E.pos = std::max(E.pos, std::min(std::atoi(bpos), E.ctx - ntg - 1));
            E.cpu_ms = E.head_ms = E.wait_ms = E.attn_ms = 0;
            t0 = Clock::now();
            for (int i = 0; i < ntg; ++i) {
                const int32_t one = nx;
                E.forward(&one, 1);
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t0);
            std::printf("bench %d: pp%d %.1f tok/s (CPU experts %.0f of %.0f ms), tg%d %.2f tok/s = %.1f ms/token: experts %.1f, "
                        "head %.1f, attention on the CPU %.1f, waiting for the GPU %.1f, other %.1f (at cell %d)\n",
                        rep, npp, 1000.0 * npp / pp_ms, pp_cpu, pp_ms, ntg, 1000.0 * ntg / tg_ms, tg_ms / ntg, E.cpu_ms / ntg,
                        E.head_ms / ntg, E.attn_ms / ntg, E.wait_ms / ntg,
                        (tg_ms - E.cpu_ms - E.head_ms - E.attn_ms - E.wait_ms) / ntg, E.pos);
            std::fflush(stdout);
        }
        return 0;
    };
    if (const char* bp = arg(argc, argv, "--bench-pp"); bp && !arg(argc, argv, "--prompts")) return run_bench(bp);
    if (const char* pf = arg(argc, argv, "--prompts")) {
        // one line of comma-separated ids per prompt; per prompt: "P i", "gen: ...", "top s: id:logprob ..." (one load)
        FILE* f = std::fopen(pf, "rb");
        if (!f) { std::fprintf(stderr, "strata-nemotron: cannot read %s\n", pf); return 1; }
        std::string all;
        char buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) all.append(buf, n);
        std::fclose(f);
        const int ngen = arg(argc, argv, "--n") ? std::atoi(arg(argc, argv, "--n")) : 64;
        const int top = arg(argc, argv, "--top") ? std::atoi(arg(argc, argv, "--top")) : 20;
        const int top_steps = arg(argc, argv, "--top-steps") ? std::atoi(arg(argc, argv, "--top-steps")) : 8;
        size_t a0 = 0;
        int pi_ = 0;
        std::vector<float> lg;
        while (a0 < all.size()) {
            size_t e = all.find('\n', a0);
            if (e == std::string::npos) e = all.size();
            const std::string line = all.substr(a0, e - a0);
            a0 = e + 1;
            const std::vector<int32_t> ids = parse_ids(line.c_str());
            if (ids.empty()) continue;
            E.reset();
            E.cpu_ms = 0;
            const auto t0 = Clock::now();
            E.process(ids.data(), (int64_t) ids.size(), [](int64_t) { return true; });
            E.greedy();
            const double pp_ms = ms_since(t0);
            std::vector<int32_t> gen;
            std::string tops;
            const auto d0 = Clock::now();
            for (int i = 0; i < ngen; ++i) {
                if (top > 0 && i < top_steps) {
                    E.host_logits(lg);
                    const float mx = *std::max_element(lg.begin(), lg.end());
                    double s = 0;
                    for (float v : lg) s += std::exp((double) v - mx);
                    const double lse = mx + std::log(s);
                    std::vector<int> idx(lg.size());
                    for (size_t j = 0; j < idx.size(); ++j) idx[j] = (int) j;
                    std::partial_sort(idx.begin(), idx.begin() + top, idx.end(),
                                      [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
                    tops += "top " + std::to_string(i) + ":";
                    for (int j = 0; j < top; ++j) {
                        char b[64];
                        std::snprintf(b, sizeof b, " %d:%.5f", idx[(size_t) j], (double) lg[(size_t) idx[(size_t) j]] - lse);
                        tops += b;
                    }
                    tops += "\n";
                }
                const int nx = E.greedy();
                gen.push_back(nx);
                if (i + 1 < ngen) {
                    const int32_t one = nx;
                    E.forward(&one, 1);
                }
            }
            const double tg_ms = ms_since(d0);
            std::printf("P %d pp %zu %.1f tok/s tg %d %.2f tok/s\ngen:", pi_++, ids.size(), 1000.0 * ids.size() / pp_ms, ngen,
                        1000.0 * ngen / tg_ms);
            for (int t : gen) std::printf(" %d", t);
            std::printf("\n%s", tops.c_str());
            std::fflush(stdout);
        }
        return arg(argc, argv, "--bench-pp") ? run_bench(arg(argc, argv, "--bench-pp")) : 0;
    }
    if (const char* tk = arg(argc, argv, "--tokens")) {
        const std::vector<int32_t> ids = parse_ids(tk);
        const int n = arg(argc, argv, "--n") ? std::atoi(arg(argc, argv, "--n")) : 64;
        const int top = arg(argc, argv, "--top") ? std::atoi(arg(argc, argv, "--top")) : 0;
        const int top_steps = arg(argc, argv, "--top-steps") ? std::atoi(arg(argc, argv, "--top-steps")) : 8;
        E.process(ids.data(), (int64_t) ids.size(), [](int64_t) { return true; });
        std::vector<int32_t> gen;
        std::vector<float> lg;
        std::string tops;
        for (int i = 0; i < n; ++i) {
            if (top > 0 && i < top_steps) {
                E.host_logits(lg);
                const float mx = *std::max_element(lg.begin(), lg.end());
                double s = 0;
                for (float v : lg) s += std::exp((double) v - mx);
                const double lse = mx + std::log(s);
                std::vector<int> idx(lg.size());
                for (size_t j = 0; j < idx.size(); ++j) idx[j] = (int) j;
                std::partial_sort(idx.begin(), idx.begin() + top, idx.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
                tops += "top " + std::to_string(i) + ":";
                for (int j = 0; j < top; ++j) {
                    char b[64];
                    std::snprintf(b, sizeof b, " %d:%.5f", idx[(size_t) j], (double) lg[(size_t) idx[(size_t) j]] - lse);
                    tops += b;
                }
                tops += "\n";
            }
            const int nx = E.greedy();
            gen.push_back(nx);
            if (i + 1 < n) {
                const int32_t one = nx;
                E.forward(&one, 1);
            }
        }
        std::printf("gen:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n%s", tops.c_str());
        return 0;
    }
    std::fprintf(stderr, "strata-nemotron: nothing to do (--serve, --tokens or --bench-pp)\n");
    return 2;
}
