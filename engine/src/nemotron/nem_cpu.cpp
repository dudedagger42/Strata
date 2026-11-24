// src/nemotron/nem_cpu.cpp - see nem_cpu.hpp.
#include "nem_cpu.hpp"

#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#pragma comment(lib, "Synchronization.lib")
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#define NEM_PAUSE() _mm_pause()
#else
#include <immintrin.h>
#define NEM_PAUSE() _mm_pause()
#endif

namespace strata::nemotron {

namespace Q = strata::quant;

#if defined(_WIN32)
// CPU set ids in the order threads should take them (see SpinPool's pin)
static std::vector<ULONG> cpu_order(int pin) {
    std::vector<ULONG> out;
    ULONG len = 0;
    GetSystemCpuSetInformation(nullptr, 0, &len, GetCurrentProcess(), 0);
    std::vector<uint8_t> buf(len);
    if (!len || !GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data()), len, &len,
                                            GetCurrentProcess(), 0))
        return out;
    struct C { ULONG id; int cls, core, lp; };
    std::vector<C> cs;
    for (ULONG off = 0; off < len;) {
        auto* e = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data() + off);
        if (e->Type == CpuSetInformation)
            cs.push_back({e->CpuSet.Id, e->CpuSet.EfficiencyClass, e->CpuSet.CoreIndex, e->CpuSet.LogicalProcessorIndex});
        off += e->Size;
    }
    int top = 0;
    for (auto& c : cs) top = std::max(top, c.cls);
    if (pin == 1) {
        for (auto& c : cs)
            if (c.cls == top) out.push_back(c.id);
    } else {
        // first logical CPU of each core, fastest class first, then the second hyperthreads
        std::stable_sort(cs.begin(), cs.end(), [](const C& a, const C& b) { return a.cls > b.cls; });
        std::vector<int> seen;
        std::vector<ULONG> second;
        for (auto& c : cs) {
            if (std::find(seen.begin(), seen.end(), c.core) == seen.end()) {
                seen.push_back(c.core);
                out.push_back(c.id);
            } else {
                second.push_back(c.id);
            }
        }
        out.insert(out.end(), second.begin(), second.end());
    }
    return out;
}
#endif

SpinPool::SpinPool(int threads, int pin) : n_(std::max(1, threads)) {
#if defined(_WIN32)
    std::vector<ULONG> ids = pin ? cpu_order(pin) : std::vector<ULONG>{};
    auto bind = [ids](int i) {
        if (!ids.empty()) {
            ULONG id = ids[(size_t) i % ids.size()];
            SetThreadSelectedCpuSets(GetCurrentThread(), &id, 1);
        }
    };
    bind(0);
    for (int i = 1; i < n_; ++i) th_.emplace_back([this, i, bind] { bind(i); worker(i); });
#else
    for (int i = 1; i < n_; ++i) th_.emplace_back([this, i] { worker(i); });
#endif
}

SpinPool::~SpinPool() {
    quit_.store(true);
    gen_.fetch_add(1);
#if defined(_WIN32)
    WakeByAddressAll(&gen_);
#endif
    for (auto& t : th_) t.join();
}

void SpinPool::worker(int) {
    uint64_t seen = 0;
    for (;;) {
        uint64_t g;
        int spins = 0;
        while ((g = gen_.load(std::memory_order_acquire)) == seen) {
            if (++spins < spin) {
                NEM_PAUSE();
            } else {
#if defined(_WIN32)
                // asleep until run() bumps gen_ (spinning cores cost a laptop its turbo budget)
                uint64_t cur = seen;
                WaitOnAddress(&gen_, &cur, sizeof cur, 50);
#else
                std::this_thread::yield();
#endif
            }
        }
        seen = g;
        if (quit_.load()) return;
        for (;;) {
            const int i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= tasks_) break;
            (*fn_)(i);
        }
        done_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void SpinPool::run(int tasks, const std::function<void(int)>& fn) {
    if (tasks <= 0) return;
    if (n_ == 1 || tasks == 1) {
        for (int i = 0; i < tasks; ++i) fn(i);
        return;
    }
    fn_ = &fn;
    tasks_ = tasks;
    next_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_acq_rel);
#if defined(_WIN32)
    WakeByAddressAll(&gen_);
#endif
    for (;;) {
        const int i = next_.fetch_add(1, std::memory_order_relaxed);
        if (i >= tasks_) break;
        fn(i);
    }
    while (done_.load(std::memory_order_acquire) < n_ - 1) NEM_PAUSE();
}

bool expert_layer_init(ExpertLayer& L, std::string& err) {
    const Q::TypeInfo* tu = Q::type_info(L.up_type);
    const Q::TypeInfo* td = Q::type_info(L.down_type);
    if (!tu || !td || !tu->has_dot || !td->has_dot) {
        err = "expert type without a CPU dot";
        return false;
    }
    const Q::Arith ar = Q::default_arith();
    const Q::DotKernel* ku = Q::dot_kernel((Q::Type) L.up_type, ar);
    const Q::DotKernel* kd = Q::dot_kernel((Q::Type) L.down_type, ar);
    const Q::ActKernel* au = Q::act_kernel(tu->act, ar);
    const Q::ActKernel* ad = Q::act_kernel(td->act, ar);
    if (!ku || !kd || !au || !ad) {
        err = std::string("no CPU kernel for ") + tu->name + "/" + td->name;
        return false;
    }
    L.up_dot = ku;
    L.down_dot = kd;
    L.up_act = au;
    L.down_act = ad;
    L.up_row = Q::row_bytes(L.up_type, L.n_lat);
    L.down_row = Q::row_bytes(L.down_type, L.n_ff);
    L.up_act_bytes = Q::row_bytes((int) tu->act, L.n_lat);
    L.down_act_bytes = Q::row_bytes((int) td->act, L.n_ff);
    if (!L.up_row || !L.down_row || !L.up_act_bytes || !L.down_act_bytes) {
        err = "expert row sizes are not whole blocks";
        return false;
    }
    return true;
}

bool dense_cpu(int type, const uint8_t* W, int n_in, int n_out, const float* x, float* y, SpinPool& pool,
               std::vector<uint8_t>& act) {
    const Q::TypeInfo* ti = Q::type_info(type);
    if (!ti || !ti->has_dot) return false;
    const Q::Arith ar = Q::default_arith();
    const Q::DotKernel* kd = Q::dot_kernel((Q::Type) type, ar);
    const Q::ActKernel* ak = Q::act_kernel(ti->act, ar);
    if (!kd || !ak) return false;
    const size_t rb = Q::row_bytes(type, n_in);
    act.resize(Q::row_bytes((int) ti->act, n_in));
    ak->fn(x, act.data(), n_in);
    constexpr int R = 256;
    pool.run((n_out + R - 1) / R, [&](int i) {
        const int r1 = std::min(n_out, (i + 1) * R);
        for (int r = i * R; r < r1; ++r) kd->fn(n_in, y + r, W + (size_t) r * rb, act.data());
    });
    return true;
}

// exp for x <= 0 (softmax after the max is subtracted): Cephes-style range reduction + degree-6 polynomial, ~1e-7 relative
static inline __m256 exp256(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.f));
    const __m256 l2e = _mm256_set1_ps(1.44269504088896341f);
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, l2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

void attn_decode(const uint16_t* K, const uint16_t* V, int nkeys, int H, int HKV, const float* q, float* out, float scale,
                 SpinPool& pool) {
    constexpr int HD = 128, BK = 512, GMAX = 32;
    const int G = H / HKV, kvd = HKV * HD;
    const int nb = (nkeys + BK - 1) / BK;
    const int tasks = nb * HKV;
    // per task: m[G], l[G], acc[G][HD]
    const size_t per = (size_t) G * (HD + 2);
    std::vector<float> part((size_t) tasks * per);
    pool.run(tasks, [&](int ti) {
        const int g = ti % HKV, b = ti / HKV;
        const int k0 = b * BK, k1 = std::min(nkeys, k0 + BK), n = k1 - k0;
        alignas(32) float s[GMAX][BK];
        alignas(32) float qs[GMAX][HD];
        for (int h = 0; h < G; ++h)
            for (int c = 0; c < HD; ++c) qs[h][c] = q[(size_t) (g * G + h) * HD + c] * scale;
        // scores
        for (int j = 0; j < n; ++j) {
            const uint16_t* kr = K + (size_t) (k0 + j) * kvd + g * HD;
            __m256 kf[HD / 8];
            for (int c = 0; c < HD / 8; ++c) kf[c] = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kr + c * 8)));
            for (int h = 0; h < G; ++h) {
                __m256 a0 = _mm256_mul_ps(kf[0], _mm256_load_ps(qs[h]));
                __m256 a1 = _mm256_mul_ps(kf[1], _mm256_load_ps(qs[h] + 8));
                for (int c = 2; c < HD / 8; c += 2) {
                    a0 = _mm256_fmadd_ps(kf[c], _mm256_load_ps(qs[h] + c * 8), a0);
                    a1 = _mm256_fmadd_ps(kf[c + 1], _mm256_load_ps(qs[h] + c * 8 + 8), a1);
                }
                s[h][j] = hsum256(_mm256_add_ps(a0, a1));
            }
        }
        float* pm = part.data() + (size_t) ti * per;
        float* pl = pm + G;
        float* pa = pl + G;
        for (int h = 0; h < G; ++h) {
            float mx = -INFINITY;
            for (int j = 0; j < n; ++j) mx = std::max(mx, s[h][j]);
            const __m256 vm = _mm256_set1_ps(mx);
            __m256 sum = _mm256_setzero_ps();
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                const __m256 e = exp256(_mm256_sub_ps(_mm256_loadu_ps(&s[h][j]), vm));
                _mm256_storeu_ps(&s[h][j], e);
                sum = _mm256_add_ps(sum, e);
            }
            float ls = hsum256(sum);
            for (; j < n; ++j) {
                s[h][j] = std::exp(s[h][j] - mx);
                ls += s[h][j];
            }
            pm[h] = mx;
            pl[h] = ls;
        }
        // acc[h] = sum_j p[h][j] V[j]
        // 8 dims x 8 heads per pass: each V slice is converted once and feeds 8 accumulators (fits the 16 ymm registers)
        for (int h0 = 0; h0 < G; h0 += 8) {
            const int nh = std::min(8, G - h0);
            for (int c = 0; c < HD / 8; ++c) {
                __m256 acc[8];
                for (int hh = 0; hh < 8; ++hh) acc[hh] = _mm256_setzero_ps();
                const uint16_t* vb = V + (size_t) k0 * kvd + g * HD + c * 8;
                if (nh == 8) {
                    for (int jj = 0; jj < n; ++jj) {
                        const __m256 v = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vb + (size_t) jj * kvd)));
#pragma unroll
                        for (int hh = 0; hh < 8; ++hh) acc[hh] = _mm256_fmadd_ps(_mm256_broadcast_ss(&s[h0 + hh][jj]), v, acc[hh]);
                    }
                } else {
                    for (int jj = 0; jj < n; ++jj) {
                        const __m256 v = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vb + (size_t) jj * kvd)));
                        for (int hh = 0; hh < nh; ++hh) acc[hh] = _mm256_fmadd_ps(_mm256_broadcast_ss(&s[h0 + hh][jj]), v, acc[hh]);
                    }
                }
                for (int hh = 0; hh < nh; ++hh) _mm256_storeu_ps(pa + (size_t) (h0 + hh) * HD + c * 8, acc[hh]);
            }
        }
    });
    // merge the blocks of each KV head
    for (int g = 0; g < HKV; ++g)
        for (int h = 0; h < G; ++h) {
            float M = -INFINITY;
            for (int b = 0; b < nb; ++b) M = std::max(M, part[(size_t) (b * HKV + g) * per + h]);
            float L = 0.f;
            float* o = out + (size_t) (g * G + h) * HD;
            for (int c = 0; c < HD; ++c) o[c] = 0.f;
            for (int b = 0; b < nb; ++b) {
                const float* pm = part.data() + (size_t) (b * HKV + g) * per;
                const float w = std::exp(pm[h] - M);
                L += pm[G + h] * w;
                const float* pa = pm + 2 * G + (size_t) h * HD;
                for (int c = 0; c < HD; ++c) o[c] += pa[c] * w;
            }
            const float inv = 1.f / L;
            for (int c = 0; c < HD; ++c) o[c] *= inv;
        }
}

void experts_forward(const ExpertLayer& L, SpinPool& pool, ExpertScratch& s, int nt, int k, const float* lat,
                     const int32_t* ids, const float* w, float* out) {
    const Q::DotKernel* ku = static_cast<const Q::DotKernel*>(L.up_dot);
    const Q::DotKernel* kd = static_cast<const Q::DotKernel*>(L.down_dot);
    const Q::ActKernel* au = static_cast<const Q::ActKernel*>(L.up_act);
    const Q::ActKernel* ad = static_cast<const Q::ActKernel*>(L.down_act);
    const int P = nt * k;
    s.act.resize((size_t) nt * L.up_act_bytes);
    s.h.resize((size_t) P * L.n_ff);
    s.hq.resize((size_t) P * L.down_act_bytes);
    s.y.resize((size_t) P * L.n_lat);
    // the pairs grouped by expert (counting sort, stable: a token's pairs keep their order)
    s.bounds.assign((size_t) L.n_expert + 1, 0);
    for (int p = 0; p < P; ++p) s.bounds[(size_t) ids[p] + 1]++;
    for (int e = 0; e < L.n_expert; ++e) s.bounds[(size_t) e + 1] += s.bounds[(size_t) e];
    s.order.resize((size_t) P);
    {
        std::vector<int32_t> cur(s.bounds.begin(), s.bounds.end() - 1);
        for (int p = 0; p < P; ++p) s.order[(size_t) cur[(size_t) ids[p]]++] = p;
    }
    // the experts this batch uses, and work items of (expert, row chunk)
    struct Item { int32_t e, r0, r1; };
    std::vector<Item> up_items, down_items;
    const int up_chunk = nt == 1 ? 64 : 32, down_chunk = nt == 1 ? 64 : 32;
    for (int e = 0; e < L.n_expert; ++e) {
        if (s.bounds[(size_t) e] == s.bounds[(size_t) e + 1]) continue;
        for (int r = 0; r < L.n_ff; r += up_chunk) up_items.push_back({e, r, std::min(L.n_ff, r + up_chunk)});
        for (int r = 0; r < L.n_lat; r += down_chunk) down_items.push_back({e, r, std::min(L.n_lat, r + down_chunk)});
    }
    // 1. the latent inputs quantized (the up weights' activation format); one token: here, no pool round
    if (nt == 1) au->fn(lat, s.act.data(), L.n_lat);
    else pool.run(nt, [&](int t) { au->fn(lat + (size_t) t * L.n_lat, s.act.data() + (size_t) t * L.up_act_bytes, L.n_lat); });
    // 2. up + ReLU^2; the task that finishes an expert's last rows quantizes its hidden rows (no separate round)
    std::vector<std::atomic<int>> left((size_t) L.n_expert);
    for (auto& a : left) a.store(0, std::memory_order_relaxed);
    for (const Item& it : up_items) left[(size_t) it.e].fetch_add(1, std::memory_order_relaxed);
    pool.run((int) up_items.size(), [&](int i) {
        const Item it = up_items[(size_t) i];
        const uint8_t* W = L.up + ((size_t) it.e * L.n_ff) * L.up_row;
        const int32_t b0 = s.bounds[(size_t) it.e], b1 = s.bounds[(size_t) it.e + 1];
        for (int r = it.r0; r < it.r1; ++r) {
            const uint8_t* wr = W + (size_t) r * L.up_row;
            for (int32_t q = b0; q < b1; ++q) {
                const int p = s.order[(size_t) q];
                const int t = p / k;
                float v = 0.f;
                ku->fn(L.n_lat, &v, wr, s.act.data() + (size_t) t * L.up_act_bytes);
                v = v > 0.f ? v : 0.f;
                s.h[(size_t) p * L.n_ff + r] = v * v;
            }
        }
        if (left[(size_t) it.e].fetch_sub(1, std::memory_order_acq_rel) == 1)   // 3. this expert's rows are complete
            for (int32_t q = b0; q < b1; ++q) {
                const int p = s.order[(size_t) q];
                ad->fn(s.h.data() + (size_t) p * L.n_ff, s.hq.data() + (size_t) p * L.down_act_bytes, L.n_ff);
            }
    });
    // 4. down
    pool.run((int) down_items.size(), [&](int i) {
        const Item it = down_items[(size_t) i];
        const uint8_t* W = L.down + ((size_t) it.e * L.n_lat) * L.down_row;
        const int32_t b0 = s.bounds[(size_t) it.e], b1 = s.bounds[(size_t) it.e + 1];
        for (int r = it.r0; r < it.r1; ++r) {
            const uint8_t* wr = W + (size_t) r * L.down_row;
            for (int32_t q = b0; q < b1; ++q) {
                const int p = s.order[(size_t) q];
                float v = 0.f;
                kd->fn(L.n_ff, &v, wr, s.hq.data() + (size_t) p * L.down_act_bytes);
                s.y[(size_t) p * L.n_lat + r] = v;
            }
        }
    });
    // 5. the weighted sum in llama.cpp's order: (y0 w0) + (y1 w1) + ...
    for (int t = 0; t < nt; ++t) {
        float* o = out + (size_t) t * L.n_lat;
        const float* y0 = s.y.data() + (size_t) (t * k) * L.n_lat;
        const float w0 = w[(size_t) t * k];
        for (int c = 0; c < L.n_lat; ++c) o[c] = y0[c] * w0;
        for (int j = 1; j < k; ++j) {
            const float* yj = s.y.data() + (size_t) (t * k + j) * L.n_lat;
            const float wj = w[(size_t) t * k + j];
            for (int c = 0; c < L.n_lat; ++c) o[c] += yj[c] * wj;
        }
    }
}

}  // namespace strata::nemotron
