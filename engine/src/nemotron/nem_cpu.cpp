// src/nemotron/nem_cpu.cpp - see nem_cpu.hpp.
#include "nem_cpu.hpp"

#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/types.hpp"

#include <algorithm>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#define NEM_PAUSE() _mm_pause()
#else
#include <immintrin.h>
#define NEM_PAUSE() _mm_pause()
#endif

namespace strata::nemotron {

namespace Q = strata::quant;

SpinPool::SpinPool(int threads) : n_(std::max(1, threads)) {
    for (int i = 1; i < n_; ++i) th_.emplace_back([this, i] { worker(i); });
}

SpinPool::~SpinPool() {
    quit_.store(true);
    gen_.fetch_add(1);
    for (auto& t : th_) t.join();
}

void SpinPool::worker(int) {
    uint64_t seen = 0;
    for (;;) {
        uint64_t g;
        int spins = 0;
        while ((g = gen_.load(std::memory_order_acquire)) == seen) {
            if (++spins < 20000) NEM_PAUSE();
            else std::this_thread::yield();
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
