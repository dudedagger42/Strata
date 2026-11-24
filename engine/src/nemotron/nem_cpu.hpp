// src/nemotron/nem_cpu.hpp - Nemotron-H latent MoE: the routed experts in RAM, computed on the CPU.
//
// An expert is up (n_lat -> n_ff), ReLU^2, down (n_ff -> n_lat), straight from the GGUF's [n_expert] stacked tensors (read
// in place from the mapped file). The dots are Strata's quant layer (the bits of ggml-cpu's AVX2 vec_dot), the activations
// quantized the way ggml-cpu does (the weight type's vec_dot_type), and the weighted sum is taken in the order llama.cpp's
// graph adds the experts, so a token's routed output matches llama.cpp with -ot exps=CPU.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace strata::nemotron {

/// A spinning pool: run(n, fn) calls fn(i) for i in [0, n) on every worker and the caller, and returns when all are done.
class SpinPool {
public:
    /// pin: 0 = let Windows place the threads; 1 = one thread per logical CPU of the fastest core class (P-cores), in order;
    /// 2 = one thread per physical core, fastest class first (no two threads on one core's hyperthreads)
    explicit SpinPool(int threads, int pin = 0);
    ~SpinPool();
    SpinPool(const SpinPool&) = delete;
    SpinPool& operator=(const SpinPool&) = delete;
    int size() const { return n_; }
    /// pause loops a waiting worker spins before it sleeps (WaitOnAddress) until the next run()
    int spin = 20000;
    /// fn(task) for task in [0, tasks), work-shared; blocks until every task is done.
    void run(int tasks, const std::function<void(int)>& fn);

private:
    void worker(int id);
    int n_ = 1;
    std::vector<std::thread> th_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> next_{0}, done_{0};
    std::atomic<bool> quit_{false};
    int tasks_ = 0;
    const std::function<void(int)>* fn_ = nullptr;
};

struct ExpertLayer {
    int n_lat = 0, n_ff = 0, n_expert = 0;
    int up_type = -1, down_type = -1;       // ggml types
    const uint8_t* up = nullptr;            // [n_expert][n_ff rows][n_lat]
    const uint8_t* down = nullptr;          // [n_expert][n_lat rows][n_ff]
    size_t up_row = 0, down_row = 0;        // bytes per row
    size_t up_act_bytes = 0, down_act_bytes = 0;
    const void* up_dot = nullptr;           // strata::quant::DotKernel*
    const void* down_dot = nullptr;
    const void* up_act = nullptr;           // strata::quant::ActKernel*
    const void* down_act = nullptr;
};

/// Fills the kernels of a layer; false with a reason when the quant layer has no dot for a type.
bool expert_layer_init(ExpertLayer& L, std::string& err);

/// The routed experts of `nt` tokens: out[t] = sum_j w[t][j] * down_e(relu(up_e(lat[t]))^2), e = ids[t][j], j in [0, k).
/// lat: nt x n_lat floats; ids/w: nt x k; out: nt x n_lat floats. Scratch is the caller's (grown here).
struct ExpertScratch {
    std::vector<uint8_t> act;     // nt x up_act_bytes
    std::vector<float> h;         // (nt * k) x n_ff
    std::vector<uint8_t> hq;      // (nt * k) x down_act_bytes
    std::vector<float> y;         // (nt * k) x n_lat
    std::vector<int32_t> order;   // pairs grouped by expert
    std::vector<int32_t> bounds;
};
/// y[r] = W[r] . x for rows r in [0, n_out) of a dense GGUF matrix in RAM (the output head when it stays on the CPU).
/// act: scratch for the quantized x (grown here).
bool dense_cpu(int type, const uint8_t* W, int n_in, int n_out, const float* x, float* y, SpinPool& pool,
               std::vector<uint8_t>& act);

/// One query token's attention over `nkeys` cached keys (f16 rows of HKV*128, K and V in RAM): out[H*128].  Grouped-query:
/// H / HKV query heads share each KV head.  q is unscaled; scale multiplies the scores.
void attn_decode(const uint16_t* K, const uint16_t* V, int nkeys, int H, int HKV, const float* q, float* out, float scale,
                 SpinPool& pool);

void experts_forward(const ExpertLayer& L, SpinPool& pool, ExpertScratch& s, int nt, int k, const float* lat,
                     const int32_t* ids, const float* w, float* out);

}  // namespace strata::nemotron
