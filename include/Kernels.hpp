#pragma once
#include "Weight.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Host-callable CUDA entry points. Float activations are row-major device buffers.
// Launches use the default CUDA stream unless a stream parameter is provided.
namespace kernels::detail {

inline constexpr unsigned threads = 256;

inline void check_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(error));
}

}

namespace kernels {

// Owns a nonblocking stream. Destruction releases it without waiting for work.
// Buffers used by queued work must remain alive until that work completes.
class CudaStream {
public:
    CudaStream();
    ~CudaStream();
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    cudaStream_t get() const;
    void synchronize() const;
    bool ready() const;
private:
    cudaStream_t stream_{nullptr};
};

void rms_norm(const float* input, const Weight& scale, float* output,
    size_t rows, size_t width, float epsilon);

// weight dimensions are [input_width, output_width] in GGUF order.
// Supports Q4_K and Q6_K; input_width must be a positive multiple of 256.
// Clears output [rows, output_width] before atomically accumulating the product.
// Clear and product are queued asynchronously on stream; no host synchronization.
// Input, weights and output must not alias and must stay alive until completion.
// Callers manage dependencies when producing/consuming buffers on other streams.
void matmul(const float* input, const Weight& weight, float* output, size_t rows,
    cudaStream_t stream = nullptr);
void matmul(const float* input, const Weight& weight, float* output, size_t rows,
    const CudaStream& stream);

// In-place Qwen3 split-half RoPE, without extended-context scaling.
void rope(float* input, size_t tokens, size_t heads, size_t head_dim,
    size_t position_offset, float frequency_base);

// Causal grouped-query attention, with Q/K/V in [tokens, heads, head_dim] order.
// Tiled online softmax; head_dim must be in [1, 256]. Output must not alias Q/K/V.
// Full-sequence prefill for now; KV caching can be added later.
void attention(const float* q, const float* k, const float* v, float* output,
    size_t tokens, size_t query_heads, size_t kv_heads, size_t head_dim);

void residual_add(float* state, const float* branch, size_t count);

void silu_multiply(const float* gate, const float* up, float* output, size_t count);

// Greedy next-token selection; returns the selected ID to the host.
// Ties choose the lowest ID. NaNs are treated as negative infinity.
uint32_t argmax(const float* logits, size_t vocabulary_size);

}
