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

inline int get_sm_count() {
    int device;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA device query failed: ") + cudaGetErrorString(error));

    int count;
    error = cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA SM count query failed: ") + cudaGetErrorString(error));
    return count;
}

inline void check_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(error));
}

}

namespace kernels {

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

void matmul(const float* input, const Weight& weight, float* output, size_t rows,
    cudaStream_t stream = nullptr);

void matmul(const float* input, const Weight& weight, float* output, size_t rows,
    const CudaStream& stream);

void rope(float* input, size_t tokens, size_t heads, size_t head_dim,
    size_t position_offset, float frequency_base);

void attention(const float* q, const float* k, const float* v, float* output,
    size_t tokens, size_t query_heads, size_t kv_heads, size_t head_dim,
    size_t query_offset = 0);

void residual_add(float* state, const float* branch, size_t count);

void silu_multiply(const float* gate, const float* up, float* output, size_t count);

void dequantize_embeddings(const void* packed, float* output, size_t rows,
    size_t width, size_t row_bytes, ggml_type type);

uint32_t argmax(const float* logits, size_t vocabulary_size);

}
