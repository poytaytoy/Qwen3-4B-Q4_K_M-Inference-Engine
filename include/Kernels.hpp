#pragma once
#include "Weight.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif
#include <stdexcept>
#include <string>

// Host-callable CUDA entry points. Float activations are row-major device buffers.
// Launches use the default CUDA stream unless a stream parameter is provided.
namespace kernels::detail {

inline constexpr unsigned threads = 256;

// Internal launchers; public matmul validates inputs before dispatching.
void matmul_small(const float* input, const Weight& weight, float* output,
    size_t rows, cudaStream_t stream);
void matmul_tiled(const float* input, const Weight& weight, float* output,
    size_t rows, cudaStream_t stream);

#ifdef __CUDACC__
// Decode one weight; flattened index addresses the packed weight matrix.
__device__ __forceinline__ float dequantize_q4k(const std::byte* weight, size_t index) {
    const int lane = index % 32;
    const size_t block_index = index / 256;
    weight += block_index * 144;
    const int block_group =  index % 256 / 32;
    const float d = __half2float(*reinterpret_cast<const __half*>(weight));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(weight + 2));
    const auto* scales = reinterpret_cast<const uint8_t*>(weight + 4);
    int scale, offset;
    if (block_group < 4) {
        scale = scales[block_group] & 63;
        offset = scales[block_group + 4] & 63;
    } else {
        const int g = block_group - 4;
        scale = ((scales[g] >> 6) << 4) | (scales[8 + g] & 15);
        offset = ((scales[4 + g] >> 6) << 4) | (scales[8 + g] >> 4);
    }

    uint8_t q = static_cast<uint8_t>((weight + 16)[(block_group / 2) * 32 + lane]);

    q = (block_group % 2 == 0) ? q & 15 : q >> 4;

    return d * scale * q - dmin * offset;
}

__device__ __forceinline__ float dequantize_q6k(const std::byte* weight, size_t index) {
    const int lane = index % 32;
    weight += (index / 256) * 210;
    const int group = (index % 128) / 32;
    const int half = (index % 256) / 128;
    const auto* ql = reinterpret_cast<const uint8_t*>(weight);
    const auto* qh = reinterpret_cast<const uint8_t*>(weight + 128);
    const auto* scales = reinterpret_cast<const int8_t*>(weight + 192);
    const float d = __half2float(*reinterpret_cast<const __half*>(weight + 208));
    const int low = (ql[half * 64 + (group % 2) * 32 + lane] >> (4 * (group / 2))) & 15;
    const int high = (qh[half * 32 + lane] >> (2 * group)) & 3;
    const int q = (low | (high << 4)) - 32;
    return d * scales[half * 8 + group * 2 + lane / 16] * q;
}
#endif

inline void check_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(error));
}

}

namespace kernels {

// Updated outside capture; kernels read the current token and absolute cache position.
struct DecodeToken {
    uint32_t token;
    size_t position;
};

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
    size_t rows, size_t width, float epsilon, cudaStream_t stream = nullptr);

// Explicit implementations for comparison: v1 retains FP32 inputs; v2 quantizes
// activations to signed bytes and computes packed integer dot products.
namespace v1 {
void matmul(const float* input, const Weight& weight, float* output, size_t rows,
    cudaStream_t stream = nullptr);
}
namespace v2 {
// Workspace needs count + (count / 32) * sizeof(float) bytes, where count = rows * input_width.
// Its address must be 4-byte aligned.
// It may be reused after stream completion; this overload performs no allocations.
void matmul_with_workspace(const float* input, const Weight& weight, float* output,
    size_t rows, void* workspace, cudaStream_t stream);
void matmul(const float* input, const Weight& weight, float* output, size_t rows,
    cudaStream_t stream = nullptr);
}

// Defaults to packed v2; INFERENCE_MATMUL=v1 selects the preserved FP32 kernels.
// Overwrites output asynchronously on stream; buffers must remain alive until completion.
void matmul(const float* input, const Weight& weight, float* output, size_t rows,
    cudaStream_t stream = nullptr);

void matmul(const float* input, const Weight& weight, float* output, size_t rows,
    const CudaStream& stream);

void rope(float* input, size_t tokens, size_t heads, size_t head_dim,
    size_t position_offset, float frequency_base, cudaStream_t stream = nullptr,
    const DecodeToken* metadata = nullptr);

void attention(const float* q, const float* k, const float* v, float* output,
    size_t tokens, size_t query_heads, size_t kv_heads, size_t head_dim,
    size_t query_offset = 0, cudaStream_t stream = nullptr,
    const DecodeToken* metadata = nullptr);

void residual_add(float* state, const float* branch, size_t count, cudaStream_t stream = nullptr);

void silu_multiply(const float* gate, const float* up, float* output, size_t count, cudaStream_t stream = nullptr);

// Capture-safe operations: all storage is caller-owned and all work uses stream.
void embedding(const Weight& table, const DecodeToken* metadata, float* output,
    size_t width, cudaStream_t stream);
void append_kv(const float* key, const float* value, float* key_cache, float* value_cache,
    size_t width, const DecodeToken* metadata, cudaStream_t stream);
void argmax_device(const float* logits, unsigned long long* result,
    size_t vocabulary_size, cudaStream_t stream);

uint32_t argmax(const float* logits, size_t vocabulary_size);

}
