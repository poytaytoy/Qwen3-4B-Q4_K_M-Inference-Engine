// Capture-safe embedding lookup and writes to dynamically positioned cache rows.
#include "Kernels.hpp"

namespace {
template<ggml_type Type>
__global__ void embedding_kernel(const std::byte* table, const kernels::DecodeToken* metadata,
    float* output, size_t width) {
    const size_t d = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (d >= width) return;
    const size_t index = size_t(metadata->token) * width + d;
    if constexpr (Type == GGML_TYPE_Q4_K) output[d] = kernels::detail::dequantize_q4k(table, index);
    else if constexpr (Type == GGML_TYPE_Q6_K) output[d] = kernels::detail::dequantize_q6k(table, index);
    else if constexpr (Type == GGML_TYPE_F16) output[d] = __half2float(reinterpret_cast<const __half*>(table)[index]);
    else output[d] = reinterpret_cast<const float*>(table)[index];
}

__global__ void append_kernel(const float* key, const float* value, float* key_cache,
    float* value_cache, size_t width, const kernels::DecodeToken* metadata) {
    const size_t d = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (d < width) {
        const size_t destination = metadata->position * width + d;
        key_cache[destination] = key[d];
        value_cache[destination] = value[d];
    }
}
} // namespace

namespace kernels {
void embedding(const Weight& table, const DecodeToken* metadata, float* output,
    size_t width, cudaStream_t stream) {
    const auto* data = static_cast<const std::byte*>(table.device_buffer.data);
    const size_t blocks = (width + detail::threads - 1) / detail::threads;
    switch (table.datatype) {
    case GGML_TYPE_Q4_K: embedding_kernel<GGML_TYPE_Q4_K><<<blocks, detail::threads, 0, stream>>>(data, metadata, output, width); break;
    case GGML_TYPE_Q6_K: embedding_kernel<GGML_TYPE_Q6_K><<<blocks, detail::threads, 0, stream>>>(data, metadata, output, width); break;
    case GGML_TYPE_F16: embedding_kernel<GGML_TYPE_F16><<<blocks, detail::threads, 0, stream>>>(data, metadata, output, width); break;
    case GGML_TYPE_F32: embedding_kernel<GGML_TYPE_F32><<<blocks, detail::threads, 0, stream>>>(data, metadata, output, width); break;
    default: throw std::invalid_argument("Unsupported GPU embedding type");
    }
    detail::check_launch();
}

void append_kv(const float* key, const float* value, float* key_cache, float* value_cache,
    size_t width, const DecodeToken* metadata, cudaStream_t stream) {
    append_kernel<<<(width + detail::threads - 1) / detail::threads, detail::threads, 0, stream>>>(
        key, value, key_cache, value_cache, width, metadata);
    detail::check_launch();
}
} // namespace kernels
