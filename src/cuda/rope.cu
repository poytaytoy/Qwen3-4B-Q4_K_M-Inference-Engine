#include "Kernels.hpp"

namespace {
using kernels::detail::threads;

// Qwen3 split-half rotation pairs j with j + head_dim / 2.
__global__ void rope_kernel(float* input, size_t pairs, size_t heads, size_t head_dim,
    size_t position_offset, float frequency_base, const kernels::DecodeToken* metadata) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    const size_t half = head_dim / 2;
    const size_t head = i / half;
    const size_t j = i % half;
    const size_t position = (metadata ? metadata->position : position_offset) + head / heads;
    const float angle = static_cast<float>(position) *
        powf(frequency_base, -2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
    float sine, cosine;
    sincosf(angle, &sine, &cosine);
    const size_t index = head * head_dim + j;
    const float x = input[index];
    const float y = input[index + half];
    input[index] = x * cosine - y * sine;
    input[index + half] = x * sine + y * cosine;
}
}

namespace kernels {
using detail::check_launch;
using detail::threads;

void rope(float* input, size_t tokens, size_t heads, size_t head_dim,
    size_t position_offset, float frequency_base, cudaStream_t stream, const DecodeToken* metadata) {
    if (tokens == 0) return;
    if (heads == 0 || head_dim == 0 || head_dim % 2 != 0 || !(frequency_base > 0.0f))
        throw std::invalid_argument("RoPE requires heads, an even head dimension and a positive frequency base");
    const size_t pairs = tokens * heads * (head_dim / 2);
    rope_kernel<<<(pairs + threads - 1) / threads, threads, 0, stream>>>(input, pairs, heads, head_dim, position_offset, frequency_base, metadata);
    check_launch();
}
}
