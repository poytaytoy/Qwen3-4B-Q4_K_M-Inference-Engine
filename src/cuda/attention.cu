#include "Kernels.hpp"
#include <math_constants.h>
#include <limits>

namespace {
constexpr unsigned attention_threads = 128;
constexpr unsigned tile_size = 16;

// One block per query token/head. Online softmax keeps only a K/V tile,
// a running normalization and the output accumulator, never a T-by-T matrix.
__global__ void attention_kernel(const float* q, const float* k, const float* v,
    float* output, size_t tokens, size_t query_heads, size_t kv_heads, size_t head_dim) {
    extern __shared__ float tile[];
    float* keys = tile;
    float* values = keys + tile_size * head_dim;
    __shared__ float probabilities[tile_size];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float correction;

    const size_t query_row = blockIdx.x;
    const size_t token = query_row / query_heads;
    const size_t head = query_row % query_heads;
    const size_t kv_head = head / (query_heads / kv_heads);
    const float* query = q + query_row * head_dim;
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp = threadIdx.x / 32;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;

    if (threadIdx.x == 0) {
        maximum = -CUDART_INF_F;
        denominator = 0.0f;
    }
    __syncthreads();

    // Only visit keys up to this query's position (causal mask).
    for (size_t start = 0; start <= token; start += tile_size) {
        const unsigned count = static_cast<unsigned>(min(size_t{tile_size}, token + 1 - start));
        for (size_t i = threadIdx.x; i < count * head_dim; i += blockDim.x) {
            const size_t key_token = start + i / head_dim;
            const size_t source = (key_token * kv_heads + kv_head) * head_dim + i % head_dim;
            keys[i] = k[source];
            values[i] = v[source];
        }
        __syncthreads();

        // Each warp computes one score at a time, reducing across head components.
        for (unsigned key = warp; key < count; key += attention_threads / 32) {
            float score = 0.0f;
            for (size_t d = lane; d < head_dim; d += 32)
                score += query[d] * keys[key * head_dim + d];
            for (int offset = 16; offset > 0; offset /= 2)
                score += __shfl_down_sync(0xffffffffu, score, offset);
            if (lane == 0) probabilities[key] = score * scale;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float next_maximum = maximum;
            for (unsigned key = 0; key < count; ++key)
                next_maximum = fmaxf(next_maximum, probabilities[key]);
            correction = expf(maximum - next_maximum);
            float tile_sum = 0.0f;
            for (unsigned key = 0; key < count; ++key) {
                probabilities[key] = expf(probabilities[key] - next_maximum);
                tile_sum += probabilities[key];
            }
            denominator = correction * denominator + tile_sum;
            maximum = next_maximum;
        }
        __syncthreads();

        const size_t d0 = threadIdx.x;
        const size_t d1 = d0 + attention_threads;
        accumulator0 *= correction;
        accumulator1 *= correction;
        for (unsigned key = 0; key < count; ++key) {
            if (d0 < head_dim) accumulator0 += probabilities[key] * values[key * head_dim + d0];
            if (d1 < head_dim) accumulator1 += probabilities[key] * values[key * head_dim + d1];
        }
        __syncthreads();
    }

    if (threadIdx.x < head_dim)
        output[query_row * head_dim + threadIdx.x] = accumulator0 / denominator;
    if (threadIdx.x + attention_threads < head_dim)
        output[query_row * head_dim + threadIdx.x + attention_threads] = accumulator1 / denominator;
}

}

namespace kernels {

void attention(const float* q, const float* k, const float* v, float* output,
    size_t tokens, size_t query_heads, size_t kv_heads, size_t head_dim) {
    if (tokens == 0) return;
    if (!q || !k || !v || !output || kv_heads == 0 || query_heads == 0 ||
        query_heads % kv_heads != 0 || head_dim == 0 || head_dim > 256 ||
        tokens > static_cast<size_t>(std::numeric_limits<int>::max()) / query_heads)
        throw std::invalid_argument("Attention requires valid buffers, divisible Q/KV heads and head_dim in [1, 256]");

    const size_t shared_bytes = 2 * tile_size * head_dim * sizeof(float);
    attention_kernel<<<tokens * query_heads, attention_threads, shared_bytes>>>(
        q, k, v, output, tokens, query_heads, kv_heads, head_dim);
    detail::check_launch();
}

}
