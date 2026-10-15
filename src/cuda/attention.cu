// Causal grouped-query attention with tiled online softmax and FP32 K/V caches.
#include "Kernels.hpp"
#include <math_constants.h>
#include <limits>

namespace {
constexpr unsigned warp_width = 32;
constexpr unsigned attention_threads = 128;
constexpr unsigned key_tile_rows = 32;

__device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    return __shfl_sync(0xffffffffu, value, 0);
}

__global__ void attention_kernel(const float* q, const float* k, const float* v,
    float* output, size_t query_heads, size_t kv_heads, size_t head_dim, size_t query_offset) {
    __shared__ float scores[key_tile_rows];
    __shared__ float weights[key_tile_rows];
    __shared__ float previous_scale;
    __shared__ float normalizer;

    const size_t query_row = blockIdx.x;
    const size_t position = query_offset + query_row / query_heads;
    const size_t head = query_row % query_heads;
    const size_t kv_head = head / (query_heads / kv_heads);
    const size_t kv_width = kv_heads * head_dim;
    const float* query = q + query_row * head_dim;
    const unsigned lane = threadIdx.x % warp_width;
    const unsigned warp = threadIdx.x / warp_width;
    const size_t d0 = threadIdx.x;
    const size_t d1 = d0 + attention_threads;
    const float score_scale = rsqrtf(static_cast<float>(head_dim));
    float numerator0 = 0.0f, numerator1 = 0.0f;
    float running_maximum = -CUDART_INF_F;
    float running_sum = 0.0f;

    // Stop at this query's absolute position: later prefill tokens stay masked.
    for (size_t first = 0; first <= position; first += key_tile_rows) {
        const unsigned count = static_cast<unsigned>(min(size_t{key_tile_rows}, position + 1 - first));

        // Neighboring lanes read neighboring head components for each Q.K dot.
        for (unsigned slot = warp; slot < key_tile_rows; slot += attention_threads / warp_width) {
            float dot = 0.0f;
            if (slot < count) {
                const size_t base = (first + slot) * kv_width + kv_head * head_dim;
                for (size_t d = lane; d < head_dim; d += warp_width)
                    dot += query[d] * k[base + d];
            }
            dot = warp_sum(dot);
            if (lane == 0) scores[slot] = slot < count ? dot * score_scale : -CUDART_INF_F;
        }
        __syncthreads();

        // Stable online softmax: rescale the previous numerator/denominator
        // whenever a new tile raises the maximum. No full score matrix is stored.
        if (warp == 0) {
            const float maximum = fmaxf(running_maximum, warp_max(scores[lane]));
            const float rescale = expf(running_maximum - maximum);
            const float weight = expf(scores[lane] - maximum);
            weights[lane] = weight;
            running_sum = running_sum * rescale + warp_sum(weight);
            running_maximum = maximum;
            if (lane == 0) {
                previous_scale = rescale;
                normalizer = running_sum;
            }
        }
        __syncthreads();

        // Accumulate this tile's weighted values; normalize once at the end.
        numerator0 *= previous_scale;
        numerator1 *= previous_scale;
        for (unsigned slot = 0; slot < count; ++slot) {
            const size_t base = (first + slot) * kv_width + kv_head * head_dim;
            if (d0 < head_dim) numerator0 += weights[slot] * v[base + d0];
            if (d1 < head_dim) numerator1 += weights[slot] * v[base + d1];
        }
        __syncthreads(); // Finish shared reads before the next tile overwrites them.
    }

    if (d0 < head_dim) output[query_row * head_dim + d0] = numerator0 / normalizer;
    if (d1 < head_dim) output[query_row * head_dim + d1] = numerator1 / normalizer;
}
} // namespace

namespace kernels {
void attention(const float* q, const float* k, const float* v, float* output,
    size_t tokens, size_t query_heads, size_t kv_heads, size_t head_dim, size_t query_offset) {
    if (tokens == 0) return;
    if (!q || !k || !v || !output || kv_heads == 0 || query_heads == 0 ||
        query_heads % kv_heads != 0 || head_dim == 0 || head_dim > 256 ||
        tokens > static_cast<size_t>(std::numeric_limits<int>::max()) / query_heads ||
        tokens > std::numeric_limits<size_t>::max() / query_heads / head_dim / sizeof(float))
        throw std::invalid_argument("Attention requires valid buffers, divisible Q/KV heads and head_dim in [1, 256]");
    const size_t max_cached_tokens = std::numeric_limits<size_t>::max() / kv_heads / head_dim / sizeof(float);
    if (tokens > max_cached_tokens || query_offset > max_cached_tokens - tokens)
        throw std::invalid_argument("Attention cache dimensions overflow");

    attention_kernel<<<tokens * query_heads, attention_threads>>>(
        q, k, v, output, query_heads, kv_heads, head_dim, query_offset);
    detail::check_launch();
}
} // namespace kernels
