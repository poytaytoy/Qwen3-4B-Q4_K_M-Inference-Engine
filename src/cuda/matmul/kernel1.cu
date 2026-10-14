#include "Kernels.hpp"
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

namespace {

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

constexpr size_t strides_per_warp = 32;

// Decode once, accumulate several products locally, then reduce once per row.
template<ggml_type Type>
__global__ void matmul_kernel_strided(const float* __restrict input, const std::byte* __restrict weight,
    float* __restrict output, size_t rows, size_t input_width, size_t output_width,
    size_t warps_per_output) {
    const size_t warp_id = (size_t(blockDim.x) * blockIdx.x + threadIdx.x) / 32;
    const size_t output_column = warp_id / warps_per_output;
    if (output_column >= output_width) return;

    const size_t lane = threadIdx.x % 32;
    const size_t begin = (warp_id % warps_per_output) * strides_per_warp * 32 + lane;
    float decoded[strides_per_warp]{};
    #pragma unroll
    for (size_t stride = 0; stride < strides_per_warp; ++stride) {
        const size_t index = begin + stride * 32;
        if (index < input_width) {
            const size_t weight_index = output_column * input_width + index;
            if constexpr (Type == GGML_TYPE_Q4_K)
                decoded[stride] = dequantize_q4k(weight, weight_index);
            else
                decoded[stride] = dequantize_q6k(weight, weight_index);
        }
    }

    for (size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        #pragma unroll
        for (size_t stride = 0; stride < strides_per_warp; ++stride) {
            const size_t index = begin + stride * 32;
            if (index < input_width)
                sum += input[row * input_width + index] * decoded[stride];
        }
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) {
            auto* destination = output + row * output_width + output_column;
            if (warps_per_output == 1) *destination = sum;
            else atomicAdd(destination, sum);
        }
    }
}

} // namespace

namespace kernels {

inline namespace v1 {
    void matmul(const float* input, const Weight& weight, float* output, size_t rows, cudaStream_t stream) {
        if (weight.datatype != GGML_TYPE_Q4_K && weight.datatype != GGML_TYPE_Q6_K)
            throw std::invalid_argument("matmul supports Q4_K and Q6_K weights only");

        if (weight.dimension.size() < 2 || weight.dimension[0] <= 0 ||
            weight.dimension[0] % 256 != 0 || weight.dimension[1] <= 0)
            throw std::invalid_argument("Invalid matmul weight dimensions");

        if (rows == 0) return;

        const size_t input_width = weight.dimension[0];
        const size_t output_width = weight.dimension[1];

        const size_t chunks_per_output = input_width / 32;
        const size_t warps_per_output = (chunks_per_output - 1) / strides_per_warp + 1;
        const size_t groups = output_width * warps_per_output;
        const size_t blocks = (groups - 1) / (detail::threads / 32) + 1;

        const size_t count = rows * output_width;

        const auto error = cudaMemsetAsync(output, 0, count * sizeof(float), stream);

        if (error != cudaSuccess)
            throw std::runtime_error(std::string("matmul output clear failed: ") + cudaGetErrorString(error));

        const auto* data = reinterpret_cast<const std::byte*>(weight.device_buffer.data);
        if (weight.datatype == GGML_TYPE_Q4_K)
            matmul_kernel_strided<GGML_TYPE_Q4_K><<<blocks, detail::threads, 0, stream>>>(
                input, data, output, rows, input_width, output_width, warps_per_output);
        else
            matmul_kernel_strided<GGML_TYPE_Q6_K><<<blocks, detail::threads, 0, stream>>>(
                input, data, output, rows, input_width, output_width, warps_per_output);

        detail::check_launch();
    }

    void matmul(const float* input, const Weight& weight, float* output, size_t rows,
        const CudaStream& stream) {
        matmul(input, weight, output, rows, stream.get());
    }
}

} // namespace kernels
