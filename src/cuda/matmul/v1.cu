// Preserved FP32-input kernels: register-array reduction and classic shared-memory tiling.
#include "Kernels.hpp"
#include "ggml.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

using kernels::detail::dequantize_q4k;
using kernels::detail::dequantize_q6k;

constexpr size_t strides_per_warp = 32;
constexpr size_t weight_rows_per_block = 2;


template<ggml_type Type>
__global__ void matmul_kernel_strided(const float* __restrict input, const std::byte* __restrict weight,
    float* __restrict output, size_t rows, size_t input_width, size_t output_width,
    size_t warps_per_output, size_t rows_per_block) {
    __shared__ float partials[32]; // At most 32 warps in a CUDA block.
    const size_t warp = threadIdx.x / 32;
    const size_t local_column = warp / warps_per_output;
    const size_t output_column = size_t(blockIdx.x) * rows_per_block + local_column;
    const size_t lane = threadIdx.x % 32;
    const size_t begin = (warp % warps_per_output) * strides_per_warp * 32 + lane;
    float decoded[strides_per_warp]{};
    #pragma unroll
    for (size_t stride = 0; stride < strides_per_warp; ++stride) {
        const size_t index = begin + stride * 32;
        if (output_column < output_width && index < input_width) {
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
            if (output_column < output_width && index < input_width)
                sum += input[row * input_width + index] * decoded[stride];
        }

        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) partials[warp] = sum;
        __syncthreads(); // Every warp publishes before the writer reads.

        if (threadIdx.x == 1) {
            for (size_t column = 0; column < rows_per_block; ++column) {
                const size_t destination_column = size_t(blockIdx.x) * rows_per_block + column;
                if (destination_column < output_width) {
                    float total = 0.0f;
                    for (size_t w = 0; w < warps_per_output; ++w)
                        total += partials[column * warps_per_output + w];
                    output[row * output_width + destination_column] = total;
                }
            }
        }

        __syncthreads(); // Finish reads before reusing partials for the next token.
    }
}

} // namespace

namespace kernels::detail {
    void matmul_small(const float* input, const Weight& weight, float* output, size_t rows, cudaStream_t stream) {
        const size_t input_width = weight.dimension[0];
        const size_t output_width = weight.dimension[1];

        const size_t chunks_per_output = input_width / 32;
        const size_t warps_per_output = (chunks_per_output - 1) / strides_per_warp + 1;
        if (warps_per_output > 32)
            throw std::invalid_argument("matmul weight row requires more than 32 warps");
        // Keep complete rows together without exceeding 1024 threads per block.
        const size_t rows_per_block = std::min(weight_rows_per_block, 32 / warps_per_output);
        const size_t block_threads = rows_per_block * warps_per_output * 32;
        const size_t blocks = (output_width - 1) / rows_per_block + 1;

        const auto* data = reinterpret_cast<const std::byte*>(weight.device_buffer.data);
        if (weight.datatype == GGML_TYPE_Q4_K)
            matmul_kernel_strided<GGML_TYPE_Q4_K><<<blocks, block_threads, 0, stream>>>(
                input, data, output, rows, input_width, output_width, warps_per_output, rows_per_block);
        else
            matmul_kernel_strided<GGML_TYPE_Q6_K><<<blocks, block_threads, 0, stream>>>(
                input, data, output, rows, input_width, output_width, warps_per_output, rows_per_block);

        check_launch();
    }

} // namespace kernels::detail

// Classic shared-memory tiled X * W^T.
namespace {
using kernels::detail::dequantize_q4k;
using kernels::detail::dequantize_q6k;

constexpr size_t tile_rows = 16;
constexpr size_t tile_columns = 16;
constexpr size_t tile_k = 256; // One complete Q4_K/Q6_K quantization block.
constexpr size_t tile_pitch = tile_k + 1; // Avoid shared bank conflicts between rows.
constexpr size_t tile_threads = tile_rows * tile_columns;

// A thread owns one output element. Cooperatively load X and decoded W tiles,
// multiply them from shared memory, and retain the sum across K tiles.
template<ggml_type Type>
__global__ void matmul_kernel_tiled(const float* input, const std::byte* weight,
    float* output, size_t rows, size_t input_width, size_t output_width) {
    __shared__ float input_tile[tile_rows][tile_pitch];
    __shared__ float weight_tile[tile_columns][tile_pitch];
    const size_t local_row = threadIdx.y;
    const size_t local_column = threadIdx.x;
    const size_t thread = local_row * tile_columns + local_column;
    const size_t row_begin = size_t(blockIdx.y) * tile_rows;
    const size_t column_begin = size_t(blockIdx.x) * tile_columns;
    float sum = 0.0f;

    for (size_t start = 0; start < input_width; start += tile_k) {
        for (size_t i = thread; i < tile_rows * tile_k; i += tile_threads) {
            const size_t row = row_begin + i / tile_k;
            const size_t k = start + i % tile_k;
            input_tile[i / tile_k][i % tile_k] = row < rows && k < input_width ? input[row * input_width + k] : 0.0f;
        }
        for (size_t i = thread; i < tile_columns * tile_k; i += tile_threads) {
            const size_t column = column_begin + i / tile_k;
            const size_t k = start + i % tile_k;
            float value = 0.0f;
            if (column < output_width && k < input_width) {
                const size_t index = column * input_width + k;
                if constexpr (Type == GGML_TYPE_Q4_K) value = dequantize_q4k(weight, index);
                else value = dequantize_q6k(weight, index);
            }
            weight_tile[i / tile_k][i % tile_k] = value;
        }
        __syncthreads(); // Both tiles must be ready before multiplication.

        for (size_t k = 0; k < tile_k; ++k)
            sum += input_tile[local_row][k] * weight_tile[local_column][k];
        __syncthreads(); // Finish reads before loading the next K tile.
    }

    const size_t row = row_begin + local_row;
    const size_t column = column_begin + local_column;
    if (row < rows && column < output_width) output[row * output_width + column] = sum;
}

template<ggml_type Type>
void launch(const float* input, const Weight& weight, float* output, size_t rows, cudaStream_t stream) {
    const dim3 block(tile_columns, tile_rows);
    const dim3 grid((weight.dimension[1] + tile_columns - 1) / tile_columns,
        (rows + tile_rows - 1) / tile_rows);
    matmul_kernel_tiled<Type><<<grid, block, 0, stream>>>(input,
        static_cast<const std::byte*>(weight.device_buffer.data), output,
        rows, weight.dimension[0], weight.dimension[1]);
    kernels::detail::check_launch();
}
} // namespace

namespace kernels::detail {
void matmul_tiled(const float* input, const Weight& weight, float* output, size_t rows, cudaStream_t stream) {
    if (weight.datatype == GGML_TYPE_Q4_K) launch<GGML_TYPE_Q4_K>(input, weight, output, rows, stream);
    else launch<GGML_TYPE_Q6_K>(input, weight, output, rows, stream);
}

} // namespace kernels::detail

namespace kernels::v1 {
void matmul(const float* input, const Weight& weight, float* output, size_t rows, cudaStream_t stream) {
    if (weight.datatype != GGML_TYPE_Q4_K && weight.datatype != GGML_TYPE_Q6_K)
        throw std::invalid_argument("matmul supports Q4_K and Q6_K weights only");
    if (weight.dimension.size() < 2 || weight.dimension[0] <= 0 ||
        weight.dimension[0] % 256 != 0 || weight.dimension[1] <= 0)
        throw std::invalid_argument("Invalid matmul weight dimensions");
    if (rows == 0) return;
    if (rows < 5) detail::matmul_small(input, weight, output, rows, stream);
    else detail::matmul_tiled(input, weight, output, rows, stream);
}
} // namespace kernels::v1
