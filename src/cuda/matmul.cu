#include "Kernels.hpp"
#include <cuda_fp16.h>

namespace {
using kernels::detail::threads;

// Decode one component from a GGUF weight row.
__device__ float weight_value(const uint8_t* row, ggml_type type, size_t column) {
    if (type == GGML_TYPE_F32) return reinterpret_cast<const float*>(row)[column];

    const size_t j = column % 256;
    if (type == GGML_TYPE_Q4_K) {
        const uint8_t* block = row + (column / 256) * 144;
        const float d = __half2float(reinterpret_cast<const __half*>(block)[0]);
        const float dmin = __half2float(reinterpret_cast<const __half*>(block)[1]);
        const uint8_t* scales = block + 4;
        const unsigned group = j / 32;
        const unsigned scale = group < 4 ? scales[group] & 63
            : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
        const unsigned min = group < 4 ? scales[group + 4] & 63
            : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
        const unsigned packed = block[16 + (group / 2) * 32 + j % 32];
        const unsigned quant = (packed >> ((group % 2) * 4)) & 15;
        return d * scale * quant - dmin * min;
    }

    const uint8_t* block = row + (column / 256) * 210;
    const unsigned half = j / 128;
    const unsigned group = (j % 128) / 32;
    const unsigned position = j % 32;
    const unsigned low = (block[half * 64 + (group % 2) * 32 + position] >> (4 * (group / 2))) & 15;
    const unsigned high = (block[128 + half * 32 + position] >> (2 * group)) & 3;
    const int quant = static_cast<int>((high << 4) | low) - 32;
    const int scale = reinterpret_cast<const int8_t*>(block + 192)[j / 16];
    const float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
    return d * scale * quant;
}

// TODO: Optimize later with tiled matrix multiplication, shared-memory reuse,
// and dedicated quantized kernels. Baseline: one thread per output element.
__global__ void matmul_kernel(const float* input, const void* weight, ggml_type type,
    float* output, size_t rows, size_t input_width, size_t output_width, size_t row_bytes) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * output_width) return;
    const size_t row = index / output_width;
    const size_t column = index % output_width;
    const uint8_t* weight_row = static_cast<const uint8_t*>(weight) + column * row_bytes;
    float sum = 0.0f;
    for (size_t i = 0; i < input_width; ++i)
        sum += input[row * input_width + i] * weight_value(weight_row, type, i);
    output[index] = sum;
}

}

namespace kernels {

void matmul(const float* input, const Weight& weight, float* output, size_t rows) {
    if (rows == 0) return;
    if (weight.dimension.size() < 2 || weight.dimension[0] <= 0 || weight.dimension[1] <= 0 ||
        (weight.datatype != GGML_TYPE_F32 && weight.datatype != GGML_TYPE_Q4_K && weight.datatype != GGML_TYPE_Q6_K))
        throw std::invalid_argument("Matmul requires an F32, Q4_K or Q6_K weight matrix");

    const size_t input_width = weight.dimension[0];
    const size_t output_width = weight.dimension[1];
    if (weight.datatype != GGML_TYPE_F32 && input_width % 256 != 0)
        throw std::invalid_argument("Quantized matmul requires input width divisible by 256");
    const size_t row_bytes = weight.datatype == GGML_TYPE_F32 ? input_width * sizeof(float)
        : (input_width / 256) * (weight.datatype == GGML_TYPE_Q4_K ? 144 : 210);
    matmul_kernel<<<(rows * output_width + threads - 1) / threads, threads>>>(
        input, weight.device_buffer.data, weight.datatype, output, rows, input_width, output_width, row_bytes);
    detail::check_launch();
}

}
