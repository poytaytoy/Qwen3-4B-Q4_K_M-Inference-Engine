#include "Kernels.hpp"
#include <cuda_fp16.h>

namespace {

__global__ void dequantize_q4k(const uint8_t* packed, float* output,
    size_t count, size_t width, size_t row_bytes) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const size_t column = index % width;
    const auto* row = packed + (index / width) * row_bytes;

    const unsigned position = column % 256;
    const auto* block = row + (column / 256) * 144;

    const unsigned group = position / 32;
    const auto* scales = block + 4;

    const int scale = group < 4 ? scales[group] & 63
        : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
    const int minimum = group < 4 ? scales[group + 4] & 63
        : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);

    const unsigned code = (block[16 + (group / 2) * 32 + position % 32]
        >> ((group % 2) * 4)) & 15;

    const float d = __half2float(*reinterpret_cast<const __half*>(block));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));

    output[index] = (d * scale) * code - dmin * minimum;
}

__global__ void dequantize_q6k(const uint8_t* packed, float* output,
    size_t count, size_t width, size_t row_bytes) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const size_t column = index % width;
    const auto* row = packed + (index / width) * row_bytes;

    const unsigned position = column % 256;
    const auto* block = row + (column / 256) * 210;

    const unsigned half = position / 128;
    const unsigned group = (position % 128) / 32;
    const unsigned lane = position % 32;

    const unsigned low = (block[half * 64 + (group % 2) * 32 + lane]
        >> (4 * (group / 2))) & 15;
    const unsigned high = (block[128 + half * 32 + lane] >> (2 * group)) & 3;
    const int code = static_cast<int>((high << 4) | low) - 32;

    const int scale = reinterpret_cast<const int8_t*>(block + 192)[position / 16];

    const float d = __half2float(*reinterpret_cast<const __half*>(block + 208));

    output[index] = (d * scale) * code;
}

}

namespace kernels {

void dequantize_embeddings(const void* packed, float* output, size_t rows,
    size_t width, size_t row_bytes, ggml_type type) {
    if (rows == 0) return;

    if (type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q6_K)
        throw std::invalid_argument("Unsupported embedding storage type");

    const size_t count = rows * width;

    const size_t blocks = (count + detail::threads - 1) / detail::threads;

    if (type == GGML_TYPE_Q4_K)
        dequantize_q4k<<<blocks, detail::threads>>>(
            static_cast<const uint8_t*>(packed), output, count, width, row_bytes);
    else
        dequantize_q6k<<<blocks, detail::threads>>>(
            static_cast<const uint8_t*>(packed), output, count, width, row_bytes);

    detail::check_launch();
}

}
