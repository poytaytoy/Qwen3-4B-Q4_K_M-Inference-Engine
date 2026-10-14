#include "Kernels.hpp"
#include "ggml.h"
#include <limits>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

namespace {

struct Q4KScaleOffset {
    int scale;
    int offset;
};

// scales points to the 12-byte field at block + 4; group is in [0, 7].
__device__ __forceinline__ Q4KScaleOffset q4k_scale_offset(const uint8_t* scales, int group) {
    if (group < 4)
        return {scales[group] & 63, scales[group + 4] & 63};

    const int g = group - 4;
    return {
        ((scales[g] >> 6) << 4) | (scales[8 + g] & 15),
        ((scales[4 + g] >> 6) << 4) | (scales[8 + g] >> 4)
    };
}

__global__ void matmul_kernel_q4k(const float* __restrict input, const std::byte* __restrict weight,
    float* __restrict output, size_t rows, size_t input_width, size_t output_width) {
    const std::size_t warp_id = (size_t(blockDim.x) * blockIdx.x + threadIdx.x) / 32;
    const size_t groups = output_width * (input_width / 32);
    if (warp_id >= groups) return;
    const std::size_t index_loc = warp_id * 32;
    const int true_thread_id = threadIdx.x % 32;
    const std::size_t block_index = index_loc / 256;
    weight += block_index * 144;
    const int block_group =  index_loc % 256 / 32;
    const float d = __half2float(*reinterpret_cast<const __half*>(weight));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(weight + 2));
    const auto scale_offset = q4k_scale_offset(
        reinterpret_cast<const uint8_t*>(weight + 4), block_group);
    const int scale = scale_offset.scale;
    const int offset = scale_offset.offset;

    uint8_t q = static_cast<uint8_t>((weight + 16)[(block_group / 2) * 32 + true_thread_id]);

    q = (block_group % 2 == 0) ? q & 15 : q >> 4;  

    float unquantized = d * scale * q - dmin * offset;
    const std::size_t input_index = (index_loc + true_thread_id) % input_width;

    for (size_t i = 0; i < rows; i++){
        float true_weight = input[input_index] * unquantized;
        for (int offset = 16; offset > 0; offset /= 2)
            true_weight += __shfl_down_sync(0xffffffff, true_weight, offset);
        if (true_thread_id == 0){
            atomicAdd(&output[index_loc / input_width], true_weight);
        }
        input += input_width;
        output += output_width;
    }

}

__global__ void matmul_kernel_q6k(const float* __restrict input, const std::byte* __restrict weight,
    
    float* __restrict output, size_t rows, size_t input_width, size_t output_width) {
    const size_t warp_id = (size_t(blockDim.x) * blockIdx.x + threadIdx.x) / 32;
    if (warp_id >= output_width * (input_width / 32)) return;

    const size_t index_loc = warp_id * 32;
    const int lane = threadIdx.x % 32;
    weight += (index_loc / 256) * 210;
    const int group = (index_loc % 128) / 32;
    const int half = (index_loc % 256) / 128;
    const auto* ql = reinterpret_cast<const uint8_t*>(weight);
    const auto* qh = reinterpret_cast<const uint8_t*>(weight + 128);
    const auto* scales = reinterpret_cast<const int8_t*>(weight + 192);
    const float d = __half2float(*reinterpret_cast<const __half*>(weight + 208));
    const int low = (ql[half * 64 + (group % 2) * 32 + lane] >> (4 * (group / 2))) & 15;
    const int high = (qh[half * 32 + lane] >> (2 * group)) & 3;
    const int q = (low | (high << 4)) - 32;
    const float unquantized = d * scales[half * 8 + group * 2 + lane / 16] * q;
    const size_t input_index = (index_loc + lane) % input_width;

    for (size_t row = 0; row < rows; ++row) {
        float sum = input[input_index] * unquantized;
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) atomicAdd(&output[index_loc / input_width], sum);
        input += input_width;
        output += output_width;
    }
}

} // namespace

namespace kernels {

void matmul(const float* input, const Weight& weight, float* output, size_t rows, cudaStream_t stream) {
    if (weight.datatype != GGML_TYPE_Q4_K && weight.datatype != GGML_TYPE_Q6_K)
        throw std::invalid_argument("matmul supports Q4_K and Q6_K weights only");
    
    if (weight.dimension.size() < 2 || weight.dimension[0] <= 0 ||
        weight.dimension[0] % 256 != 0 || weight.dimension[1] <= 0)
        throw std::invalid_argument("Invalid matmul weight dimensions");
    
    if (rows == 0) return;

    const size_t input_width = weight.dimension[0];
    const size_t output_width = weight.dimension[1];
    
    if (rows > std::numeric_limits<size_t>::max() / output_width ||
        output_width > std::numeric_limits<size_t>::max() / (input_width / 32))
        throw std::length_error("matmul dimensions overflow");
    
    const size_t groups = output_width * (input_width / 32);
    const size_t blocks = (groups - 1) / (detail::threads / 32) + 1;
    
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("matmul launch exceeds grid size limit");
    
    const size_t count = rows * output_width;
    
    if (count > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::length_error("matmul output byte size overflow");
    
    const auto error = cudaMemsetAsync(output, 0, count * sizeof(float), stream);
    
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("matmul output clear failed: ") + cudaGetErrorString(error));

    const auto* data = reinterpret_cast<const std::byte*>(weight.device_buffer.data);
    if (weight.datatype == GGML_TYPE_Q4_K)
        matmul_kernel_q4k<<<blocks, detail::threads, 0, stream>>>(input, data, output, rows, input_width, output_width);
    else
        matmul_kernel_q6k<<<blocks, detail::threads, 0, stream>>>(input, data, output, rows, input_width, output_width);
    
    detail::check_launch();
}

void matmul(const float* input, const Weight& weight, float* output, size_t rows,
    const CudaStream& stream) {
    matmul(input, weight, output, rows, stream.get());
}

} // namespace kernels
