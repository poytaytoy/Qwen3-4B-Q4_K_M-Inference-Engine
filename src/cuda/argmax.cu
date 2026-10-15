#include "Kernels.hpp"
#include <algorithm>
#include <math_constants.h>

namespace {
using kernels::detail::threads;

// Pack score and ID together so atomicMax selects a consistent winner.
__device__ unsigned long long candidate(float value, uint32_t index) {
    if (isnan(value)) value = -CUDART_INF_F;
    if (value == 0.0f) value = 0.0f; // Give +0 and -0 identical ordering.
    const uint32_t bits = __float_as_uint(value);
    const uint32_t ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    return (static_cast<unsigned long long>(ordered) << 32) | ~index;
}

__global__ void argmax_kernel(const float* logits, unsigned long long* result, size_t count) {
    __shared__ unsigned long long block_best;
    if (threadIdx.x == 0) block_best = 0;
    __syncthreads();

    unsigned long long best = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        best = max(best, candidate(logits[i], static_cast<uint32_t>(i)));

    for (int offset = 16; offset > 0; offset /= 2)
        best = max(best, __shfl_down_sync(0xffffffffu, best, offset));

    if (threadIdx.x % 32 == 0) atomicMax(&block_best, best);
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(result, block_best);
}

void check_cuda(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA argmax failed: ") + cudaGetErrorString(error));
}

}

namespace kernels {

uint32_t argmax(const float* logits, size_t vocabulary_size) {
    if (!logits || vocabulary_size == 0 || vocabulary_size > UINT32_MAX)
        throw std::invalid_argument("Invalid argmax input");

    DeviceBuffer storage(sizeof(unsigned long long));
    auto* device_result = static_cast<unsigned long long*>(storage.data);
    check_cuda(cudaMemset(device_result, 0, sizeof(unsigned long long)));

    const unsigned blocks = std::min(size_t{256}, (vocabulary_size + threads - 1) / threads);
    argmax_kernel<<<blocks, threads>>>(logits, device_result, vocabulary_size);
    detail::check_launch();

    unsigned long long result;
    check_cuda(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost));
    return ~static_cast<uint32_t>(result);
}

}
