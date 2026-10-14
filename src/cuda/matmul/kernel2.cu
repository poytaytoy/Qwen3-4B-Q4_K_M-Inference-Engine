#include "Kernels.hpp"
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

namespace {

// Decode once, accumulate several products locally, then reduce once per row.
template<ggml_type Type>
__global__ void matmul_kernel_strided(const float* __restrict input, const std::byte* __restrict weight,
    float* __restrict output, size_t rows, size_t input_width, size_t output_width,
    size_t warps_per_output) {

}

} // namespace

namespace kernels {

namespace v2 {

    void matmul(const float* input, const Weight& weight, float* output, size_t rows,
        const CudaStream& stream) {
    }
}

} // namespace kernels
