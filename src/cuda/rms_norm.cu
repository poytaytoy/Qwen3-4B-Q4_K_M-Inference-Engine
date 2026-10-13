#include "Kernels.hpp"

namespace {
using kernels::detail::threads;

__global__ void rms_norm_kernel(const float* input, const float* scale, float* output,
    size_t width, float epsilon) {
    __shared__ float sums[threads];
    const size_t row = static_cast<size_t>(blockIdx.x) * width;
    float sum = 0.0f;
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) {
        const float value = input[row + col];
        sum += value * value;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(sums[0] / static_cast<float>(width) + epsilon);
    for (size_t col = threadIdx.x; col < width; col += blockDim.x)
        output[row + col] = input[row + col] * inverse_rms * scale[col];
}
}

namespace kernels {
using detail::check_launch;
using detail::threads;

void rms_norm(const float* input, const Weight& scale, float* output,
    size_t rows, size_t width, float epsilon) {
    if (rows == 0) return;
    if (width == 0 || !(epsilon > 0.0f) || scale.datatype != GGML_TYPE_F32 ||
        scale.dimension.empty() || static_cast<size_t>(scale.dimension[0]) != width)
        throw std::invalid_argument("RMSNorm requires an F32 scale matching the row width and positive epsilon");
    rms_norm_kernel<<<rows, threads>>>(input, static_cast<const float*>(scale.device_buffer.data), output, width, epsilon);
    check_launch();
}
}
