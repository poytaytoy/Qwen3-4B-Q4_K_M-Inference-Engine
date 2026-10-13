#include "Kernels.hpp"

namespace {
using kernels::detail::threads;

__global__ void silu_multiply_kernel(const float* gate, const float* up, float* output, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float x = gate[i];
        output[i] = (x / (1.0f + expf(-x))) * up[i];
    }
}
}

namespace kernels {
using detail::check_launch;
using detail::threads;

void silu_multiply(const float* gate, const float* up, float* output, size_t count) {
    if (count == 0) return;
    silu_multiply_kernel<<<(count + threads - 1) / threads, threads>>>(gate, up, output, count);
    check_launch();
}
}
