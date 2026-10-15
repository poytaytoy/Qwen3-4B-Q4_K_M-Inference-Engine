#include "Kernels.hpp"

namespace {
using kernels::detail::threads;

__global__ void residual_add_kernel(float* state, const float* branch, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) state[i] += branch[i];
}
}

namespace kernels {
using detail::check_launch;
using detail::threads;

void residual_add(float* state, const float* branch, size_t count, cudaStream_t stream) {
    if (count == 0) return;
    residual_add_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(state, branch, count);
    check_launch();
}
}
