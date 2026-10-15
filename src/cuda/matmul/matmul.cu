#include "Kernels.hpp"
#include <cstdlib>
#include <string_view>

namespace kernels {
void matmul(const float* input, const Weight& weight, float* output, size_t rows, cudaStream_t stream) {
    // Packed v2 is the default; retain v1 for explicit comparisons.
    static const bool packed = [] {
        const char* version = std::getenv("INFERENCE_MATMUL");
        return !version || std::string_view(version) != "v1";
    }();
    if (packed) v2::matmul(input, weight, output, rows, stream);
    else v1::matmul(input, weight, output, rows, stream);
}

void matmul(const float* input, const Weight& weight, float* output, size_t rows, const CudaStream& stream) {
    matmul(input, weight, output, rows, stream.get());
}
} // namespace kernels
