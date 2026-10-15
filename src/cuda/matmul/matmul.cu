#include "Kernels.hpp"
#include <cstdlib>
#include <string_view>

namespace kernels {
void matmul(const float* input, const Weight& weight, float* output, size_t rows, cudaStream_t stream) {
    // Keep v1 as the default; select the experimental path per process.
    static const bool packed = [] {
        const char* version = std::getenv("INFERENCE_MATMUL");
        return version && std::string_view(version) == "v2";
    }();
    if (packed) v2::matmul(input, weight, output, rows, stream);
    else v1::matmul(input, weight, output, rows, stream);
}

void matmul(const float* input, const Weight& weight, float* output, size_t rows, const CudaStream& stream) {
    matmul(input, weight, output, rows, stream.get());
}
} // namespace kernels
