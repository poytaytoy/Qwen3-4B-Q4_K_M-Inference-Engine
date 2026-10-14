#include "Kernels.hpp"
#include <utility>

namespace {
void check_stream(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA stream error: ") + cudaGetErrorString(error));
}
}

namespace kernels {

CudaStream::CudaStream() {
    check_stream(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

CudaStream::~CudaStream() {
    if (stream_) cudaStreamDestroy(stream_);
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (stream_) cudaStreamDestroy(stream_);
        stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
}

cudaStream_t CudaStream::get() const {
    if (!stream_) throw std::logic_error("CUDA stream was moved from");
    return stream_;
}

void CudaStream::synchronize() const {
    check_stream(cudaStreamSynchronize(get()));
}

bool CudaStream::ready() const {
    const auto error = cudaStreamQuery(get());
    if (error == cudaErrorNotReady) return false;
    check_stream(error);
    return true;
}

} // namespace kernels
