#include "Weight.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

DeviceBuffer::DeviceBuffer(const void* source, std::size_t source_bytes, cudaStream_t allocation_stream)
    : stream(allocation_stream) {
    if (source_bytes == 0) return;
    if (!source) throw std::invalid_argument("Nonempty DeviceBuffer requires host data");

    auto error = cudaMallocAsync(&data, source_bytes, stream);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("cudaMallocAsync failed: ") + cudaGetErrorString(error));

    error = cudaMemcpyAsync(data, source, source_bytes, cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        cudaFreeAsync(data, stream);
        data = nullptr;
        throw std::runtime_error(std::string("Host-to-device copy failed: ") + cudaGetErrorString(error));
    }
    bytes = source_bytes;
}

DeviceBuffer::DeviceBuffer(std::size_t source_bytes, cudaStream_t allocation_stream)
    : stream(allocation_stream) {
    if (source_bytes == 0) return;
    const auto error = cudaMallocAsync(&data, source_bytes, stream);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("cudaMallocAsync failed: ") + cudaGetErrorString(error));
    bytes = source_bytes;
}

DeviceBuffer::~DeviceBuffer() {
    if (data) cudaFreeAsync(data, stream);
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data(std::exchange(other.data, nullptr)),
      bytes(std::exchange(other.bytes, 0)),
      stream(std::exchange(other.stream, nullptr)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        if (data) cudaFreeAsync(data, stream);
        data = std::exchange(other.data, nullptr);
        bytes = std::exchange(other.bytes, 0);
        stream = std::exchange(other.stream, nullptr);
    }
    return *this;
}

Weight::Weight(const GGUF& model, const std::string& tensor_name)
    : name(tensor_name) {
    const auto* tensor = model.get_tensor_info(name);
    datatype = tensor->type;
    dimension.assign(tensor->dimensions.begin(), tensor->dimensions.end());
    const auto data = model.read_tensor_data(name);
    device_buffer = DeviceBuffer(data.data(), data.size());
    // The buffer constructor waits for the copy before releasing the host data.
}
