#include "Weight.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

DeviceBuffer::DeviceBuffer(const void* source, std::size_t source_bytes) {
    // An empty buffer needs no CUDA allocation or active device.
    if (source_bytes == 0) return;
    if (!source) throw std::invalid_argument("Nonempty DeviceBuffer requires host data");

    cudaError_t error = cudaMalloc(&data, source_bytes);
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(error));
    }

    error = cudaMemcpy(data, source, source_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        // The destructor won't run if construction throws.
        cudaFree(data);
        data = nullptr;
        throw std::runtime_error(std::string("Host-to-device copy failed: ") + cudaGetErrorString(error));
    }
    bytes = source_bytes;
}

DeviceBuffer::DeviceBuffer(std::size_t source_bytes){
    if (source_bytes == 0) return;
    
    cudaError_t error = cudaMalloc(&data, source_bytes);
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(error));
    }
}

DeviceBuffer::~DeviceBuffer() {
    if (data) cudaFree(data);
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data(std::exchange(other.data, nullptr)),
      bytes(std::exchange(other.bytes, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        if (data) cudaFree(data);
        data = std::exchange(other.data, nullptr);
        bytes = std::exchange(other.bytes, 0);
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
    // cudaMemcpy is synchronous; the temporary CPU buffer can now be released.
}
