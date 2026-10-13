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
    const ggml_tensor* tensor = model.get_tensor(name);
    if (!tensor) {
        throw std::runtime_error("Tensor unavailable: " + name +
            ". Load the GGUF with load_weights=true and check the tensor name.");
    }
    switch (tensor->type) {
    case GGML_TYPE_Q6_K: datatype = DataType::Q6_K; break;
    case GGML_TYPE_Q4_K: datatype = DataType::Q4_K; break;
    case GGML_TYPE_F32: datatype = DataType::F32; break;
    default:
        throw std::invalid_argument("Unsupported weight type for " + name +
            ": " + ggml_type_name(tensor->type));
    }
    dimension.assign(std::begin(tensor->ne), std::end(tensor->ne));
    device_buffer = DeviceBuffer(tensor->data, ggml_nbytes(tensor));
}
