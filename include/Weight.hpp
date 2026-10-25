#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "GGUF.hpp"

struct DeviceBuffer{
    void* data{nullptr};
    std::size_t bytes{0};

    cudaStream_t stream{nullptr};

    DeviceBuffer(const void* source, std::size_t bytes, cudaStream_t stream = nullptr);
    explicit DeviceBuffer(std::size_t bytes, cudaStream_t stream = nullptr);
    ~DeviceBuffer();
    
    DeviceBuffer(const DeviceBuffer& other) = delete; 
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
};

struct Weight {
    std::string name; 
    ggml_type datatype; 
    DeviceBuffer device_buffer{nullptr, 0};
    std::vector<std::int64_t> dimension; 

    Weight(const GGUF& model, const std::string& name);
};
