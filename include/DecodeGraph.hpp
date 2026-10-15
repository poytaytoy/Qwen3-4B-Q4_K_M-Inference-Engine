#pragma once
#include "Kernels.hpp"
#include <map>

// Per-session persistent storage and executable graph for one-token decoding.
// Destruction waits for its stream before releasing the graph and buffers.
struct DecodeGraph {
    DecodeGraph(size_t width, size_t query_width, size_t kv_width, size_t ffn_width, size_t vocabulary);
    ~DecodeGraph();
    DecodeGraph(const DecodeGraph&) = delete;
    DecodeGraph& operator=(const DecodeGraph&) = delete;

    kernels::CudaStream stream;
    DeviceBuffer hidden, normalized, query, key, value, query_normalized, key_normalized;
    DeviceBuffer attended, branch, gate, up, activated, logits, quantization, metadata, winner;
    std::map<std::string, const void*> cache_addresses;
    cudaGraphExec_t executable = nullptr;
};
