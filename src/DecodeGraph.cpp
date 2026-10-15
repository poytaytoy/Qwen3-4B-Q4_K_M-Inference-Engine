#include "DecodeGraph.hpp"
#include <algorithm>

DecodeGraph::DecodeGraph(size_t width, size_t query_width, size_t kv_width,
    size_t ffn_width, size_t vocabulary)
    : hidden(width * sizeof(float)), normalized(width * sizeof(float)),
      query(query_width * sizeof(float)), key(kv_width * sizeof(float)), value(kv_width * sizeof(float)),
      query_normalized(query_width * sizeof(float)), key_normalized(kv_width * sizeof(float)),
      attended(query_width * sizeof(float)), branch(width * sizeof(float)),
      gate(ffn_width * sizeof(float)), up(ffn_width * sizeof(float)), activated(ffn_width * sizeof(float)),
      logits(vocabulary * sizeof(float)),
      quantization(std::max({width, query_width, ffn_width}) * (32 + sizeof(float)) / 32),
      metadata(sizeof(kernels::DecodeToken)), winner(sizeof(unsigned long long)) {}

DecodeGraph::~DecodeGraph() {
    // This also protects buffer lifetime if a CUDA error interrupted inference.
    cudaStreamSynchronize(stream.get());
    if (executable) cudaGraphExecDestroy(executable);
}
