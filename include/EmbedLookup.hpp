#pragma once
#include "GGUF.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include "Weight.hpp"

// Owns an embedding table in its original storage format in CPU RAM.
class EmbedLookup {
public:
    explicit EmbedLookup(const GGUF& model, const std::string& tensor_name = "token_embd.weight");
    EmbedLookup(const EmbedLookup&) = delete;
    EmbedLookup& operator=(const EmbedLookup&) = delete;
    EmbedLookup(EmbedLookup&& other) noexcept;
    EmbedLookup& operator=(EmbedLookup&& other) noexcept;

    // Copies selected packed rows to the GPU and dequantizes them to FP32.
    // One embedding per input ID, contiguous in input order.
    // Result shape: [token_ids.size(), get_embedding_size()].
    DeviceBuffer lookup(const std::vector<uint32_t>& token_ids) const;
    size_t get_embedding_size() const { return embedding_size_; }
    size_t get_vocabulary_size() const { return vocabulary_size_; }
    ggml_type get_type() const { return type_; }
    size_t get_row_bytes() const { return row_bytes_; }

private:
    ggml_type type_{GGML_TYPE_COUNT};
    std::vector<std::byte> table_;
    size_t embedding_size_{0};
    size_t vocabulary_size_{0};
    size_t row_bytes_{0};
};
