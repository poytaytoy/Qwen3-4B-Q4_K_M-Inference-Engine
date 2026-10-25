#include "EmbedLookup.hpp"
#include "Weight.hpp"
#include "Kernels.hpp"
#include "ggml.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

EmbedLookup::EmbedLookup(const GGUF& model, const std::string& tensor_name) {
    const auto* info = model.get_tensor_info(tensor_name);
    type_ = info->type;
    const auto& dims = info->dimensions;

    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] != 1 || dims[3] != 1 ||
        dims[0] % ggml_blck_size(type_) != 0)
        throw std::invalid_argument("Invalid embedding dimensions for tensor storage type");
    
    embedding_size_ = static_cast<size_t>(dims[0]);
    vocabulary_size_ = static_cast<size_t>(dims[1]);
    row_bytes_ = ggml_row_size(type_, dims[0]);
    
    if (vocabulary_size_ > std::numeric_limits<size_t>::max() / row_bytes_ ||
        info->bytes != vocabulary_size_ * row_bytes_)
        throw std::invalid_argument("Embedding tensor byte size does not match its dimensions");
    
    table_ = model.read_tensor_data(tensor_name);
}

EmbedLookup::EmbedLookup(EmbedLookup&& other) noexcept
    : type_(std::exchange(other.type_, GGML_TYPE_COUNT)),
      table_(std::move(other.table_)),
      embedding_size_(std::exchange(other.embedding_size_, 0)),
      vocabulary_size_(std::exchange(other.vocabulary_size_, 0)),
      row_bytes_(std::exchange(other.row_bytes_, 0)) {}

EmbedLookup& EmbedLookup::operator=(EmbedLookup&& other) noexcept {
    if (this != &other) {
        type_ = std::exchange(other.type_, GGML_TYPE_COUNT);
        table_ = std::move(other.table_);
        embedding_size_ = std::exchange(other.embedding_size_, 0);
        vocabulary_size_ = std::exchange(other.vocabulary_size_, 0);
        row_bytes_ = std::exchange(other.row_bytes_, 0);
    }
    return *this;
}

DeviceBuffer EmbedLookup::lookup(const std::vector<uint32_t>& token_ids) const {
    for (auto id : token_ids){
        if (id >= vocabulary_size_) {
            throw std::out_of_range("Embedding token ID out of range");
        }
    }
        
    if (type_ != GGML_TYPE_F32 && type_ != GGML_TYPE_Q4_K && type_ != GGML_TYPE_Q6_K){
        throw std::invalid_argument("Unsupported embedding storage type");
    }

    const size_t count = token_ids.size() * embedding_size_;
    if (token_ids.empty()) return DeviceBuffer(size_t{0});

    std::vector<std::byte> packed(token_ids.size() * row_bytes_);

    for (size_t i = 0; i < token_ids.size(); ++i){
        std::memcpy(packed.data() + i * row_bytes_,table_.data() + static_cast<size_t>(token_ids[i]) * row_bytes_, row_bytes_);
    }
        
    DeviceBuffer device_rows(packed.data(), packed.size());
    if (type_ == GGML_TYPE_F32) return device_rows;

    DeviceBuffer output(count * sizeof(float));

    kernels::dequantize_embeddings(device_rows.data, static_cast<float*>(output.data),
        token_ids.size(), embedding_size_, row_bytes_, type_);

    return output;
}
