#include "EmbedLookup.hpp"
#include "Weight.hpp"
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
    for (auto id : token_ids)
        if (id >= vocabulary_size_) throw std::out_of_range("Embedding token ID out of range");
    
    if (embedding_size_ && token_ids.size() > std::numeric_limits<size_t>::max() / embedding_size_)
        throw std::length_error("Embedding lookup result is too large");
    
    using RowConverter = void (EmbedLookup::*)(const std::byte*, float*) const;
    RowConverter convert;
    switch (type_) {
        case GGML_TYPE_F32:  convert = &EmbedLookup::dequantize_row_fp32; break;
        case GGML_TYPE_Q4_K: convert = &EmbedLookup::dequantize_row_q4k; break;
        case GGML_TYPE_Q6_K: convert = &EmbedLookup::dequantize_row_q6k; break;
        default: throw std::invalid_argument("Unsupported embedding storage type");
    }

    std::vector<float> output(token_ids.size() * embedding_size_);
    #pragma omp parallel for schedule(static) if(token_ids.size() >= 32 && output.size() >= 65536)
    for (size_t i = 0; i < token_ids.size(); ++i) {
        (this->*convert)(table_.data() + static_cast<size_t>(token_ids[i]) * row_bytes_,
            output.data() + i * embedding_size_);
    }

    return DeviceBuffer(output.data(), output.size() * sizeof(float));
}

void EmbedLookup::dequantize_row_fp32(const std::byte* source, float* output) const {
    std::memcpy(output, source, row_bytes_);
}

void EmbedLookup::dequantize_row_q4k(const std::byte* source, float* __restrict output) const {
    for (size_t i = 0; i < embedding_size_ / 256; ++i) {
        const std::byte* block = source + i * 144;
        const auto* header = reinterpret_cast<const ggml_fp16_t*>(block);
        const float d = ggml_fp16_to_fp32(header[0]);
        const float dmin = ggml_fp16_to_fp32(header[1]);
        const auto* scales = reinterpret_cast<const uint8_t*>(block + 4);
        const auto* qs = reinterpret_cast<const uint8_t*>(block + 16);

        for (int group = 0; group < 8; ++group) {
            const int scale = group < 4 ? scales[group] & 63
                : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);

            const int min = group < 4 ? scales[group + 4] & 63
                : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);

            const float multiplier = d * scale;
            const float offset = dmin * min;
            const auto* quants = qs + (group / 2) * 32;
            const int shift = (group % 2) * 4;
            float* dst = output + i * 256 + group * 32;

            #pragma omp simd
            for (int j = 0; j < 32; ++j)
                dst[j] = multiplier * ((quants[j] >> shift) & 15) - offset;
        }
    }
}

void EmbedLookup::dequantize_row_q6k(const std::byte* source, float* __restrict output) const {
    size_t quantize_to_lookup = embedding_size_ / 256; 

    for (size_t i = 0; i < quantize_to_lookup; i ++ ){
        const std::byte* quant_block = source + i * 210;
        const int8_t* scales = reinterpret_cast<const int8_t*>(quant_block + 192);         
        float d = ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t*>(quant_block + 208)[0]);
        const uint8_t* ql = reinterpret_cast<const uint8_t*>(quant_block);
        const uint8_t* qh = reinterpret_cast<const uint8_t*>(quant_block + 128); 

        #pragma omp simd
        for (int j = 0; j < 256; j++){

            int half = j / 128;
            int group = (j % 128) / 32;
            int position = j % 32;

            uint8_t upper = qh[half * 32 + position];
            uint8_t lower = ql[half * 64 + (group % 2) * 32 + position];

            upper = (upper >> (2 * group)) & 3;
            lower = (lower >> (4 * (group / 2))) & 15;
            
            int8_t scale = (scales + (j / 16))[0];

            int16_t q = (upper << 4) + lower;
            q -= 32; 
            
            output[i * 256 + j] = d * scale * q;
        }
    }
}
