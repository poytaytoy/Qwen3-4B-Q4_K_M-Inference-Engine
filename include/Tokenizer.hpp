#pragma once
#include "GGUF.hpp"
#include <array>
#include <map>
#include <memory>
#include <unordered_map>

// Qwen2/Qwen3 byte-level BPE. The GGUF object must outlive this object.
class Tokenizer {
public:
    explicit Tokenizer(const GGUF& model);
    Tokenizer(GGUF&&) = delete;
    std::vector<uint32_t> encode(const std::string& text, bool allow_special = false) const;
    // Returns UTF-8 bytes; incomplete token sequences may contain incomplete UTF-8.
    std::string decode(uint32_t id, bool skip_special = true) const;
    std::string decode(const std::vector<uint32_t>& ids, bool skip_special = true) const;
private:
    const std::vector<std::string>& tokens;
    const std::vector<std::string>& merges;
    const std::vector<int32_t>& token_types;
    std::string tokenizer_model;
    std::string pre_tokenizer;
    std::optional<uint32_t> bos_token_id, eos_token_id;
    bool add_bos_token, add_eos_token;
    std::unordered_map<std::string, uint32_t> token_map;
    std::map<std::pair<std::string, std::string>, size_t> merges_map;
    std::vector<uint32_t> reserved_ids;
    std::vector<bool> special_ids;
    std::array<std::string, 256> byte_encoder;
    std::unordered_map<std::string, unsigned char> byte_decoder;
    std::shared_ptr<void> pattern;
    void build_lookups();
    std::vector<uint32_t> encode_ordinary(const std::string& text) const;
    std::vector<uint32_t> encode_with_special_tokens(const std::string& text) const;
    std::vector<std::string> pre_tokenize(const std::string& text) const;
    std::string encode_bytes(const std::string& text) const;
    std::string decode_bytes(const std::string& text) const;
    std::vector<uint32_t> apply_bpe(const std::string& encoded_chunk) const;
    bool is_special_token(uint32_t id) const;
    void add_sequence_tokens(std::vector<uint32_t>& ids) const;
};
