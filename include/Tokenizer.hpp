#pragma once
#include "GGUF.hpp"
#include <memory>
#include <string>
#include <vector>

class Tokenizer {
  public:
    explicit Tokenizer(const GGUF &model);
    ~Tokenizer();

    Tokenizer(GGUF &&) = delete;
    Tokenizer(const Tokenizer &) = delete;
    Tokenizer &operator=(const Tokenizer &) = delete;

    std::vector<uint32_t> encode(const std::string &text, bool allow_special = false) const;

    // Preserve raw bytes so individual tokens can be streamed, including partial UTF-8.
    std::string decode(uint32_t id, bool skip_special = true) const;
    std::string decode(const std::vector<uint32_t> &ids, bool skip_special = true) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
