#pragma once
#include "GGUF.hpp"
#include "Tokenizer.hpp"
#include "ChatFormatter.hpp"
#include "Weight.hpp"
#include <random>

class Engine {
public:
    explicit Engine(const std::filesystem::path& file);
    std::string run(const std::vector<ChatMessage>& messages,
        bool enable_thinking = false, size_t max_new_tokens = 512);
private:
    GGUF gguf;
    Tokenizer tokenizer;
    ChatFormatter chat_formatter;
    std::map<std::string, Weight> weights;
    std::vector<int32_t> chat_session;
    int32_t eos_token;
    std::mt19937 random{std::random_device{}()};
    int32_t simulate_inference();
};
