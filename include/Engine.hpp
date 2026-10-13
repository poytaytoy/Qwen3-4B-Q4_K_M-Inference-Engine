#pragma once
#include "EmbedLookup.hpp"
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
    EmbedLookup embedlookup; 
    
    std::map<std::string, Weight> weights;
    std::vector<uint32_t> chat_session;
    uint32_t eos_token;
    std::mt19937 random{std::random_device{}()};
    uint32_t simulate_inference();
    uint32_t inference(); 
};
