#pragma once
#include "EmbedLookup.hpp"
#include "GGUF.hpp"
#include "Tokenizer.hpp"
#include "ChatFormatter.hpp"
#include "ChatSession.hpp"
#include "Weight.hpp"
#include <random>

class Engine {
public:
    static Engine& start(); 
    std::string run(ChatSession& session,
        bool enable_thinking = false, size_t max_new_tokens = 512,
        bool stream_output = true, bool stop_at_eos = true);

    uint32_t simulate_inference();
    uint32_t inference(ChatSession& session);

    void append_token(ChatSession& session, uint32_t token);

    
private:
    static const char* const model_path;

    ~Engine() = default;

    explicit Engine(const std::filesystem::path& file = model_path);

    Engine(const Engine& other) = delete;
    Engine(Engine&& other) = delete;
    Engine& operator=(const Engine& other) = delete;
    Engine& operator=(Engine&& other) = delete;  

    GGUF gguf;
    Tokenizer tokenizer;
    ChatFormatter chat_formatter;
    EmbedLookup embedlookup; 
    
    std::map<std::string, Weight> weights;
    uint32_t eos_token;
    std::mt19937 random{std::random_device{}()}; 
};
