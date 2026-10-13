#include "Engine.hpp"
#include <stdexcept>

Engine::Engine(const std::filesystem::path& file)
    : gguf(file, true), tokenizer(gguf), chat_formatter(gguf) {
    for (const auto& tensor : gguf.get_tensors())
        weights.try_emplace(tensor.name, gguf, tensor.name);

    const auto eos = gguf.get_eos_token_id();
    if (!eos || *eos >= gguf.get_tokens().size())
        throw std::invalid_argument("Model requires a valid EOS token");
    eos_token = static_cast<int32_t>(*eos);
}

int32_t Engine::simulate_inference() {
    if (std::uniform_int_distribution<int>(0, 63)(random) == 0)
        return eos_token;
    // Sample every other vocabulary ID uniformly, keeping EOS at exactly 1/64.
    int32_t token = std::uniform_int_distribution<int32_t>(
        0, static_cast<int32_t>(gguf.get_tokens().size()) - 2)(random);
    return token >= eos_token ? token + 1 : token;
}

std::string Engine::run(const std::vector<ChatMessage>& messages,
    bool enable_thinking, size_t max_new_tokens) {
    const auto prompt = chat_formatter.render(messages, enable_thinking);
    chat_session = tokenizer.encode(prompt, true);

    std::vector<int32_t> generated;
    for (size_t step = 0; step < max_new_tokens; ++step) {
        const int32_t next = simulate_inference();
        chat_session.push_back(next);
        
        if (next == eos_token) break;

        generated.push_back(next);
    }

    return tokenizer.decode(generated);
}
