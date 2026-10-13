#pragma once
#include "GGUF.hpp"
#include <minja/minja.hpp>

struct ChatMessage {
    std::string role;
    std::string content;
};

// Renders the model's actual GGUF template. Does not require GPU weights.
class ChatFormatter {
public:
    explicit ChatFormatter(const GGUF& model);
    std::string render(const std::vector<ChatMessage>& messages,
        bool enable_thinking = false, bool add_generation_prompt = true) const;
private:
    std::shared_ptr<minja::TemplateNode> template_;
    std::string bos_token_, eos_token_;
};
