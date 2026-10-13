#include "ChatFormatter.hpp"
#include <stdexcept>

ChatFormatter::ChatFormatter(const GGUF& model) {
    const auto source = model.get_value<std::string>("tokenizer.chat_template");
    if (!source || source->empty())
        throw std::invalid_argument("GGUF has no string tokenizer.chat_template");
    const auto& tokens = model.get_tokens();
    auto token = [&](std::optional<uint32_t> id) -> std::string {
        if (!id) return {};
        if (*id >= tokens.size()) throw std::invalid_argument("Invalid chat-template token ID");
        return tokens[*id];
    };
    bos_token_ = token(model.get_bos_token_id());
    eos_token_ = token(model.get_eos_token_id());
    template_ = minja::Parser::parse(*source, {
        /* trim_blocks */ true, /* lstrip_blocks */ true, /* keep_trailing_newline */ false});
}

std::string ChatFormatter::render(const std::vector<ChatMessage>& messages,
    bool enable_thinking, bool add_generation_prompt) const {
    if (messages.empty()) throw std::invalid_argument("Chat requires at least one message");
    using Json = nlohmann::ordered_json;
    auto input = Json::array();
    for (const auto& message : messages) {
        if (message.role != "system" && message.role != "user" && message.role != "assistant")
            throw std::invalid_argument("Basic chat \ system, user and assistant roles");
        input.push_back({{"role", message.role}, {"content", message.content}});
    }
    auto context = minja::Context::make(minja::Value(Json{
        {"messages", input}, {"tools", Json::array()},
        {"add_generation_prompt", add_generation_prompt},
        {"enable_thinking", enable_thinking},
        {"bos_token", bos_token_}, {"eos_token", eos_token_}
    }));
    return template_->render(context);
}
