#include "ChatFormatter.hpp"
#include "TestModel.hpp"
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

TEST_CASE("ChatFormatter renders real messages and Qwen prompt options", "[chat]") {
    GGUF model(test_model_path);
    ChatFormatter formatter(model);
    const std::vector<ChatMessage> messages = {{"system", "Be concise."},
                                               {"user", "Hello there."},
                                               {"assistant", "Hi!"},
                                               {"user", "What is two plus two?"}};
    const auto prompt = formatter.render(messages);

    for (const auto &message : messages) {
        REQUIRE(prompt.find(message.content) != std::string::npos);
    }
    const auto without_generation = formatter.render(messages, false, false);
    REQUIRE(prompt.starts_with(without_generation));
    REQUIRE(prompt.size() > without_generation.size());
    REQUIRE(prompt.find("<|im_start|>assistant\n", without_generation.size()) != std::string::npos);
    REQUIRE(prompt.ends_with("<think>\n\n</think>\n\n"));
    const auto thinking = formatter.render(messages, true);
    REQUIRE(thinking.starts_with(without_generation));
    REQUIRE(thinking != prompt);
    REQUIRE(thinking.find("</think>", without_generation.size()) == std::string::npos);
}

TEST_CASE("ChatFormatter rejects empty chats and unsupported roles", "[chat]") {
    GGUF model(test_model_path);
    ChatFormatter formatter(model);
    REQUIRE_THROWS_AS(formatter.render({}), std::invalid_argument);
    REQUIRE_THROWS_AS(formatter.render({{"invalid", "hi"}}), std::invalid_argument);
}
