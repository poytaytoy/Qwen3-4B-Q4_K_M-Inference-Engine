#include "TestModel.hpp"
#include "Tokenizer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>

TEST_CASE("Tokenizer round trips ordinary text and empty input", "[tokenizer]") {
    GGUF model(test_model_path);
    Tokenizer tokenizer(model);
    const std::string text = "Hello, café! 世界\nThis preserves token ordering.";
    const auto ids = tokenizer.encode(text);
    REQUIRE(ids.size() > 1);

    for (int repeat = 0; repeat < 8; ++repeat) {
        REQUIRE(tokenizer.decode(ids) == text);
    }
    REQUIRE(tokenizer.encode("").empty());
    REQUIRE(tokenizer.decode(std::vector<uint32_t>{}).empty());
}

TEST_CASE("Tokenizer handles special tokens and invalid IDs", "[tokenizer]") {
    GGUF model(test_model_path);
    Tokenizer tokenizer(model);
    const auto eos = model.get_eos_token_id();
    REQUIRE(eos.has_value());
    const auto &special = model.get_tokens().at(*eos);
    REQUIRE(tokenizer.encode(special, true) == std::vector<uint32_t>{*eos});
    REQUIRE(tokenizer.encode(special, false) != std::vector<uint32_t>{*eos});
    REQUIRE(tokenizer.decode(*eos, false) == special);
    REQUIRE(tokenizer.decode(*eos, true).empty());
    auto ids = tokenizer.encode("hello");
    ids.push_back(*eos);
    REQUIRE(tokenizer.decode(ids, true) == "hello");
    REQUIRE(tokenizer.decode(ids, false) == "hello" + special);
    REQUIRE_THROWS_AS(tokenizer.decode(std::numeric_limits<uint32_t>::max()), std::out_of_range);
    REQUIRE_THROWS_AS(
        tokenizer.decode(std::vector<uint32_t>{0, std::numeric_limits<uint32_t>::max()}),
        std::out_of_range);
}

TEST_CASE("Tokenizer preserves Unicode bytes when streaming individual tokens", "[tokenizer]") {
    GGUF model(test_model_path);
    Tokenizer tokenizer(model);
    const std::string text = "世界 😀 café مرحبا";
    const auto ids = tokenizer.encode(text);
    std::string streamed;

    for (auto id : ids) {
        streamed += tokenizer.decode(id);
    }

    REQUIRE(streamed == text);
    REQUIRE(tokenizer.decode(ids) == text);
    REQUIRE(tokenizer.encode("cafe\xcc\x81") == tokenizer.encode("café"));
}
