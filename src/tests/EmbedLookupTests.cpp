#include "TestModel.hpp"
#include "EmbedLookup.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <stdexcept>

TEST_CASE("EmbedLookup validates model dimensions and token IDs", "[embedding]") {
    GGUF model(test_model_path);
    EmbedLookup lookup(model);
    REQUIRE(lookup.get_embedding_size() == model.get_embedding_length().value());
    REQUIRE(lookup.get_vocabulary_size() == model.get_tokens().size());
    const auto empty = lookup.lookup({});
    REQUIRE(empty.bytes == 0);
    REQUIRE(empty.data == nullptr);
    REQUIRE_THROWS_AS(lookup.lookup({std::numeric_limits<uint32_t>::max()}), std::out_of_range);
}

TEST_CASE("EmbedLookup returns ordered embeddings matching GGML dequantization",
          "[embedding][gpu]") {
    GGUF model(test_model_path);
    EmbedLookup lookup(model);
    const auto table = model.read_tensor_data("token_embd.weight");
    const auto *traits = ggml_get_type_traits(lookup.get_type());
    REQUIRE(traits->to_float != nullptr);

    // Enough rows to exercise the parallel lookup path; repeat IDs out of order.
    std::vector<uint32_t> ids;

    for (size_t i = 0; i < 32; ++i) {
        ids.push_back(i % 3 == 0 ? 42 : i % 3 == 1 ? 7 : 42);
    }
    const auto device = lookup.lookup(ids);
    const auto width = lookup.get_embedding_size();
    REQUIRE(device.bytes == ids.size() * width * sizeof(float));

    std::vector<float> actual(ids.size() * width), expected(width);

    REQUIRE(cudaMemcpy(actual.data(), device.data, device.bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    for (size_t row = 0; row < ids.size(); ++row) {
        traits->to_float(table.data() + ids[row] * lookup.get_row_bytes(), expected.data(), width);

        for (size_t column = 0; column < width; ++column) {
            REQUIRE(std::isfinite(actual[row * width + column]));
            REQUIRE(actual[row * width + column] == Catch::Approx(expected[column]).margin(1e-6));
        }
    }
}
