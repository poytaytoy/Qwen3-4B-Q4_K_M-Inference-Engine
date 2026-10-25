#pragma once
#include "Kernels.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace matmul_test {

inline std::vector<float> quantize_reference(const std::vector<float> &input) {
    auto result = input;

    for (size_t begin = 0; begin < input.size(); begin += 32) {
        float maximum = 0;

        for (size_t i = 0; i < 32; ++i) {
            maximum = std::max(maximum, std::abs(input[begin + i]));
        }
        const float scale = maximum / 127.0f;

        for (size_t i = 0; i < 32; ++i) {
            result[begin + i] =
                maximum == 0 ? 0 : std::round(input[begin + i] / maximum * 127.0f) * scale;
        }
    }

    return result;
}

inline std::vector<float> make_input(size_t count) {
    std::vector<float> input(count);

    for (size_t i = 0; i < count; ++i) {
        input[i] = std::sin(float(i) * 0.031f) * 0.1f;
    }

    return input;
}

// Independent CPU Q8 quantization and GGML weight decoding.
inline std::vector<double> reference_product(const GGUF &model, const char *name,
                                             const Weight &weight, const std::vector<float> &input,
                                             size_t rows, size_t columns) {
    const size_t width = weight.dimension[0];
    const auto packed = model.read_tensor_data(name);
    const auto *traits = ggml_get_type_traits(weight.datatype);
    const size_t row_bytes = ggml_row_size(weight.datatype, width);
    const auto quantized = quantize_reference(input);
    std::vector<float> decoded(width);
    std::vector<double> expected(rows * columns);

    for (size_t column = 0; column < columns; ++column) {
        traits->to_float(packed.data() + column * row_bytes, decoded.data(), width);

        for (size_t row = 0; row < rows; ++row) {
            for (size_t i = 0; i < width; ++i) {
                expected[row * columns + column] += double(quantized[row * width + i]) * decoded[i];
            }
        }
    }

    return expected;
}

inline void copy_to_host(const DeviceBuffer &source, std::vector<float> &destination) {
    REQUIRE(cudaMemcpy(destination.data(), source.data, destination.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
}

inline void require_product(const std::vector<float> &actual, const std::vector<double> &expected,
                            size_t offset = 0, double margin = 3e-5) {
    REQUIRE(actual.size() >= offset + expected.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i);
        REQUIRE(actual[offset + i] == Catch::Approx(expected[i]).margin(margin).epsilon(1e-4));
    }
}

} // namespace matmul_test
