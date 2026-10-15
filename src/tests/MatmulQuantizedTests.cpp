#include "TestModel.hpp"
#include "Kernels.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

TEST_CASE("Packed integer matmul matches CPU Q8 activation reference", "[matmul][packed][gpu]") {
    GGUF model(test_model_path);
    for (const auto* name : {"blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.0.ffn_down.weight"}) {
        for (size_t rows : {size_t{1}, size_t{5}}) {
            DYNAMIC_SECTION(name << " rows=" << rows) {
                Weight weight(model, name);
                constexpr size_t columns = 19;
                const size_t width = weight.dimension[0];
                weight.dimension[1] = columns;
                const auto packed = model.read_tensor_data(name);
                const auto* traits = ggml_get_type_traits(weight.datatype);
                const auto row_bytes = ggml_row_size(weight.datatype, width);
                std::vector<float> input(rows * width), quantized(input.size()), decoded(width);
                for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * 0.031f) * 0.1f;
                // Independent CPU quantization, then GGML's weight dequantization.
                for (size_t begin = 0; begin < input.size(); begin += 32) {
                    float maximum = 0;
                    for (size_t i = 0; i < 32; ++i) maximum = std::max(maximum, std::abs(input[begin + i]));
                    const float scale = maximum / 127.0f;
                    for (size_t i = 0; i < 32; ++i)
                        quantized[begin + i] = maximum == 0 ? 0 : std::round((input[begin + i] / maximum) * 127.0f) * scale;
                }
                std::vector<double> expected(rows * columns);
                for (size_t column = 0; column < columns; ++column) {
                    traits->to_float(packed.data() + column * row_bytes, decoded.data(), width);
                    for (size_t row = 0; row < rows; ++row)
                        for (size_t k = 0; k < width; ++k)
                            expected[row * columns + column] += double(quantized[row * width + k]) * decoded[k];
                }
                DeviceBuffer device_input(input.data(), input.size() * sizeof(float));
                std::vector<float> actual(rows * columns + 2, 123.0f);
                DeviceBuffer output(actual.data(), actual.size() * sizeof(float));
                auto* product = static_cast<float*>(output.data) + 1;
                kernels::CudaStream stream;
                for (int repeat = 0; repeat < 2; ++repeat) {
                    kernels::v2::matmul(static_cast<const float*>(device_input.data), weight, product, rows, stream.get());
                    stream.synchronize();
                    REQUIRE(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
                    REQUIRE(actual.front() == 123.0f);
                    REQUIRE(actual.back() == 123.0f);
                    for (size_t i = 0; i < expected.size(); ++i)
                        REQUIRE(actual[i + 1] == Catch::Approx(expected[i]).margin(3e-5).epsilon(1e-4));
                }
                REQUIRE(cudaMemset(device_input.data, 0, input.size() * sizeof(float)) == cudaSuccess);
                kernels::v2::matmul(static_cast<const float*>(device_input.data), weight, product, rows);
                REQUIRE(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
                for (size_t i = 0; i < expected.size(); ++i) REQUIRE(actual[i + 1] == 0.0f);
            }
        }
    }
}
