#include "TestModel.hpp"
#include "Kernels.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <vector>
#include <utility>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<kernels::CudaStream>);
static_assert(std::is_nothrow_move_constructible_v<kernels::CudaStream>);

TEST_CASE("Matmul Q4_K and Q6_K match real model CPU reference and overwrite reused output", "[matmul][gpu]") {
    GGUF model(test_model_path);
    for (const auto* name : {"blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.0.ffn_down.weight"}) {
        DYNAMIC_SECTION(name) {
            Weight weight(model, name);
            REQUIRE((weight.datatype == GGML_TYPE_Q4_K || weight.datatype == GGML_TYPE_Q6_K));
            if (std::string(name) != "blk.0.ffn_down.weight")
                REQUIRE(weight.datatype == (std::string(name) == "blk.0.attn_q.weight"
                    ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K));
            // Use five real output columns; retain the model's full input width.
            const size_t width = weight.dimension[0];
            constexpr size_t columns = 5, rows = 3;
            REQUIRE(weight.dimension[1] >= columns);
            weight.dimension[1] = columns;
            const auto packed = model.read_tensor_data(name);
            const auto* traits = ggml_get_type_traits(weight.datatype);
            const size_t row_bytes = ggml_row_size(weight.datatype, width);
            std::vector<float> input(rows * width), dequantized(width);
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = std::sin(float(i) * 0.031f) * 0.1f;
            std::vector<double> expected(rows * columns);
            for (size_t column = 0; column < columns; ++column) {
                traits->to_float(packed.data() + column * row_bytes, dequantized.data(), width);
                for (size_t row = 0; row < rows; ++row)
                    for (size_t i = 0; i < width; ++i)
                        expected[row * columns + column] += double(input[row * width + i]) * dequantized[i];
            }
            DeviceBuffer device_input(input.data(), input.size() * sizeof(float));
            std::vector<float> actual(rows * columns, 123.0f);
            DeviceBuffer device_output(actual.data(), actual.size() * sizeof(float));
            for (int repeat = 0; repeat < 2; ++repeat) {
                kernels::v1::matmul(static_cast<const float*>(device_input.data), weight,
                    static_cast<float*>(device_output.data), rows);
                REQUIRE(cudaMemcpy(actual.data(), device_output.data, actual.size() * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess);
                for (size_t i = 0; i < actual.size(); ++i)
                    REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(1e-5).epsilon(1e-4));
            }
            // Independent outputs on nonblocking streams; explicitly join before reading.
            DeviceBuffer other_output(actual.data(), actual.size() * sizeof(float));
            kernels::CudaStream first, second;
            kernels::v1::matmul(static_cast<const float*>(device_input.data), weight,
                static_cast<float*>(device_output.data), rows, first.get());
            kernels::v1::matmul(static_cast<const float*>(device_input.data), weight,
                static_cast<float*>(other_output.data), rows, second.get());
            first.synchronize();
            second.synchronize();
            for (const auto* buffer : {&device_output, &other_output}) {
                REQUIRE(cudaMemcpy(actual.data(), buffer->data, actual.size() * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess);
                for (size_t i = 0; i < actual.size(); ++i)
                    REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(1e-5).epsilon(1e-4));
            }
            // A second call with different input must replace the previous product.
            REQUIRE(cudaMemset(device_input.data, 0, input.size() * sizeof(float)) == cudaSuccess);
            kernels::v1::matmul(static_cast<const float*>(device_input.data), weight,
                static_cast<float*>(device_output.data), 1);
            REQUIRE(cudaMemcpy(actual.data(), device_output.data, columns * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess);
            for (size_t i = 0; i < columns; ++i) REQUIRE(actual[i] == 0.0f);
        }
    }
}

TEST_CASE("CudaStream transfers ownership and rejects moved-from access", "[stream][gpu]") {
    kernels::CudaStream original;
    const auto handle = original.get();
    unsigned flags = 0;
    REQUIRE(cudaStreamGetFlags(handle, &flags) == cudaSuccess);
    REQUIRE((flags & cudaStreamNonBlocking) != 0);
    kernels::CudaStream moved(std::move(original));
    REQUIRE(moved.get() == handle);
    REQUIRE_THROWS_AS(original.get(), std::logic_error);
    REQUIRE_THROWS_AS(original.synchronize(), std::logic_error);
    kernels::CudaStream assigned;
    assigned = std::move(moved);
    REQUIRE(assigned.get() == handle);
    REQUIRE_THROWS_AS(moved.ready(), std::logic_error);
    assigned.synchronize();
    REQUIRE(assigned.ready());
}

TEST_CASE("Matmul dispatch matches real weights around the row threshold", "[matmul][tiled][gpu]") {
    GGUF model(test_model_path);
    for (const auto* name : {"blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.0.ffn_down.weight"}) {
        for (size_t rows : {size_t{4}, size_t{5}, size_t{101}}) {
            DYNAMIC_SECTION(name << " rows=" << rows) {
                Weight weight(model, name);
                const size_t width = weight.dimension[0];
                constexpr size_t columns = 19;
                REQUIRE(weight.dimension[1] >= columns);
                weight.dimension[1] = columns;
                const auto packed = model.read_tensor_data(name);
                const auto* traits = ggml_get_type_traits(weight.datatype);
                const size_t row_bytes = ggml_row_size(weight.datatype, width);
                std::vector<float> input(rows * width), decoded(width);
                for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * 0.031f) * 0.1f;
                std::vector<double> expected(rows * columns);
                for (size_t column = 0; column < columns; ++column) {
                    traits->to_float(packed.data() + column * row_bytes, decoded.data(), width);
                    for (size_t row = 0; row < rows; ++row)
                        for (size_t k = 0; k < width; ++k)
                            expected[row * columns + column] += double(input[row * width + k]) * decoded[k];
                }
                DeviceBuffer device_input(input.data(), input.size() * sizeof(float));
                std::vector<float> actual(rows * columns + 2, 123.0f);
                DeviceBuffer output(actual.data(), actual.size() * sizeof(float));
                float* product = static_cast<float*>(output.data) + 1;
                kernels::CudaStream stream;
                for (int repeat = 0; repeat < 2; ++repeat) {
                    kernels::v1::matmul(static_cast<const float*>(device_input.data), weight, product, rows, stream.get());
                    stream.synchronize();
                    REQUIRE(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
                    REQUIRE(actual.front() == 123.0f);
                    REQUIRE(actual.back() == 123.0f);
                    for (size_t i = 0; i < expected.size(); ++i)
                        REQUIRE(actual[i + 1] == Catch::Approx(expected[i]).margin(3e-5).epsilon(1e-4));
                }
                REQUIRE(cudaMemset(device_input.data, 0, input.size() * sizeof(float)) == cudaSuccess);
                kernels::v1::matmul(static_cast<const float*>(device_input.data), weight, product, rows);
                REQUIRE(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
                for (size_t i = 0; i < expected.size(); ++i) REQUIRE(actual[i + 1] == 0.0f);
            }
        }
    }
}
