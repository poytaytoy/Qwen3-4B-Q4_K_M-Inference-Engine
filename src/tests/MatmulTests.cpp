#include "reference/MatmulReference.hpp"
#include "TestModel.hpp"
#include "Kernels.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>
#include <type_traits>

TEST_CASE("Matmul Q4_K and Q6_K match real model CPU reference and overwrite reused output",
          "[matmul][gpu]") {
    GGUF model(test_model_path);

    for (const auto *name :
         {"blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.0.ffn_down.weight"}) {
        DYNAMIC_SECTION(name) {
            Weight weight(model, name);
            REQUIRE((weight.datatype == GGML_TYPE_Q4_K || weight.datatype == GGML_TYPE_Q6_K));

            if (std::string(name) != "blk.0.ffn_down.weight") {
                REQUIRE(
                    weight.datatype ==
                    (std::string(name) == "blk.0.attn_q.weight" ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K));
            }

            // Use five real output columns; retain the model's full input width.
            const size_t width = weight.dimension[0];
            constexpr size_t columns = 5;
            constexpr size_t rows = 3;
            REQUIRE(weight.dimension[1] >= columns);
            weight.dimension[1] = columns;

            const auto input = matmul_test::make_input(rows * width);

            const auto expected =
                matmul_test::reference_product(model, name, weight, input, rows, columns);

            DeviceBuffer device_input(input.data(), input.size() * sizeof(float));

            std::vector<float> actual(rows * columns, 123.0f);
            DeviceBuffer device_output(actual.data(), actual.size() * sizeof(float));

            for (int repeat = 0; repeat < 2; ++repeat) {
                kernels::matmul(static_cast<const float *>(device_input.data), weight,
                                static_cast<float *>(device_output.data), rows);
                matmul_test::copy_to_host(device_output, actual);
                matmul_test::require_product(actual, expected, 0, 1e-5);
            }

            // Independent outputs on nonblocking streams; explicitly join before reading.
            DeviceBuffer other_output(actual.data(), actual.size() * sizeof(float));

            kernels::CudaStream first, second;
            kernels::matmul(static_cast<const float *>(device_input.data), weight,
                            static_cast<float *>(device_output.data), rows, first);
            kernels::matmul(static_cast<const float *>(device_input.data), weight,
                            static_cast<float *>(other_output.data), rows, second);
            first.synchronize();
            second.synchronize();

            for (const auto *buffer : {&device_output, &other_output}) {
                matmul_test::copy_to_host(*buffer, actual);
                matmul_test::require_product(actual, expected, 0, 1e-5);
            }

            // A second call with different input must replace the previous product.

            REQUIRE(cudaMemset(device_input.data, 0, input.size() * sizeof(float)) == cudaSuccess);
            kernels::matmul(static_cast<const float *>(device_input.data), weight,
                            static_cast<float *>(device_output.data), 1);

            REQUIRE(cudaMemcpy(actual.data(), device_output.data, columns * sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess);

            for (size_t i = 0; i < columns; ++i) {
                REQUIRE(actual[i] == 0.0f);
            }
        }
    }
}

TEST_CASE("Matmul matches quantized CPU reference across row counts", "[matmul][gpu]") {
    GGUF model(test_model_path);

    for (const auto *name :
         {"blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.0.ffn_down.weight"}) {
        for (size_t rows : {size_t{1}, size_t{4}, size_t{5}, size_t{101}}) {
            DYNAMIC_SECTION(name << " rows=" << rows) {
                Weight weight(model, name);
                const size_t width = weight.dimension[0];
                constexpr size_t columns = 19;
                REQUIRE(weight.dimension[1] >= columns);
                weight.dimension[1] = columns;

                const auto input = matmul_test::make_input(rows * width);

                const auto expected =
                    matmul_test::reference_product(model, name, weight, input, rows, columns);

                DeviceBuffer device_input(input.data(), input.size() * sizeof(float));

                std::vector<float> actual(rows * columns + 2, 123.0f);
                DeviceBuffer output(actual.data(), actual.size() * sizeof(float));
                float *product = static_cast<float *>(output.data) + 1;

                kernels::CudaStream stream;

                for (int repeat = 0; repeat < 2; ++repeat) {
                    kernels::matmul(static_cast<const float *>(device_input.data), weight, product,
                                    rows, stream);
                    stream.synchronize();
                    matmul_test::copy_to_host(output, actual);
                    REQUIRE(actual.front() == 123.0f);
                    REQUIRE(actual.back() == 123.0f);
                    matmul_test::require_product(actual, expected, 1);
                }

                REQUIRE(cudaMemset(device_input.data, 0, input.size() * sizeof(float)) ==
                        cudaSuccess);
                kernels::matmul(static_cast<const float *>(device_input.data), weight, product,
                                rows);
                matmul_test::copy_to_host(output, actual);

                for (size_t i = 0; i < expected.size(); ++i) {
                    REQUIRE(actual[i + 1] == 0.0f);
                }
            }
        }
    }
}
