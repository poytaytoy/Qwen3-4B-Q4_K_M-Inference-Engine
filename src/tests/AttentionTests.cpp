#include "reference/AttentionReference.hpp"
#include "Kernels.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

TEST_CASE("Online attention matches CPU causal softmax with cached prefixes", "[attention][gpu]") {
    struct Shape {
        const char *name;
        size_t queries;
        size_t prefix;
        size_t heads;
        size_t kv_heads;
        size_t dim;
    };
    const Shape cases[] = {
        {"single value", 1, 0, 1, 1, 1},
        {"prefill with grouped query heads", 35, 0, 4, 2, 128},
        {"cached prefix with partial warp", 3, 31, 4, 1, 33},
        {"single token decode", 1, 64, 4, 2, 256},
        {"head dimension across tile boundary", 2, 32, 2, 2, 129},
    };

    for (const auto &shape : cases) {
        DYNAMIC_SECTION(shape.name) {
            const auto [name, queries, prefix, heads, kv_heads, dim] = shape;
            CAPTURE(queries, prefix, heads, kv_heads, dim);
            const size_t cached = prefix + queries;
            std::vector<float> q(queries * heads * dim);
            std::vector<float> k(cached * kv_heads * dim);
            std::vector<float> v(k.size());

            for (size_t i = 0; i < q.size(); ++i) {
                q[i] = std::sin(float(i) * 0.13f) * 2.0f;
            }

            for (size_t i = 0; i < k.size(); ++i) {
                k[i] = std::cos(float(i) * 0.07f) * 9.0f;
                v[i] = std::sin(float(i) * 0.11f);
            }

            const auto expected =
                test_reference::attention(q, k, v, queries, prefix, heads, kv_heads, dim);

            DeviceBuffer dq(q.data(), q.size() * sizeof(float));
            DeviceBuffer dk(k.data(), k.size() * sizeof(float));
            DeviceBuffer dv(v.data(), v.size() * sizeof(float));

            std::vector<float> actual(q.size() + 2, 123.0f);
            DeviceBuffer output(actual.data(), actual.size() * sizeof(float));

            // Reusing output must overwrite it, including nonzero initial contents.

            for (int repeat = 0; repeat < 2; ++repeat) {
                kernels::attention(
                    static_cast<const float *>(dq.data), static_cast<const float *>(dk.data),
                    static_cast<const float *>(dv.data), static_cast<float *>(output.data) + 1,
                    queries, heads, kv_heads, dim, prefix);

                REQUIRE(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost) == cudaSuccess);
                REQUIRE(actual.front() == 123.0f);
                REQUIRE(actual.back() == 123.0f);

                for (size_t i = 0; i < expected.size(); ++i) {
                    CAPTURE(i);
                    REQUIRE(actual[i + 1] == Catch::Approx(expected[i]).margin(3e-5));
                }
            }
        }
    }
}
