#include "Kernels.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

TEST_CASE("Online attention matches CPU causal softmax with cached prefixes", "[attention][gpu]") {
    // Queries, cached prefix, query heads, KV heads, head dimension.
    for (auto shape : {std::array<size_t, 5>{1, 0, 1, 1, 1},
                      {35, 0, 4, 2, 128}, {3, 31, 4, 1, 33},
                      {1, 64, 4, 2, 256}, {2, 32, 2, 2, 129}}) {
        const auto [queries, prefix, heads, kv_heads, dim] = shape;
        CAPTURE(queries, prefix, heads, kv_heads, dim);
        const size_t cached = prefix + queries;
        std::vector<float> q(queries * heads * dim), k(cached * kv_heads * dim), v(k.size());
        for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(float(i) * 0.13f) * 2.0f;
        for (size_t i = 0; i < k.size(); ++i) {
            k[i] = std::cos(float(i) * 0.07f) * 9.0f;
            v[i] = std::sin(float(i) * 0.11f);
        }
        DeviceBuffer dq(q.data(), q.size() * sizeof(float));
        DeviceBuffer dk(k.data(), k.size() * sizeof(float));
        DeviceBuffer dv(v.data(), v.size() * sizeof(float));
        std::vector<float> actual(q.size() + 2, 123.0f);
        DeviceBuffer output(actual.data(), actual.size() * sizeof(float));
        // Reusing output must overwrite it, including nonzero initial contents.
        for (int repeat = 0; repeat < 2; ++repeat) {
            kernels::attention(static_cast<const float*>(dq.data), static_cast<const float*>(dk.data),
                static_cast<const float*>(dv.data), static_cast<float*>(output.data) + 1,
                queries, heads, kv_heads, dim, prefix);
            REQUIRE(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
            REQUIRE(actual.front() == 123.0f);
            REQUIRE(actual.back() == 123.0f);
            for (size_t row = 0; row < queries; ++row) {
                for (size_t head = 0; head < heads; ++head) {
                    const size_t kv_head = head / (heads / kv_heads);
                    std::vector<double> weights(prefix + row + 1);
                    double maximum = -INFINITY, denominator = 0;
                    for (size_t token = 0; token < weights.size(); ++token) {
                        for (size_t d = 0; d < dim; ++d)
                            weights[token] += double(q[(row * heads + head) * dim + d]) *
                                k[(token * kv_heads + kv_head) * dim + d];
                        weights[token] /= std::sqrt(double(dim));
                        maximum = std::max(maximum, weights[token]);
                    }
                    for (auto& weight : weights) {
                        weight = std::exp(weight - maximum);
                        denominator += weight;
                    }
                    for (size_t d = 0; d < dim; ++d) {
                        double expected = 0;
                        for (size_t token = 0; token < weights.size(); ++token)
                            expected += weights[token] * v[(token * kv_heads + kv_head) * dim + d] / denominator;
                        REQUIRE(actual[1 + (row * heads + head) * dim + d] ==
                            Catch::Approx(expected).margin(3e-5));
                    }
                }
            }
        }
    }
}
