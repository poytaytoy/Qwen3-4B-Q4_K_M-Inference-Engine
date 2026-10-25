#pragma once
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace test_reference {

inline std::vector<double> attention(const std::vector<float>& q, const std::vector<float>& k,
    const std::vector<float>& v, size_t queries, size_t prefix,
    size_t heads, size_t kv_heads, size_t dim) {
    const size_t key_count = prefix + queries;
    std::vector<double> result(q.size());

    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_cpu_init(), ggml_backend_free);

    if (!backend) {
        throw std::runtime_error("Could not initialize GGML CPU reference backend");
    }

    ggml_backend_cpu_set_n_threads(backend.get(), 1);

    for (size_t head = 0; head < heads; ++head) {
        const size_t kv_head = head / (heads / kv_heads);
        const ggml_init_params params{ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);

        if (!context) {
            throw std::runtime_error("Could not allocate GGML reference context");
        }

        auto* ctx = context.get();
        auto* query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, queries);
        auto* key = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, key_count);
        auto* value = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, key_count);

        // Separate GGML operations; no fused attention or handwritten softmax.
        auto* scores = ggml_mul_mat(ctx, key, query);
        auto* scaled = ggml_scale(ctx, scores, 1.0f / std::sqrt(static_cast<float>(dim)));
        auto* masked = ggml_diag_mask_inf(ctx, scaled, static_cast<int>(prefix));
        auto* probabilities = ggml_soft_max(ctx, masked);
        auto* value_transposed = ggml_cont(ctx, ggml_transpose(ctx, value));
        auto* output = ggml_mul_mat(ctx, value_transposed, probabilities);

        auto* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);

        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())), ggml_gallocr_free);

        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("Could not allocate GGML reference graph");
        }

        // Extract the selected heads from the engine's token-major layout.
        auto copy_head = [dim](const std::vector<float>& input, size_t rows,
            size_t head_count, size_t selected_head, ggml_tensor* tensor) {
            for (size_t row = 0; row < rows; ++row) {
                const auto* source = input.data() + (row * head_count + selected_head) * dim;
                ggml_backend_tensor_set(tensor, source, row * dim * sizeof(float), dim * sizeof(float));
            }
        };

        copy_head(q, queries, heads, head, query);
        copy_head(k, key_count, kv_heads, kv_head, key);
        copy_head(v, key_count, kv_heads, kv_head, value);

        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("GGML reference attention failed");
        }

        std::vector<float> head_output(queries * dim);
        ggml_backend_tensor_get(output, head_output.data(), 0, head_output.size() * sizeof(float));

        for (size_t row = 0; row < queries; ++row) {
            for (size_t d = 0; d < dim; ++d) {
                result[(row * heads + head) * dim + d] = head_output[row * dim + d];
            }
        }
    }

    return result;
}

} // namespace test_reference
