#include "TestModel.hpp"
#include "ChatSession.hpp"
#include "EmbedLookup.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

TEST_CASE("Captured decode follows changing token IDs and cache positions", "[graph][gpu]") {
    GGUF model(test_model_path);
    Weight table(model, "token_embd.weight"), norm(model, "blk.0.attn_norm.weight");
    Weight qw(model, "blk.0.attn_q.weight"), kw(model, "blk.0.attn_k.weight"), vw(model, "blk.0.attn_v.weight");
    Weight qn(model, "blk.0.attn_q_norm.weight"), kn(model, "blk.0.attn_k_norm.weight");
    const size_t width = table.dimension[0], dim = qn.dimension[0];
    constexpr size_t heads = 4, kv_heads = 2, positions = 35;
    const size_t query_width = heads * dim, kv_width = kv_heads * dim;
    qw.dimension[1] = query_width;
    kw.dimension[1] = vw.dimension[1] = kv_width;
    const float epsilon = model.get_value<float>("qwen3.attention.layer_norm_rms_epsilon").value();
    const float base = model.get_value<float>("qwen3.rope.freq_base").value();
    ChatSession captured_cache, eager_cache;
    captured_cache.cache_capacity = eager_cache.cache_capacity = positions;
    float* captured_k = captured_cache.update_cache("key", kv_width);
    float* captured_v = captured_cache.update_cache("value", kv_width);
    float* eager_k = eager_cache.update_cache("key", kv_width);
    float* eager_v = eager_cache.update_cache("value", kv_width);
    DecodeGraph captured(width, query_width, kv_width, 1, query_width);
    DecodeGraph eager(width, query_width, kv_width, 1, query_width);
    auto fp = [](DeviceBuffer& b) { return static_cast<float*>(b.data); };
    auto enqueue = [&](DecodeGraph& g, float* keys, float* values, size_t position, bool dynamic) {
        const auto stream = g.stream.get();
        const auto* meta = static_cast<const kernels::DecodeToken*>(g.metadata.data);
        kernels::embedding(table, meta, fp(g.hidden), width, stream);
        kernels::rms_norm(fp(g.hidden), norm, fp(g.normalized), 1, width, epsilon, stream);
        kernels::v2::matmul_with_workspace(fp(g.normalized), qw, fp(g.query), 1, g.quantization.data, stream);
        kernels::v2::matmul_with_workspace(fp(g.normalized), kw, fp(g.key), 1, g.quantization.data, stream);
        kernels::v2::matmul_with_workspace(fp(g.normalized), vw, fp(g.value), 1, g.quantization.data, stream);
        kernels::rms_norm(fp(g.query), qn, fp(g.query_normalized), heads, dim, epsilon, stream);
        kernels::rms_norm(fp(g.key), kn, fp(g.key_normalized), kv_heads, dim, epsilon, stream);
        kernels::rope(fp(g.query_normalized), 1, heads, dim, position, base, stream, dynamic ? meta : nullptr);
        kernels::rope(fp(g.key_normalized), 1, kv_heads, dim, position, base, stream, dynamic ? meta : nullptr);
        if (dynamic) kernels::append_kv(fp(g.key_normalized), fp(g.value), keys, values, kv_width, meta, stream);
        else {
            REQUIRE(cudaMemcpyAsync(keys + position * kv_width, g.key_normalized.data,
                kv_width * sizeof(float), cudaMemcpyDeviceToDevice, stream) == cudaSuccess);
            REQUIRE(cudaMemcpyAsync(values + position * kv_width, g.value.data,
                kv_width * sizeof(float), cudaMemcpyDeviceToDevice, stream) == cudaSuccess);
        }
        kernels::attention(fp(g.query_normalized), keys, values, fp(g.attended),
            1, heads, kv_heads, dim, position, stream, dynamic ? meta : nullptr);
        kernels::argmax_device(fp(g.attended), static_cast<unsigned long long*>(g.winner.data), query_width, stream);
    };
    kernels::DecodeToken meta{42, 0};
    REQUIRE(cudaMemcpy(captured.metadata.data, &meta, sizeof(meta), cudaMemcpyHostToDevice) == cudaSuccess);
    enqueue(captured, captured_k, captured_v, 0, true); // Warm module loading outside capture.
    captured.stream.synchronize();
    cudaGraph_t graph = nullptr;
    REQUIRE(cudaStreamBeginCapture(captured.stream.get(), cudaStreamCaptureModeThreadLocal) == cudaSuccess);
    enqueue(captured, captured_k, captured_v, 0, true);
    REQUIRE(cudaStreamEndCapture(captured.stream.get(), &graph) == cudaSuccess);
    const auto status = cudaGraphInstantiate(&captured.executable, graph, 0);
    REQUIRE(cudaGraphDestroy(graph) == cudaSuccess);
    REQUIRE(status == cudaSuccess);

    EmbedLookup cpu_lookup(model);
    std::vector<float> actual(query_width), expected(query_width);
    for (size_t position = 0; position < positions; ++position) {
        CAPTURE(position);
        meta = {static_cast<uint32_t>(42 + position * 7), position};
        REQUIRE(cudaMemcpy(captured.metadata.data, &meta, sizeof(meta), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(eager.metadata.data, &meta, sizeof(meta), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaGraphLaunch(captured.executable, captured.stream.get()) == cudaSuccess);
        enqueue(eager, eager_k, eager_v, position, false);
        captured.stream.synchronize();
        eager.stream.synchronize();
        REQUIRE(cudaMemcpy(actual.data(), captured.attended.data, query_width * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(expected.data(), eager.attended.data, query_width * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        for (size_t i = 0; i < query_width; ++i) REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(1e-6));
        unsigned long long actual_id, expected_id;
        REQUIRE(cudaMemcpy(&actual_id, captured.winner.data, sizeof(actual_id), cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(&expected_id, eager.winner.data, sizeof(expected_id), cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(actual_id == expected_id);
        if (position == 0 || position == positions - 1) {
            auto reference = cpu_lookup.lookup({meta.token});
            std::vector<float> embedding(width), cpu(width);
            REQUIRE(cudaMemcpy(embedding.data(), captured.hidden.data, width * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
            REQUIRE(cudaMemcpy(cpu.data(), reference.data, width * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
            for (size_t i = 0; i < width; ++i) REQUIRE(embedding[i] == Catch::Approx(cpu[i]).margin(1e-6));
        }
    }
    // All earlier cache rows must survive later replays unchanged.
    std::vector<float> actual_cache(positions * kv_width), expected_cache(actual_cache.size());
    for (auto buffers : {std::pair{captured_k, eager_k}, std::pair{captured_v, eager_v}}) {
        REQUIRE(cudaMemcpy(actual_cache.data(), buffers.first, actual_cache.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(expected_cache.data(), buffers.second, expected_cache.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        for (size_t i = 0; i < actual_cache.size(); ++i)
            REQUIRE(actual_cache[i] == Catch::Approx(expected_cache[i]).margin(1e-6));
    }
}
