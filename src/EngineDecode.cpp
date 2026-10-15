// Capture and replay a complete one-token forward pass using per-session storage.
#include "Engine.hpp"
#include "Kernels.hpp"
#include <format>

namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA decode graph failed: ") + cudaGetErrorString(error));
}
float* fp(DeviceBuffer& buffer) { return static_cast<float*>(buffer.data); }
} // namespace

void Engine::enqueue_decode(ChatSession& session, DecodeGraph& graph) {
    using namespace kernels;
    const auto stream = graph.stream.get();
    const auto* metadata = static_cast<const DecodeToken*>(graph.metadata.data);
    const size_t width = embedlookup.get_embedding_size();
    const size_t heads = gguf.get_attention_head_count().value();
    const size_t kv_heads = gguf.get_value<uint32_t>("qwen3.attention.head_count_kv").value();
    const size_t head_dim = gguf.get_value<uint32_t>("qwen3.attention.key_length").value();
    const size_t ffn_width = gguf.get_value<uint32_t>("qwen3.feed_forward_length").value();
    const float epsilon = gguf.get_value<float>("qwen3.attention.layer_norm_rms_epsilon").value();
    const float rope_base = gguf.get_value<float>("qwen3.rope.freq_base").value();
    auto project = [&](const DeviceBuffer& input, const Weight& weight, DeviceBuffer& output) {
        v2::matmul_with_workspace(static_cast<const float*>(input.data), weight, fp(output),
            1, graph.quantization.data, stream);
    };

    embedding(weights.at("token_embd.weight"), metadata, fp(graph.hidden), width, stream);
    for (size_t layer = 0; layer < gguf.get_layer_count().value(); ++layer) {
        const auto prefix = std::format("blk.{}.", layer);
        auto weight = [&](const char* name) -> const Weight& { return weights.at(prefix + name); };
        auto* keys = static_cast<float*>(session.cache.at(prefix + "key").buffer.data);
        auto* values = static_cast<float*>(session.cache.at(prefix + "value").buffer.data);

        rms_norm(fp(graph.hidden), weight("attn_norm.weight"), fp(graph.normalized), 1, width, epsilon, stream);
        project(graph.normalized, weight("attn_q.weight"), graph.query);
        project(graph.normalized, weight("attn_k.weight"), graph.key);
        project(graph.normalized, weight("attn_v.weight"), graph.value);
        rms_norm(fp(graph.query), weight("attn_q_norm.weight"), fp(graph.query_normalized), heads, head_dim, epsilon, stream);
        rms_norm(fp(graph.key), weight("attn_k_norm.weight"), fp(graph.key_normalized), kv_heads, head_dim, epsilon, stream);
        rope(fp(graph.query_normalized), 1, heads, head_dim, 0, rope_base, stream, metadata);
        rope(fp(graph.key_normalized), 1, kv_heads, head_dim, 0, rope_base, stream, metadata);
        // A captured host pointer cannot advance per token; this kernel reads the
        // current position on-device and writes the appropriate cache rows.
        append_kv(fp(graph.key_normalized), fp(graph.value), keys, values, kv_heads * head_dim, metadata, stream);
        attention(fp(graph.query_normalized), keys, values, fp(graph.attended),
            1, heads, kv_heads, head_dim, 0, stream, metadata);
        project(graph.attended, weight("attn_output.weight"), graph.branch);
        residual_add(fp(graph.hidden), fp(graph.branch), width, stream);

        rms_norm(fp(graph.hidden), weight("ffn_norm.weight"), fp(graph.normalized), 1, width, epsilon, stream);
        project(graph.normalized, weight("ffn_gate.weight"), graph.gate);
        project(graph.normalized, weight("ffn_up.weight"), graph.up);
        silu_multiply(fp(graph.gate), fp(graph.up), fp(graph.activated), ffn_width, stream);
        project(graph.activated, weight("ffn_down.weight"), graph.branch);
        residual_add(fp(graph.hidden), fp(graph.branch), width, stream);
    }
    rms_norm(fp(graph.hidden), weights.at("output_norm.weight"), fp(graph.normalized), 1, width, epsilon, stream);
    project(graph.normalized, weights.at("token_embd.weight"), graph.logits);
    argmax_device(fp(graph.logits), static_cast<unsigned long long*>(graph.winner.data),
        embedlookup.get_vocabulary_size(), stream);
}

uint32_t Engine::decode(ChatSession& session) {
    const size_t width = embedlookup.get_embedding_size();
    const size_t heads = gguf.get_attention_head_count().value();
    const size_t kv_heads = gguf.get_value<uint32_t>("qwen3.attention.head_count_kv").value();
    const size_t head_dim = gguf.get_value<uint32_t>("qwen3.attention.key_length").value();
    const size_t ffn_width = gguf.get_value<uint32_t>("qwen3.feed_forward_length").value();
    if (session.tokens.back() >= embedlookup.get_vocabulary_size())
        throw std::out_of_range("Token ID out of range");

    // Ensure allocations exist before capture and rebuild if a cache was replaced.
    bool rebuild = !session.decode_graph;
    for (size_t layer = 0; layer < gguf.get_layer_count().value(); ++layer) {
        for (const auto* kind : {"key", "value"}) {
            const auto name = std::format("blk.{}.{}", layer, kind);
            auto* pointer = session.update_cache(name, kv_heads * head_dim);
            session.cache_write_ptr(name, 1); // Validate the actual buffer capacity.
            if (session.decode_graph) {
                const auto found = session.decode_graph->cache_addresses.find(name);
                rebuild |= found == session.decode_graph->cache_addresses.end() || found->second != pointer;
            }
        }
    }
    if (rebuild) {
        session.decode_graph = std::make_unique<DecodeGraph>(width, heads * head_dim,
            kv_heads * head_dim, ffn_width, embedlookup.get_vocabulary_size());
        for (const auto& [name, entry] : session.cache)
            session.decode_graph->cache_addresses.emplace(name, entry.buffer.data);
    }
    auto& graph = *session.decode_graph;
    const auto stream = graph.stream.get();
    const kernels::DecodeToken token{session.tokens.back(), session.cached_tokens};
    check(cudaMemcpyAsync(graph.metadata.data, &token, sizeof(token), cudaMemcpyHostToDevice, stream));

    if (!graph.executable) {
        // Warm lazy CUDA module loading outside capture. Replaying the same token
        // overwrites the same cache row and leaves all preceding positions intact.
        enqueue_decode(session, graph);
        graph.stream.synchronize();
        cudaGraph_t captured = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        try {
            enqueue_decode(session, graph);
            check(cudaStreamEndCapture(stream, &captured));
            check(cudaGraphInstantiate(&graph.executable, captured, 0));
        } catch (...) {
            cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(stream, &status);
            if (status != cudaStreamCaptureStatusNone) {
                cudaGraph_t abandoned = nullptr;
                cudaStreamEndCapture(stream, &abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
            }
            if (captured) cudaGraphDestroy(captured);
            throw;
        }
        check(cudaGraphDestroy(captured));
    }

    check(cudaGraphLaunch(graph.executable, stream));
    unsigned long long winner;
    check(cudaMemcpyAsync(&winner, graph.winner.data, sizeof(winner), cudaMemcpyDeviceToHost, stream));
    graph.stream.synchronize();
    ++session.cached_tokens;
    return ~static_cast<uint32_t>(winner);
}
