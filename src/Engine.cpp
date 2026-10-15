#include "Engine.hpp"
#include "Kernels.hpp"
#include <format>
#include <cstdlib>
#include <string_view>
#include <stdexcept>

const char* const Engine::model_path = INFERENCE_MODEL_PATH;

/*static*/ Engine& Engine::start(){
    static Engine engine; 

    return engine; 
};

Engine::Engine(const std::filesystem::path& file)
    : gguf(file, false), tokenizer(gguf), chat_formatter(gguf), embedlookup(gguf) {
    for (const auto& tensor : gguf.get_tensors())
        weights.try_emplace(tensor.name, gguf, tensor.name);
    const auto eos = gguf.get_eos_token_id();
    if (!eos || *eos >= gguf.get_tokens().size())
        throw std::invalid_argument("Model requires a valid EOS token");
    eos_token = static_cast<uint32_t>(*eos);
}

uint32_t Engine::simulate_inference() {
    if (std::uniform_int_distribution<int>(0, 63)(random) == 0)
        return eos_token;
    // Sample every other vocabulary ID uniformly, keeping EOS at exactly 1/64.
    uint32_t token = std::uniform_int_distribution<uint32_t>(
        0, static_cast<uint32_t>(gguf.get_tokens().size()) - 2)(random);
    return token >= eos_token ? token + 1 : token;
}

uint32_t Engine::inference(ChatSession& session) {
    using namespace kernels;

    if (session.tokens.empty()) throw std::invalid_argument("Inference requires tokens");

    const size_t offset = session.cached_tokens;
    if (session.tokens.size() > session.cache_capacity)
        throw std::length_error("Chat session cache capacity exceeded");
    if (offset >= session.tokens.size())
        throw std::invalid_argument("Inference requires uncached tokens");
    const size_t tokens = session.tokens.size() - offset;
    const char* graph_option = std::getenv("INFERENCE_CUDA_GRAPHS");
    const char* matmul_option = std::getenv("INFERENCE_MATMUL");
    const bool graphs = !graph_option || std::string_view(graph_option) != "0";
    const bool packed = !matmul_option || std::string_view(matmul_option) != "v1";
    if (tokens == 1 && graphs && packed) return decode(session);
    const size_t width = embedlookup.get_embedding_size();
    const size_t heads = gguf.get_attention_head_count().value();
    const size_t kv_heads = gguf.get_value<uint32_t>("qwen3.attention.head_count_kv").value();
    const size_t head_dim = gguf.get_value<uint32_t>("qwen3.attention.key_length").value();
    const size_t ffn_width = gguf.get_value<uint32_t>("qwen3.feed_forward_length").value();
    const float epsilon = gguf.get_value<float>("qwen3.attention.layer_norm_rms_epsilon").value();
    const float rope_base = gguf.get_value<float>("qwen3.rope.freq_base").value();

    const std::vector<uint32_t> pending(session.tokens.begin() + offset, session.tokens.end());
    DeviceBuffer token_buffer{embedlookup.lookup(pending)};
    DeviceBuffer normalized(tokens * width * sizeof(float));
    DeviceBuffer q(tokens * heads * head_dim * sizeof(float));
    DeviceBuffer k(tokens * kv_heads * head_dim * sizeof(float));
    DeviceBuffer q_normalized(tokens * heads * head_dim * sizeof(float));
    DeviceBuffer attended(tokens * heads * head_dim * sizeof(float));
    DeviceBuffer branch(tokens * width * sizeof(float));
    DeviceBuffer gate(tokens * ffn_width * sizeof(float));
    DeviceBuffer up(tokens * ffn_width * sizeof(float));
    DeviceBuffer activated(tokens * ffn_width * sizeof(float));
    auto fp = [](DeviceBuffer& buffer) { return static_cast<float*>(buffer.data); };

    // Qwen3-4B has 36 blocks; read the count from the model metadata.
    for (size_t i = 0; i < gguf.get_layer_count().value(); ++i) {
        const auto prefix = std::format("blk.{}.", i);
        auto weight = [&](const char* name) -> const Weight& { return weights.at(prefix + name); };

        const auto key_name = prefix + "key";
        const auto value_name = prefix + "value";
        float* cached_k = session.update_cache(key_name, kv_heads * head_dim);
        float* cached_v = session.update_cache(value_name, kv_heads * head_dim);
        float* new_k = session.cache_write_ptr(key_name, tokens);
        float* new_v = session.cache_write_ptr(value_name, tokens);

        rms_norm(fp(token_buffer), weight("attn_norm.weight"), fp(normalized), tokens, width, epsilon);
        matmul(fp(normalized), weight("attn_q.weight"), fp(q), tokens);
        matmul(fp(normalized), weight("attn_k.weight"), fp(k), tokens);
        matmul(fp(normalized), weight("attn_v.weight"), new_v, tokens);

        rms_norm(fp(q), weight("attn_q_norm.weight"), fp(q_normalized), tokens * heads, head_dim, epsilon);
        rms_norm(fp(k), weight("attn_k_norm.weight"), new_k, tokens * kv_heads, head_dim, epsilon);
        rope(fp(q_normalized), tokens, heads, head_dim, offset, rope_base);
        rope(new_k, tokens, kv_heads, head_dim, offset, rope_base);

        attention(fp(q_normalized), cached_k, cached_v, fp(attended), tokens, heads, kv_heads, head_dim, offset);
        
        matmul(fp(attended), weight("attn_output.weight"), fp(branch), tokens);
        residual_add(fp(token_buffer), fp(branch), tokens * width);

        rms_norm(fp(token_buffer), weight("ffn_norm.weight"), fp(normalized), tokens, width, epsilon);
        matmul(fp(normalized), weight("ffn_gate.weight"), fp(gate), tokens);
        matmul(fp(normalized), weight("ffn_up.weight"), fp(up), tokens);
        silu_multiply(fp(gate), fp(up), fp(activated), tokens * ffn_width);
        matmul(fp(activated), weight("ffn_down.weight"), fp(branch), tokens);
        residual_add(fp(token_buffer), fp(branch), tokens * width);
    }

    // Only the final token's state is needed to predict the next token.
    rms_norm(fp(token_buffer) + (tokens - 1) * width,
        weights.at("output_norm.weight"), fp(normalized), 1, width, epsilon);
    DeviceBuffer logits(embedlookup.get_vocabulary_size() * sizeof(float));
    matmul(fp(normalized), weights.at("token_embd.weight"), fp(logits), 1);
    
    const auto next = argmax(fp(logits), embedlookup.get_vocabulary_size());
    session.cached_tokens += tokens;
    return next;
}


void Engine::append_token(ChatSession& session, uint32_t token) {
    if (token >= embedlookup.get_vocabulary_size())
        throw std::out_of_range("Token ID out of range");
    session.tokens.push_back(token);
}

std::string Engine::run(ChatSession& session,
    bool enable_thinking, size_t max_new_tokens, bool stream_output, bool stop_at_eos) {
    const auto prompt = chat_formatter.render(session.messages, enable_thinking);
    auto prompt_tokens = tokenizer.encode(prompt, true);
    if (prompt_tokens.size() > session.cache_capacity ||
        max_new_tokens > session.cache_capacity - prompt_tokens.size())
        throw std::length_error("Prompt and generation exceed chat session cache capacity");
    session.tokens = std::move(prompt_tokens);
    session.cached_tokens = 0;

    std::vector<uint32_t> generated;
    for (size_t step = 0; step < max_new_tokens; ++step) {
        const uint32_t next = inference(session);
        session.tokens.push_back(next);
        
        if (stop_at_eos && next == eos_token) break;

        if (stream_output) std::cout << tokenizer.decode(next) << std::flush;

        generated.push_back(next);
    }

    return tokenizer.decode(generated);
}
