#include "Engine.hpp"
#include "Kernels.hpp"
#include <format>
#include <stdexcept>

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

uint32_t Engine::inference() {
    using namespace kernels;

    if (chat_session.empty()) throw std::invalid_argument("Inference requires tokens");

    const size_t tokens = chat_session.size();
    const size_t width = embedlookup.get_embedding_size();
    const size_t heads = gguf.get_attention_head_count().value();
    const size_t kv_heads = gguf.get_value<uint32_t>("qwen3.attention.head_count_kv").value();
    const size_t head_dim = gguf.get_value<uint32_t>("qwen3.attention.key_length").value();
    const size_t ffn_width = gguf.get_value<uint32_t>("qwen3.feed_forward_length").value();
    const float epsilon = gguf.get_value<float>("qwen3.attention.layer_norm_rms_epsilon").value();
    const float rope_base = gguf.get_value<float>("qwen3.rope.freq_base").value();

    DeviceBuffer token_buffer{embedlookup.lookup(chat_session)};
    DeviceBuffer normalized(tokens * width * sizeof(float));
    DeviceBuffer q(tokens * heads * head_dim * sizeof(float));
    DeviceBuffer k(tokens * kv_heads * head_dim * sizeof(float));
    DeviceBuffer v(tokens * kv_heads * head_dim * sizeof(float));
    DeviceBuffer q_normalized(tokens * heads * head_dim * sizeof(float));
    DeviceBuffer k_normalized(tokens * kv_heads * head_dim * sizeof(float));
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

        rms_norm(fp(token_buffer), weight("attn_norm.weight"), fp(normalized), tokens, width, epsilon);
        matmul(fp(normalized), weight("attn_q.weight"), fp(q), tokens);
        matmul(fp(normalized), weight("attn_k.weight"), fp(k), tokens);
        matmul(fp(normalized), weight("attn_v.weight"), fp(v), tokens);

        rms_norm(fp(q), weight("attn_q_norm.weight"), fp(q_normalized), tokens * heads, head_dim, epsilon);
        rms_norm(fp(k), weight("attn_k_norm.weight"), fp(k_normalized), tokens * kv_heads, head_dim, epsilon);
        rope(fp(q_normalized), tokens, heads, head_dim, 0, rope_base);
        rope(fp(k_normalized), tokens, kv_heads, head_dim, 0, rope_base);
        attention(fp(q_normalized), fp(k_normalized), fp(v), fp(attended), tokens, heads, kv_heads, head_dim);
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
    
    return argmax(fp(logits), embedlookup.get_vocabulary_size());
}


std::string Engine::run(const std::vector<ChatMessage>& messages,
    bool enable_thinking, size_t max_new_tokens) {
    const auto prompt = chat_formatter.render(messages, enable_thinking);
    chat_session = tokenizer.encode(prompt, true);

    std::vector<uint32_t> generated;
    for (size_t step = 0; step < max_new_tokens; ++step) {
        const uint32_t next = inference();
        chat_session.push_back(next);
        
        if (next == eos_token) break;

        std::cout << tokenizer.decode(next) << std::flush; 

        generated.push_back(next);
    }

    return tokenizer.decode(generated);
}
