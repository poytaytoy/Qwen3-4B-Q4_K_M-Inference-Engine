#include "Engine.hpp"
#include <cuda_runtime.h>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

void synchronize_gpu() {
    const auto error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

template<typename Function>
double elapsed_time(Function&& operation) {
    const auto start = std::chrono::steady_clock::now();
    std::invoke(std::forward<Function>(operation));
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

size_t positive_count(std::string_view argument) {
    size_t count = 0;
    const auto [end, error] = std::from_chars(argument.data(), argument.data() + argument.size(), count);
    if (error != std::errc{} || end != argument.data() + argument.size() || count == 0)
        throw std::invalid_argument("Token count and run count must be positive integers");
    return count;
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 4) throw std::invalid_argument("Usage: benchmark [tokens=32] [runs=3] [prompt]");
        const size_t tokens = argc > 1 ? positive_count(argv[1]) : 32;
        const size_t runs = argc > 2 ? positive_count(argv[2]) : 3;
        const std::vector<ChatMessage> messages{{"user", argc > 3 ? argv[3] : "Explain what a GPU does in a few sentences."}};

        ChatSession session;
        session.messages = messages;

        const char* matmul_option = std::getenv("INFERENCE_MATMUL");
        const char* graph_option = std::getenv("INFERENCE_CUDA_GRAPHS");
        const bool packed = !matmul_option || std::string_view(matmul_option) != "v1";
        const bool graphs = packed && (!graph_option || std::string_view(graph_option) != "0");
        std::cout << "Matmul: " << (packed ? "v2" : "v1")
                  << " | CUDA graphs: " << (graphs ? "on" : "off") << '\n';
        std::cout << std::fixed << std::setprecision(3);
        const double startup = elapsed_time([] { Engine::start(); synchronize_gpu(); });
        Engine& engine = Engine::start();
        std::cout << "Model load: " << startup << " s" << std::endl;
        engine.run(session, false, 2, false, false);
        synchronize_gpu();

        double total_seconds = 0.0, total_decode_seconds = 0.0;
        for (size_t i = 0; i < runs; ++i) {
            // Reset the prompt/cache per run; exclude tokenization from timing.
            engine.run(session, false, 0, false, false);
            const size_t input_tokens = session.tokens.size();
            if (input_tokens > session.cache_capacity || tokens > session.cache_capacity - input_tokens)
                throw std::invalid_argument("Prompt plus output tokens exceeds session capacity");
            synchronize_gpu();

            // The first prediction includes prompt prefill; later ones use the KV cache.
            const double first_seconds = elapsed_time([&] {
                engine.append_token(session, engine.inference(session));
                synchronize_gpu();
            });
            double decode_seconds = 0.0;
            if (tokens > 1) {
                decode_seconds = elapsed_time([&] {
                    for (size_t token = 1; token < tokens; ++token)
                        engine.append_token(session, engine.inference(session));
                    synchronize_gpu();
                });
            }
            const double seconds = first_seconds + decode_seconds;
            total_seconds += seconds;
            total_decode_seconds += decode_seconds;

            // Generate the requested count even if EOS occurs; no printing is timed.
            std::cout << "Run " << i + 1 << ": " << input_tokens << " prompt tokens, "
                      << tokens << " output tokens | " << seconds << " s | "
                      << tokens / seconds << " tokens/s | first token "
                      << first_seconds * 1000.0 << " ms";
            if (tokens > 1)
                std::cout << " | cached decode " << (tokens - 1) / decode_seconds << " tokens/s";
            std::cout << std::endl;
        }
        const double output_count = static_cast<double>(tokens) * static_cast<double>(runs);
        std::cout << "Average: " << output_count / total_seconds << " tokens/s | "
                  << total_seconds * 1000.0 / output_count << " ms/token across "
                  << runs << " runs (includes prefill)\n";
        if (tokens > 1)
            std::cout << "Average cached decode: "
                      << static_cast<double>(tokens - 1) * static_cast<double>(runs) / total_decode_seconds
                      << " tokens/s (excludes first token)\n";

    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
