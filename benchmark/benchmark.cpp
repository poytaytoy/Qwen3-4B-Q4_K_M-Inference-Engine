#include "Engine.hpp"
#include <cuda_runtime.h>
#include <charconv>
#include <chrono>
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
        if (argc > 4) throw std::invalid_argument("Usage: benchmark [tokens=8] [runs=3] [prompt]");
        const size_t tokens = argc > 1 ? positive_count(argv[1]) : 8;
        const size_t runs = argc > 2 ? positive_count(argv[2]) : 3;
        const std::vector<ChatMessage> messages{{"user", argc > 3 ? argv[3] : "Explain what a GPU does in a few sentences."}};

        std::cout << std::fixed << std::setprecision(3);
        const double startup = elapsed_time([] { Engine::start(); synchronize_gpu(); });
        Engine& engine = Engine::start();
        std::cout << "Model load: " << startup << " s" << std::endl;
        engine.run(messages, false, 1, false, false);
        synchronize_gpu();

        double total_seconds = 0.0;
        for (size_t i = 0; i < runs; ++i) {
            // Prepare the prompt without generating tokens; reset for each run.
            engine.run(messages, false, 0, false, false);

            for (size_t token = 0; token < tokens; ++token) {
                const size_t input_tokens = engine.get_session_size();
                uint32_t next;
                synchronize_gpu();
                const double seconds = elapsed_time([&] {
                    next = engine.inference();
                    synchronize_gpu();
                });
                total_seconds += seconds;
                engine.append_token(next);

                std::cout << "Run " << i + 1 << ", token " << token + 1
                          << " (" << input_tokens << " input tokens): "
                          << seconds << " s | " << 1.0 / seconds << " tokens/s" << std::endl;
            }
        }
        std::cout << "Speed: " << static_cast<double>(tokens) * static_cast<double>(runs) / total_seconds
                  << " tokens/s\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
