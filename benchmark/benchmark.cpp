#include "Engine.hpp"
#include <cuda_runtime.h>
#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using Clock = std::chrono::steady_clock;

void synchronize_gpu() {
    const auto error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

size_t positive_count(std::string_view argument) {
    size_t count = 0;
    const auto [end, error] = std::from_chars(argument.data(), argument.data() + argument.size(), count);

    if (error != std::errc{} || end != argument.data() + argument.size() || count == 0) {
        throw std::invalid_argument("Output tokens and runs must be positive integers");
    }

    return count;
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 4 || (argc > 1 && std::string_view(argv[1]) == "--help")) {
            std::cout << "Usage: benchmark [output-tokens=128] [runs=3] [prompt]\n";
            return argc > 4 ? 1 : 0;
        }

        const size_t output_tokens = argc > 1 ? positive_count(argv[1]) : 128;
        const size_t runs = argc > 2 ? positive_count(argv[2]) : 3;
        const std::string prompt = argc > 3 ? argv[3] : "Explain what a GPU does in a few sentences.";

        int device;
        auto error = cudaGetDevice(&device);
        cudaDeviceProp properties{};

        if (error == cudaSuccess) {
            error = cudaGetDeviceProperties(&properties, device);
        }

        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA GPU access is required: ") + cudaGetErrorString(error));
        }

        const auto load_start = Clock::now();
        Engine& engine = Engine::start();
        synchronize_gpu();

        std::cout << std::fixed << std::setprecision(3)
                  << "GPU: " << properties.name << '\n'
                  << "Model load: " << seconds_since(load_start) << " s\n";

        ChatSession session;
        session.messages = {{"user", prompt}};

        // Warm up the model before measurement; suppress text printing and EOS stopping.
        engine.run(session, false, 4, false, false);
        synchronize_gpu();

        double total_decode_seconds = 0;
        double total_first_seconds = 0;

        for (size_t run = 0; run < runs; ++run) {
            // Prepare a fresh prompt and reset the KV position outside the timed region.
            engine.run(session, false, 0, false, false);
            const size_t prompt_tokens = session.tokens.size();

            if (output_tokens > session.cache_capacity - prompt_tokens) {
                throw std::invalid_argument("Prompt plus output exceeds session capacity");
            }

            synchronize_gpu();
            const auto first_start = Clock::now();
            engine.append_token(session, engine.inference(session));
            synchronize_gpu();
            const double first_seconds = seconds_since(first_start);

            const auto decode_start = Clock::now();

            for (size_t token = 1; token < output_tokens; ++token) {
                engine.append_token(session, engine.inference(session));
            }

            synchronize_gpu();
            const double decode_seconds = seconds_since(decode_start);
            total_first_seconds += first_seconds;
            total_decode_seconds += decode_seconds;

            std::cout << "Run " << run + 1 << ": " << prompt_tokens << " prompt tokens, "
                      << output_tokens << " generated tokens | "
                      << "first token " << first_seconds * 1000.0 << " ms";

            if (output_tokens > 1) {
                std::cout << " | generation after first token " << (output_tokens - 1) / decode_seconds << " tokens/s";
            }

            std::cout << '\n';
        }

        std::cout << "Average first token: " << total_first_seconds * 1000.0 / runs << " ms\n";

        if (output_tokens > 1) {
            std::cout << "Average generation after first token: "
                      << static_cast<double>(output_tokens - 1) * runs / total_decode_seconds
                      << " tokens/s\n";
        }

        std::cout << "Excludes model loading, tokenization, and text printing. "
                     "Generation continues past EOS to measure a fixed token count.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
