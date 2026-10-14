#include "Kernels.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        cudaDeviceProp gpu{};
        const auto error = cudaGetDeviceProperties(&gpu, 0);
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        GGUF model(MATMUL_MODEL_PATH);
        Weight q4(model, "blk.0.attn_q.weight"), q6(model, "blk.0.attn_v.weight");
        if (q4.datatype != GGML_TYPE_Q4_K || q6.datatype != GGML_TYPE_Q6_K ||
            q4.dimension[0] != q6.dimension[0])
            throw std::runtime_error("Benchmark requires matching Q4_K/Q6_K projection widths");
        const size_t width = q4.dimension[0];
        constexpr size_t iterations = 50, samples = 3;
        std::cout << gpu.name << " | " << iterations << " pairs/sample, " << samples << " samples\n"
                  << "Q4_K " << width << " x " << q4.dimension[1]
                  << ", Q6_K " << width << " x " << q6.dimension[1] << '\n';
        std::cout << std::fixed << std::setprecision(3);
        for (size_t rows : {size_t{1}, size_t{32}}) {
            std::vector<float> input(rows * width);
            for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * 0.031f) * 0.1f;
            DeviceBuffer device_input(input.data(), input.size() * sizeof(float));
            DeviceBuffer q4_output(rows * q4.dimension[1] * sizeof(float));
            DeviceBuffer q6_output(rows * q6.dimension[1] * sizeof(float));
            kernels::CudaStream first, second;
            auto submit = [&](bool parallel) {
                kernels::matmul(static_cast<const float*>(device_input.data), q4,
                    static_cast<float*>(q4_output.data), rows, first);
                kernels::matmul(static_cast<const float*>(device_input.data), q6,
                    static_cast<float*>(q6_output.data), rows, parallel ? second : first);
            };
            auto join = [&] { first.synchronize(); second.synchronize(); };
            // Synchronous input copy completes before nonblocking streams read it.
            for (bool parallel : {false, true}) {
                for (int i = 0; i < 10; ++i) submit(parallel);
                join();
            }
            double serial = 0, parallel = 0;
            for (size_t sample = 0; sample < samples; ++sample) {
                for (bool concurrent : {false, true}) {
                    const auto start = std::chrono::steady_clock::now();
                    for (size_t i = 0; i < iterations; ++i) submit(concurrent);
                    join();
                    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count() / iterations;
                    (concurrent ? parallel : serial) += ms / samples;
                }
            }
            std::cout << "rows=" << rows << ": serial " << serial << " ms/pair, separate streams "
                      << parallel << " ms/pair, speedup " << serial / parallel << "x\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
