#include "Engine.hpp"
#include "ChatSession.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        int device_count = 0;
        const auto error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            std::cerr << "CUDA is required. Enable GPU access and ensure a CUDA-capable GPU and NVIDIA driver are available.";
            if (error != cudaSuccess)
                std::cerr << " CUDA error: " << cudaGetErrorString(error);
            std::cerr << '\n';
            return 1;
        }

        Engine& engine = Engine::start();
        ChatSession session;
        std::string input;

        while (std::cout << "User: " && std::getline(std::cin, input)) {
            session.messages.push_back({"user", input});
            std::cout << "Assistant: ";
            const auto response = engine.run(session);
            session.messages.push_back({"assistant", response});
            std::cout << '\n';
        }
        
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
