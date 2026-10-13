#include "Engine.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        Engine engine(argc > 1 ? argv[1] : "models/Qwen3-4B-Q4_K_M.gguf");
        std::cout << "Mock inference enabled.\n";
        std::string input;
        while (std::cout << "User: " && std::getline(std::cin, input)) {
            std::cout << "Assistant: ";
            engine.run({{"user", input}});
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
