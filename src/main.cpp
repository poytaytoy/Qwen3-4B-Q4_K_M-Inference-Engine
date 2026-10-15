#include "Engine.hpp"
#include "ChatSession.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        Engine& engine = Engine::start();
        ChatSession session;
        std::cout << "Mock inference enabled.\n";
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
