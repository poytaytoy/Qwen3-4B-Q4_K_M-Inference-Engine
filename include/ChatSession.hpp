#pragma once
#include "ChatFormatter.hpp"
#include "Weight.hpp"
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct ChatSession {
    struct CacheEntry {
        DeviceBuffer buffer;
        std::size_t width;
        std::size_t capacity;

        CacheEntry(std::size_t width, std::size_t capacity)
            : buffer(width * capacity * sizeof(float)), width(width), capacity(capacity) {}
    };

    std::vector<ChatMessage> messages;
    std::vector<uint32_t> tokens;
    // Per-layer K/V buffers and their dimensions.
    std::map<std::string, CacheEntry> cache;
    std::size_t cached_tokens = 0;
    // Maximum token capacity of each per-layer K/V buffer.
    std::size_t cache_capacity = 2048;

    // Allocate an FP32 cache entry once; subsequent calls reuse its buffer.
    // width is kv_heads * head_dim; cache_capacity sets the token capacity.
    float* update_cache(const std::string& name, std::size_t width) {
        auto& entry = cache.try_emplace(name, width, cache_capacity).first->second;
        return static_cast<float*>(entry.buffer.data);
    }

    // Return the destination for new token rows; kernels can write here directly.
    // Advance cached_tokens only after every layer has finished adding its K/V.
    float* cache_write_ptr(const std::string& name, std::size_t new_tokens = 1) {
        auto& entry = cache.at(name);
        const auto capacity = entry.capacity;
        if (cached_tokens > capacity || new_tokens > capacity - cached_tokens)
            throw std::out_of_range("KV cache capacity exceeded");
        return static_cast<float*>(entry.buffer.data)
            + cached_tokens * entry.width;
    }

    ChatSession() = default;
    ~ChatSession() = default;
    ChatSession(const ChatSession&) = delete;
    ChatSession& operator=(const ChatSession&) = delete;
    ChatSession(ChatSession&&) noexcept = default;
    ChatSession& operator=(ChatSession&&) noexcept = default;
};
