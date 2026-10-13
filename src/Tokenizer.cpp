#include "Tokenizer.hpp"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unicode/normalizer2.h>
#include <unicode/unistr.h>

namespace {
std::string normalize_nfc(const std::string& text) {
    UErrorCode status = U_ZERO_ERROR;
    const auto* normalizer = icu::Normalizer2::getNFCInstance(status);
    if (U_FAILURE(status)) throw std::runtime_error("Failed to initialize NFC normalization");
    icu::UnicodeString normalized;
    normalizer->normalize(icu::UnicodeString::fromUTF8(text), normalized, status);
    if (U_FAILURE(status)) throw std::runtime_error("NFC normalization failed");
    std::string result;
    normalized.toUTF8String(result);
    return result;
}
// Byte-encoded vocabulary symbols are at most two UTF-8 bytes long.
std::string utf8(unsigned code) {
    if (code < 128) return std::string(1, static_cast<char>(code));
    return {static_cast<char>(0xc0 | (code >> 6)), static_cast<char>(0x80 | (code & 63))};
}

std::vector<std::string> symbols(const std::string& text) {
    std::vector<std::string> result;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = text[i];
        size_t count = c < 128 ? 1 : (c & 0xe0) == 0xc0 ? 2 : (c & 0xf0) == 0xe0 ? 3 : 4;
        if (i + count > text.size()) throw std::runtime_error("Incomplete vocabulary UTF-8 symbol");
        result.push_back(text.substr(i, count));
        i += count;
    }
    return result;
}
}

Tokenizer::Tokenizer(const GGUF& model)
    : tokens(model.get_tokens()), merges(model.get_merges()), token_types(model.get_token_types()),
      tokenizer_model(model.get_tokenizer_model().value_or("")),
      pre_tokenizer(model.get_value<std::string>("tokenizer.ggml.pre").value_or("")),
      bos_token_id(model.get_bos_token_id()), eos_token_id(model.get_eos_token_id()),
      add_bos_token(model.get_value<bool>("tokenizer.ggml.add_bos_token").value_or(false)),
      add_eos_token(model.get_value<bool>("tokenizer.ggml.add_eos_token").value_or(false)) {
    
    if (tokenizer_model != "gpt2" || pre_tokenizer != "qwen2")
        throw std::invalid_argument("Tokenizer requires gpt2 / qwen2 GGUF metadata");
    
    if (tokens.empty() || tokens.size() != token_types.size() ||
        tokens.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        throw std::invalid_argument("Invalid Qwen vocabulary or token types");
    
    for (auto id : {bos_token_id, eos_token_id})
        if (id && *id >= tokens.size()) throw std::invalid_argument("Invalid BOS/EOS token ID");
    
    if ((add_bos_token && !bos_token_id) || (add_eos_token && !eos_token_id))
        throw std::invalid_argument("Automatic BOS/EOS insertion requires a token ID");
    
    // Qwen2 pre-tokenization pattern; Unicode properties are evaluated by PCRE2.
    const char* regex = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    int error;
    PCRE2_SIZE offset;
    auto* compiled = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(regex), PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr);
    if (!compiled) throw std::runtime_error("Failed to compile Qwen pre-tokenizer regex");
    pattern = std::shared_ptr<void>(compiled, [](void* p) { pcre2_code_free(static_cast<pcre2_code*>(p)); });
    build_lookups();
}

void Tokenizer::build_lookups() {
    token_map.reserve(tokens.size());
    special_ids.resize(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        token_map.emplace(tokens[i], static_cast<uint32_t>(i));
        // GGUF control tokens are special; user-defined tokens are atomic added tokens.
        special_ids[i] = token_types[i] == 3;
        if (token_types[i] == 3 || token_types[i] == 4) {
            if (tokens[i].empty()) throw std::invalid_argument("Empty reserved token");
            reserved_ids.push_back(static_cast<uint32_t>(i));
        }
    }
    for (size_t i = 0; i < merges.size(); ++i) {
        const auto split = merges[i].find(' ');
        if (split == std::string::npos || split == 0 || split + 1 == merges[i].size())
            throw std::invalid_argument("Malformed BPE merge rule");
        merges_map.emplace(std::make_pair(merges[i].substr(0, split), merges[i].substr(split + 1)), i);
    }
    unsigned extra = 256;
    for (unsigned b = 0; b < 256; ++b) {
        const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
        byte_encoder[b] = utf8(direct ? b : extra++);
        byte_decoder.emplace(byte_encoder[b], static_cast<unsigned char>(b));
    }
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    auto* code = static_cast<pcre2_code*>(pattern.get());
    std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(
        pcre2_match_data_create_from_pattern(code, nullptr), pcre2_match_data_free);
    if (!match) throw std::bad_alloc();
    std::vector<std::string> chunks;
    for (size_t offset = 0; offset < text.size();) {
        int rc = pcre2_match(code, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(),
            offset, PCRE2_ANCHORED, match.get(), nullptr);
        if (rc < 0) throw std::invalid_argument("Text is invalid UTF-8 or Qwen pre-tokenization failed");
        auto* bounds = pcre2_get_ovector_pointer(match.get());
        if (bounds[1] <= offset) throw std::runtime_error("Empty pre-tokenizer match");
        chunks.push_back(text.substr(offset, bounds[1] - offset));
        offset = bounds[1];
    }
    return chunks;
}

std::string Tokenizer::encode_bytes(const std::string& text) const {
    std::string out;
    for (unsigned char b : text) out += byte_encoder[b];
    return out;
}

std::string Tokenizer::decode_bytes(const std::string& text) const {
    std::string out;
    for (const auto& symbol : symbols(text)) {
        auto it = byte_decoder.find(symbol);
        if (it == byte_decoder.end()) throw std::runtime_error("Invalid byte-encoded vocabulary symbol");
        out += static_cast<char>(it->second);
    }
    return out;
}
std::vector<uint32_t> Tokenizer::apply_bpe(const std::string& encoded_chunk) const {
    auto pieces = symbols(encoded_chunk);
    while (pieces.size() > 1) {
        size_t best = std::numeric_limits<size_t>::max();
        std::pair<std::string, std::string> pair;
        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            auto it = merges_map.find({pieces[i], pieces[i + 1]});
            if (it != merges_map.end() && it->second < best) { best = it->second; pair = it->first; }
        }
        if (best == std::numeric_limits<size_t>::max()) break;
        std::vector<std::string> next;
        for (size_t i = 0; i < pieces.size();) {
            if (i + 1 < pieces.size() && pieces[i] == pair.first && pieces[i + 1] == pair.second) {
                next.push_back(pieces[i] + pieces[i + 1]); i += 2;
            } else next.push_back(pieces[i++]);
        }
        pieces = std::move(next);
    }
    std::vector<uint32_t> ids;
    for (const auto& piece : pieces) {
        auto it = token_map.find(piece);
        if (it == token_map.end()) throw std::runtime_error("BPE piece missing from vocabulary");
        ids.push_back(it->second);
    }
    return ids;
}
std::vector<uint32_t> Tokenizer::encode_ordinary(const std::string& text) const {
    std::vector<uint32_t> ids;
    for (const auto& chunk : pre_tokenize(normalize_nfc(text))) {
        auto part = apply_bpe(encode_bytes(chunk));
        ids.insert(ids.end(), part.begin(), part.end());
    }
    return ids;
}
std::vector<uint32_t> Tokenizer::encode_with_special_tokens(const std::string& text) const {
    std::vector<uint32_t> ids;
    size_t start = 0;
    while (start < text.size()) {
        size_t position = std::string::npos;
        std::optional<uint32_t> selected;
        for (auto id : reserved_ids) {
            auto found = text.find(tokens[id], start);
            if (found != std::string::npos && (found < position ||
                (found == position && (!selected || tokens[id].size() > tokens[*selected].size())))) {
                position = found; selected = id;
            }
        }
        auto ordinary = encode_ordinary(text.substr(start,
            position == std::string::npos ? position : position - start));
        ids.insert(ids.end(), ordinary.begin(), ordinary.end());
        if (!selected) break;
        ids.push_back(*selected);
        start = position + tokens[*selected].size();
    }
    return ids;
}
std::vector<uint32_t> Tokenizer::encode(const std::string& text, bool allow_special) const {
    auto ids = allow_special ? encode_with_special_tokens(text) : encode_ordinary(text);
    add_sequence_tokens(ids);
    return ids;
}
bool Tokenizer::is_special_token(uint32_t id) const {
    if (static_cast<size_t>(id) >= tokens.size()) throw std::out_of_range("Invalid token ID");
    return special_ids[id];
}
void Tokenizer::add_sequence_tokens(std::vector<uint32_t>& ids) const {
    if (add_bos_token) ids.insert(ids.begin(), static_cast<uint32_t>(*bos_token_id));
    if (add_eos_token) ids.push_back(static_cast<uint32_t>(*eos_token_id));
}

std::string Tokenizer::decode(uint32_t id, bool skip_special) const {
    std::string out;
    
    const bool special = is_special_token(id); // Also validates the ID.
    if (special || token_types[id] == 4) out += tokens[id];
    else out += decode_bytes(tokens[id]);
    
    return out;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids, bool skip_special) const {
    std::string out;

    #pragma omp parallel for schedule(static)
    for (auto id : ids) {
        out += Tokenizer::decode(id, skip_special); 
    }
    return out;
}


