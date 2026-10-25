#include "Tokenizer.hpp"
#include <tokenizers_cpp.h>
#include <nlohmann/json.hpp>
#include <climits>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace {
// Hugging Face's ByteLevel decoder replaces incomplete UTF-8. Cache raw pieces
// instead so decoding one token at a time preserves the engine's streaming API.
std::unordered_map<std::string, char> byte_alphabet() {
    std::unordered_map<std::string, char> alphabet;
    unsigned extra = 256;

    for (unsigned byte = 0; byte < 256; ++byte) {
        const bool direct =
            (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
        const unsigned code = direct ? byte : extra++;
        std::string symbol;

        if (code < 128) {
            symbol += static_cast<char>(code);
        } else {
            symbol += static_cast<char>(0xc0 | (code >> 6));
            symbol += static_cast<char>(0x80 | (code & 63));
        }

        alphabet.emplace(symbol, static_cast<char>(byte));
    }

    return alphabet;
}
} // namespace

struct Tokenizer::Impl {
    std::unique_ptr<tokenizers::Tokenizer> ordinary;
    std::unique_ptr<tokenizers::Tokenizer> with_special;
    std::vector<std::string> pieces;
    std::vector<bool> special;
    std::optional<uint32_t> bos;
    std::optional<uint32_t> eos;
    bool add_bos;
    bool add_eos;
};

Tokenizer::Tokenizer(const GGUF &model) : impl_(std::make_unique<Impl>()) {
    const auto path = model.get_path().parent_path() / "tokenizer.json";
    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error(
            "Cannot open " + path.string() +
            ". Run bash assets/scripts/download_model.sh to download the Qwen3 tokenizer.");
    }

    auto config = nlohmann::json::parse(file);
    const auto &tokens = model.get_tokens();
    const auto &types = model.get_token_types();

    if (tokens.empty() || types.size() != tokens.size() || tokens.size() > INT32_MAX) {
        throw std::invalid_argument("Invalid GGUF vocabulary");
    }

    // Check that the official tokenizer assigns the same IDs as the model.
    auto check_token = [&](const std::string &token, size_t id) {
        if (id >= tokens.size() || tokens[id] != token) {
            throw std::invalid_argument("tokenizer.json vocabulary does not match the GGUF model");
        }
    };

    for (const auto &entry : config.at("model").at("vocab").items()) {
        check_token(entry.key(), entry.value().get<size_t>());
    }

    for (const auto &entry : config.at("added_tokens")) {
        check_token(entry.at("content").get<std::string>(), entry.at("id").get<size_t>());
    }

    impl_->with_special = tokenizers::Tokenizer::FromBlobJSON(config.dump());

    // The existing API can also encode special-token spellings as ordinary text.
    config["added_tokens"] = nlohmann::json::array();
    impl_->ordinary = tokenizers::Tokenizer::FromBlobJSON(config.dump());

    const auto alphabet = byte_alphabet();

    for (size_t id = 0; id < tokens.size(); ++id) {
        impl_->special.push_back(types[id] == 3);
        std::string piece;

        if (types[id] == 3 || types[id] == 4) {
            piece = tokens[id];
        } else {
            for (size_t pos = 0; pos < tokens[id].size();) {
                const size_t length = static_cast<unsigned char>(tokens[id][pos]) < 128 ? 1 : 2;
                piece += alphabet.at(tokens[id].substr(pos, length));
                pos += length;
            }
        }

        impl_->pieces.push_back(std::move(piece));
    }

    impl_->bos = model.get_bos_token_id();
    impl_->eos = model.get_eos_token_id();
    impl_->add_bos = model.get_value<bool>("tokenizer.ggml.add_bos_token").value_or(false);
    impl_->add_eos = model.get_value<bool>("tokenizer.ggml.add_eos_token").value_or(false);

    for (auto id : {impl_->bos, impl_->eos}) {
        if (id && *id >= tokens.size()) {
            throw std::invalid_argument("Invalid BOS/EOS token ID");
        }
    }

    if ((impl_->add_bos && !impl_->bos) || (impl_->add_eos && !impl_->eos)) {
        throw std::invalid_argument("Automatic BOS/EOS insertion requires a token ID");
    }
}

Tokenizer::~Tokenizer() = default;

std::vector<uint32_t> Tokenizer::encode(const std::string &text, bool allow_special) const {
    auto &backend = allow_special ? impl_->with_special : impl_->ordinary;
    const auto encoded = backend->Encode(text);
    std::vector<uint32_t> ids(encoded.begin(), encoded.end());

    if (impl_->add_bos) {
        ids.insert(ids.begin(), *impl_->bos);
    }

    if (impl_->add_eos) {
        ids.push_back(*impl_->eos);
    }

    return ids;
}

std::string Tokenizer::decode(uint32_t id, bool skip_special) const {
    if (id >= impl_->pieces.size()) {
        throw std::out_of_range("Invalid token ID");
    }

    if (skip_special && impl_->special[id]) {
        return {};
    }

    return impl_->pieces[id];
}

std::string Tokenizer::decode(const std::vector<uint32_t> &ids, bool skip_special) const {
    std::string text;

    for (auto id : ids) {
        text += decode(id, skip_special);
    }

    return text;
}
