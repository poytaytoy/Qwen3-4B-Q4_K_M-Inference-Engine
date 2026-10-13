#pragma once

#include <ggml.h>
#include <gguf.h>
#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

// Owns the GGUF file's metadata and optionally its CPU-resident weights.
// Returned references, spans, and tensor pointers live as long as this object.
class GGUF {
public:
    // Arbitrary metadata can have any GGUF type; arrays preserve their element type.
    using MetadataValue = std::variant<
        uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
        float, bool, std::string, uint64_t, int64_t, double,
        std::vector<uint8_t>, std::vector<int8_t>, std::vector<uint16_t>,
        std::vector<int16_t>, std::vector<uint32_t>, std::vector<int32_t>,
        std::vector<float>, std::vector<bool>, std::vector<std::string>,
        std::vector<uint64_t>, std::vector<int64_t>, std::vector<double>>;
    struct Metadata {
        gguf_type type;
        // Present only for arrays.
        std::optional<gguf_type> element_type;
        MetadataValue value;
    };
    // Describes a weight tensor without loading its weight data.
    struct TensorInfo {
        // Identifier used to find this weight, e.g. token_embd.weight.
        std::string name;
        // Size along each axis; unused axes have size 1.
        std::array<int64_t, GGML_MAX_DIMS> dimensions;
        // Weight storage format, such as F32, F16, or a quantized format.
        ggml_type type;
        // Size of the stored tensor data in bytes.
        size_t bytes;
        size_t file_offset; // Absolute offset in the GGUF file.
    };

    // Opens the file; set load_weights=true to also read weights into CPU RAM. Throws on failure.
    explicit GGUF(const std::filesystem::path& path, bool load_weights = false);
    // Copying is disabled because this object owns the allocated contexts.
    GGUF(const GGUF&) = delete;
    GGUF& operator=(const GGUF&) = delete;

    // Path used to open the file.
    const std::filesystem::path& get_path() const { return path_; }
    // GGUF format version, not the model version.
    uint32_t get_version() const;
    // Byte alignment used for tensor storage and padding.
    size_t get_alignment() const;
    // Absolute file offset where the tensor data section starts.
    size_t get_data_offset() const;
    // Whether weight data was loaded into RAM.
    bool has_weights() const { return weights_ != nullptr; }
    // All metadata, indexed by key name.
    const std::map<std::string, Metadata>& get_metadata() const { return metadata_; }
    // One metadata entry; nullptr if the key does not exist.
    const Metadata* get_metadata(const std::string& key) const;
    // Descriptions of every tensor in the file.
    const std::vector<TensorInfo>& get_tensors() const { return tensors_; }
    // Description of a named tensor; nullptr if absent.
    const TensorInfo* get_tensor_info(const std::string& name) const;
    // Loaded tensor; nullptr if absent or weights were not loaded.
    const ggml_tensor* get_tensor(const std::string& name) const;
    // Borrowed raw weight bytes, not necessarily floats. Throws if weights are unavailable or name is absent.
    std::span<const std::byte> get_tensor_data(const std::string& name) const;

    // Missing keys or mismatched types return nullopt/nullptr.
    // Read a scalar with its exact stored C++ type, e.g. get_value<uint32_t>(key).
    template<class T> std::optional<T> get_value(const std::string& key) const {
        const auto* entry = get_metadata(key);
        if (!entry) return std::nullopt;
        const auto* value = std::get_if<T>(&entry->value);
        return value ? std::optional<T>(*value) : std::nullopt;
    }
    // Array metadata; nullptr if the key is missing or is not an array.
    template<class T> const std::vector<T>* get_array(const std::string& key) const {
        const auto* entry = get_metadata(key);
        return entry ? std::get_if<std::vector<T>>(&entry->value) : nullptr;
    }
    // Convenience getters below return nullopt/nullptr for missing or mismatched metadata.
    // Human-readable model name, if provided.
    std::optional<std::string> get_name() const;
    // Model architecture identifier, e.g. llama.
    std::optional<std::string> get_architecture() const;
    // Declared context length in tokens.
    std::optional<uint32_t> get_context_length() const;
    // Number of components in each token embedding.
    std::optional<uint32_t> get_embedding_length() const;
    // Number of transformer blocks.
    std::optional<uint32_t> get_layer_count() const;
    // Number of attention query heads per layer.
    std::optional<uint32_t> get_attention_head_count() const;
    // Tokenizer identifier stored in the file; this does not implement tokenization.
    std::optional<std::string> get_tokenizer_model() const;
    // Array getters return an empty vector if absent and throw on a wrong stored type.
    // Vocabulary strings; array index is the token ID.
    const std::vector<std::string>& get_tokens() const;
    // Tokenizer scores for vocabulary entries, when provided.
    const std::vector<float>& get_token_scores() const;
    // Tokenizer categories for vocabulary entries, when provided.
    const std::vector<int32_t>& get_token_types() const;
    // Ordered merge rules used by BPE tokenizers, when provided.
    const std::vector<std::string>& get_merges() const;
    // ID of the beginning-of-sequence token.
    std::optional<uint32_t> get_bos_token_id() const;
    // ID of the end-of-sequence token.
    std::optional<uint32_t> get_eos_token_id() const;

private:
    // Looks up an architecture-specific key, e.g. llama.block_count.
    std::optional<uint32_t> architecture_value(const std::string& suffix) const;
    // Original file path.
    std::filesystem::path path_;
    // Declaration order ensures tensor storage is freed before GGUF metadata.
    // Owns GGUF metadata; automatically calls gguf_free on destruction.
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file_{nullptr, gguf_free};
    // Owns optional tensor storage; automatically calls ggml_free on destruction.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights_{nullptr, ggml_free};
    // Extracted copies of all metadata entries.
    std::map<std::string, Metadata> metadata_;
    // Extracted descriptions of all tensors.
    std::vector<TensorInfo> tensors_;
};
