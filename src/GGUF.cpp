#include "GGUF.hpp"
#include <fstream>
#include <limits>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace {
GGUF::MetadataValue read_scalar(const gguf_context* ctx, int64_t id, gguf_type type) {
    switch (type) {
    case GGUF_TYPE_UINT8: return gguf_get_val_u8(ctx, id);
    case GGUF_TYPE_INT8: return gguf_get_val_i8(ctx, id);
    case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
    case GGUF_TYPE_INT16: return gguf_get_val_i16(ctx, id);
    case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
    case GGUF_TYPE_INT32: return gguf_get_val_i32(ctx, id);
    case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, id);
    case GGUF_TYPE_BOOL: return gguf_get_val_bool(ctx, id);
    case GGUF_TYPE_UINT64: return gguf_get_val_u64(ctx, id);
    case GGUF_TYPE_INT64: return gguf_get_val_i64(ctx, id);
    case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(ctx, id);
    case GGUF_TYPE_STRING: return std::string(gguf_get_val_str(ctx, id));
    default: throw std::runtime_error("Unsupported GGUF scalar type");
    }
}

template<class T>
T array_element(const void* data, size_t index) {
    T value;
    std::memcpy(&value, static_cast<const std::byte*>(data) + index * sizeof(T), sizeof(T));
    return value;
}

template<class T>
std::vector<T> read_array(const gguf_context* ctx, int64_t id) {
    std::vector<T> values;
    const size_t count = gguf_get_arr_n(ctx, id);
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<T, std::string>)
            values.emplace_back(gguf_get_arr_str(ctx, id, i));
        else if constexpr (std::is_same_v<T, bool>)
            values.push_back(array_element<int8_t>(gguf_get_arr_data(ctx, id), i) != 0);
        else
            values.push_back(array_element<T>(gguf_get_arr_data(ctx, id), i));
    }
    return values;
}

GGUF::MetadataValue read_array_value(const gguf_context* ctx, int64_t id, gguf_type type) {
    switch (type) {
    case GGUF_TYPE_UINT8: return read_array<uint8_t>(ctx, id);
    case GGUF_TYPE_INT8: return read_array<int8_t>(ctx, id);
    case GGUF_TYPE_UINT16: return read_array<uint16_t>(ctx, id);
    case GGUF_TYPE_INT16: return read_array<int16_t>(ctx, id);
    case GGUF_TYPE_UINT32: return read_array<uint32_t>(ctx, id);
    case GGUF_TYPE_INT32: return read_array<int32_t>(ctx, id);
    case GGUF_TYPE_FLOAT32: return read_array<float>(ctx, id);
    case GGUF_TYPE_BOOL: return read_array<bool>(ctx, id);
    case GGUF_TYPE_STRING: return read_array<std::string>(ctx, id);
    case GGUF_TYPE_UINT64: return read_array<uint64_t>(ctx, id);
    case GGUF_TYPE_INT64: return read_array<int64_t>(ctx, id);
    case GGUF_TYPE_FLOAT64: return read_array<double>(ctx, id);
    default: throw std::runtime_error("Unsupported GGUF array element type");
    }
}

template<class T>
const std::vector<T>& known_array(const GGUF& reader, const std::string& key) {
    static const std::vector<T> empty;
    if (!reader.get_metadata(key)) return empty;
    const auto* values = reader.get_array<T>(key);
    if (!values) throw std::runtime_error("Wrong GGUF array type for " + key);
    return *values;
}

}

GGUF::GGUF(const std::filesystem::path& path, bool load_weights) : path_(path) {
    ggml_context* weights = nullptr;
    gguf_init_params params{};
    params.no_alloc = !load_weights;
    params.ctx = load_weights ? &weights : nullptr;
    file_.reset(gguf_init_from_file(path_.string().c_str(), params));
    weights_.reset(weights);
    if (!file_) throw std::runtime_error("Failed to read GGUF: " + path_.string());

    for (int64_t i = 0; i < gguf_get_n_kv(file_.get()); ++i) {
        const auto type = gguf_get_kv_type(file_.get(), i);
        Metadata entry{type, std::nullopt, uint8_t{0}};
        if (type == GGUF_TYPE_ARRAY) {
            const auto element_type = gguf_get_arr_type(file_.get(), i);
            entry.element_type = element_type;
            entry.value = read_array_value(file_.get(), i, element_type);
        } else {
            entry.value = read_scalar(file_.get(), i, type);
        }
        metadata_.emplace(gguf_get_key(file_.get(), i), std::move(entry));
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(file_.get()); ++i) {
        TensorInfo info{};
        info.name = gguf_get_tensor_name(file_.get(), i);
        std::copy_n(gguf_get_tensor_ne(file_.get(), i), GGML_MAX_DIMS, info.dimensions.begin());
        info.type = gguf_get_tensor_type(file_.get(), i);
        info.bytes = gguf_get_tensor_size(file_.get(), i);
        info.file_offset = get_data_offset() + gguf_get_tensor_offset(file_.get(), i);
        tensors_.push_back(std::move(info));
    }
}

uint32_t GGUF::get_version() const { return gguf_get_version(file_.get()); }
size_t GGUF::get_alignment() const { return gguf_get_alignment(file_.get()); }
size_t GGUF::get_data_offset() const { return gguf_get_data_offset(file_.get()); }

const GGUF::Metadata* GGUF::get_metadata(const std::string& key) const {
    const auto it = metadata_.find(key);
    return it == metadata_.end() ? nullptr : &it->second;
}
const GGUF::TensorInfo* GGUF::get_tensor_info(const std::string& name) const {
    const auto it = std::find_if(tensors_.begin(), tensors_.end(),
        [&](const TensorInfo& tensor) { return tensor.name == name; });
    if (it == tensors_.end()) throw std::runtime_error("Tensor unavailable: " + name);
    return &*it;
}
const ggml_tensor* GGUF::get_tensor(const std::string& name) const {
    return weights_ ? ggml_get_tensor(weights_.get(), name.c_str()) : nullptr;
}

std::vector<std::byte> GGUF::read_tensor_data(const std::string& name) const {
    const auto* info = get_tensor_info(name);
    std::ifstream file(path_, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open GGUF: " + path_.string());
    const auto end = file.tellg();
    if (end < 0 || info->file_offset > static_cast<uint64_t>(end) ||
        info->bytes > static_cast<uint64_t>(end) - info->file_offset)
        throw std::runtime_error("Incomplete GGUF tensor data: " + name);
    
    file.seekg(static_cast<std::streamoff>(info->file_offset));
    std::vector<std::byte> data(info->bytes);
    if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size())))
        throw std::runtime_error("Failed to read tensor: " + name);
    return data;
}

std::span<const std::byte> GGUF::get_tensor_data(const std::string& name) const {
    if (!weights_) throw std::logic_error("Construct GGUF with load_weights=true to access weights");
    const auto* tensor = get_tensor(name);
    if (!tensor) throw std::out_of_range("Unknown tensor: " + name);
    return {static_cast<const std::byte*>(tensor->data), ggml_nbytes(tensor)};
}
std::optional<uint32_t> GGUF::architecture_value(const std::string& suffix) const {
    const auto arch = get_architecture();
    return arch ? get_value<uint32_t>(*arch + "." + suffix) : std::nullopt;
}
std::optional<std::string> GGUF::get_name() const { return get_value<std::string>("general.name"); }
std::optional<std::string> GGUF::get_architecture() const { return get_value<std::string>("general.architecture"); }
std::optional<std::string> GGUF::get_tokenizer_model() const { return get_value<std::string>("tokenizer.ggml.model"); }
std::optional<uint32_t> GGUF::get_context_length() const { return architecture_value("context_length"); }
std::optional<uint32_t> GGUF::get_embedding_length() const { return architecture_value("embedding_length"); }
std::optional<uint32_t> GGUF::get_layer_count() const { return architecture_value("block_count"); }
std::optional<uint32_t> GGUF::get_attention_head_count() const { return architecture_value("attention.head_count"); }
const std::vector<std::string>& GGUF::get_tokens() const { return known_array<std::string>(*this, "tokenizer.ggml.tokens"); }
const std::vector<float>& GGUF::get_token_scores() const { return known_array<float>(*this, "tokenizer.ggml.scores"); }
const std::vector<int32_t>& GGUF::get_token_types() const { return known_array<int32_t>(*this, "tokenizer.ggml.token_type"); }
const std::vector<std::string>& GGUF::get_merges() const { return known_array<std::string>(*this, "tokenizer.ggml.merges"); }
std::optional<uint32_t> GGUF::get_bos_token_id() const { return get_value<uint32_t>("tokenizer.ggml.bos_token_id"); }
std::optional<uint32_t> GGUF::get_eos_token_id() const { return get_value<uint32_t>("tokenizer.ggml.eos_token_id"); }

void GGUF::free_weights() {
    weights_.reset(); 
}
