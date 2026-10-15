// Q4_K/Q6_K weights times groupwise Q8 activations, using packed byte dot products.
#include "Kernels.hpp"
#include <cmath>

namespace {
constexpr unsigned warp_width = 32;
constexpr unsigned warps_per_row = 4;

__device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

// One warp quantizes 32 FP32 activations with one shared FP32 scale.
__global__ void quantize(const float* input, int8_t* codes, float* scales, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return; // count is a multiple of 256: no partial warps.
    const float value = input[i];
    float maximum = fabsf(value);
    for (int offset = 16; offset > 0; offset /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    maximum = __shfl_sync(0xffffffffu, maximum, 0);
    const float scale = maximum / 127.0f;
    codes[i] = maximum == 0.0f ? 0 : static_cast<int8_t>(roundf((value / maximum) * 127.0f));
    if (threadIdx.x % 32 == 0) scales[i / 32] = scale;
}

// Q6_K rows/blocks have only two-byte alignment, so assemble words from halves.
__device__ __forceinline__ int read_word(const uint8_t* p) {
    const auto* halves = reinterpret_cast<const uint16_t*>(p);
    return int(unsigned(halves[0]) | (unsigned(halves[1]) << 16));
}

// A lane handles eight consecutive values in one 32-value group. Across the
// warp all eight groups of a 256-weight block are covered. Scale/minimum work
// is applied to integer sums, rather than reconstructing eight FP32 weights.
template<ggml_type Type>
__device__ __forceinline__ float block_dot(const uint8_t* block, const int8_t* input,
    const float* scales, unsigned lane) {
    const unsigned group = lane / 4;
    const unsigned offset = (lane % 4) * 8;
    const auto* x = reinterpret_cast<const int*>(input + group * 32 + offset);
    if constexpr (Type == GGML_TYPE_Q4_K) {
        const auto* packed = reinterpret_cast<const unsigned*>(block + 16 + (group / 2) * 32 + offset);
        const unsigned shift = (group % 2) * 4;
        const int a = int((packed[0] >> shift) & 0x0f0f0f0f);
        const int b = int((packed[1] >> shift) & 0x0f0f0f0f);
        const int dot = __dp4a(b, x[1], __dp4a(a, x[0], 0));
        const int input_sum = __dp4a(0x01010101, x[1], __dp4a(0x01010101, x[0], 0));
        const auto* parameters = block + 4;
        int scale, minimum;
        if (group < 4) {
            scale = parameters[group] & 63;
            minimum = parameters[group + 4] & 63;
        } else {
            const unsigned g = group - 4;
            scale = ((parameters[g] >> 6) << 4) | (parameters[8 + g] & 15);
            minimum = ((parameters[4 + g] >> 6) << 4) | (parameters[8 + g] >> 4);
        }
        const float d = __half2float(*reinterpret_cast<const __half*>(block));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
        return scales[group] * (d * scale * dot - dmin * minimum * input_sum);
    } else {
        const unsigned half = group / 4, plane = group % 4;
        int dot = 0;
        #pragma unroll
        for (unsigned word = 0; word < 2; ++word) {
            const unsigned low = unsigned(read_word(block + half * 64 + (plane % 2) * 32 + offset + word * 4));
            const unsigned high = unsigned(read_word(block + 128 + half * 32 + offset + word * 4));
            const unsigned codes = ((low >> (4 * (plane / 2))) & 0x0f0f0f0f) |
                (((high >> (2 * plane)) & 0x03030303) << 4);
            const int centered = __vsubss4(int(codes), 0x20202020);
            dot = __dp4a(centered, x[word], dot);
        }
        const int scale = reinterpret_cast<const int8_t*>(block + 192)[group * 2 + offset / 16];
        const float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
        return scales[group] * (d * scale * dot);
    }
}

// Four warps cooperate on one output element, covering different packed blocks.
// Only four partial sums are shared; each output is written once.
template<ggml_type Type>
__global__ void project(const uint8_t* weight, const int8_t* input, const float* scales,
    float* output, size_t width, size_t columns) {
    __shared__ float partials[warps_per_row];
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const size_t column = blockIdx.x, row = blockIdx.y;
    const size_t blocks = width / 256;
    constexpr size_t packed_bytes = Type == GGML_TYPE_Q4_K ? 144 : 210;
    float sum = 0.0f;
    for (size_t block = warp; block < blocks; block += warps_per_row)
        sum += block_dot<Type>(weight + (column * blocks + block) * packed_bytes,
            input + row * width + block * 256, scales + row * (width / 32) + block * 8, lane);
    sum = warp_sum(sum);
    if (lane == 0) partials[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < warps_per_row ? partials[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) output[row * columns + column] = sum;
    }
}

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace

namespace kernels::v2 {
void matmul(const float* input, const Weight& weight, float* output, size_t rows, cudaStream_t stream) {
    if (weight.datatype != GGML_TYPE_Q4_K && weight.datatype != GGML_TYPE_Q6_K)
        throw std::invalid_argument("matmul supports Q4_K and Q6_K weights only");
    if (weight.dimension.size() < 2 || weight.dimension[0] <= 0 ||
        weight.dimension[0] % 256 != 0 || weight.dimension[1] <= 0)
        throw std::invalid_argument("Invalid matmul weight dimensions");
    if (rows == 0) return;
    const size_t width = weight.dimension[0], columns = weight.dimension[1], count = rows * width;
    void* workspace = nullptr;
    check(cudaMallocAsync(&workspace, count + (count / 32) * sizeof(float), stream));
    auto* codes = static_cast<int8_t*>(workspace);
    auto* scales = reinterpret_cast<float*>(codes + count);
    try {
        quantize<<<(count + 255) / 256, 256, 0, stream>>>(input, codes, scales, count);
        detail::check_launch();
        const dim3 grid(columns, rows);
        if (weight.datatype == GGML_TYPE_Q4_K)
            project<GGML_TYPE_Q4_K><<<grid, warps_per_row * 32, 0, stream>>>(
                static_cast<const uint8_t*>(weight.device_buffer.data), codes, scales, output, width, columns);
        else
            project<GGML_TYPE_Q6_K><<<grid, warps_per_row * 32, 0, stream>>>(
                static_cast<const uint8_t*>(weight.device_buffer.data), codes, scales, output, width, columns);
        detail::check_launch();
    } catch (...) {
        cudaFreeAsync(workspace, stream);
        throw;
    }
    check(cudaFreeAsync(workspace, stream));
}
} // namespace kernels::v2
