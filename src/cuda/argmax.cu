#include "Kernels.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <math_constants.h>

namespace {
using kernels::detail::threads;
constexpr float negative_infinity = -std::numeric_limits<float>::infinity();
struct candidate {
    size_t index;
    float score;

    __host__ __device__ static candidate init(){
        return candidate{SIZE_MAX, negative_infinity};
    }
};

__host__ __device__ inline void max(candidate& best, candidate& other){
    if (other.score > best.score || (other.score == best.score && other.index < best.index)){
        best = other;  
    }
}

__global__ void argmax_kernel(const float* __restrict logits, candidate* __restrict result, size_t count, size_t stride_per_thread) {
    const size_t threadId = blockIdx.x * blockDim.x + threadIdx.x; 
    candidate best = candidate::init();
    
    for (size_t i = 0; i < stride_per_thread; i++){
        const size_t stride_threadId = threadId + i * (gridDim.x * blockDim.x); 
        if (stride_threadId < count && logits[stride_threadId] > best.score){
            best = {stride_threadId, logits[stride_threadId]}; 
        } 
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        candidate other = {
            __shfl_down_sync(0xffffffffu, best.index, offset),
            __shfl_down_sync(0xffffffffu, best.score, offset)
        };
        max(best, other); 
    }

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    constexpr int block_best_size = threads / 32; 
    __shared__ candidate block_best[block_best_size];
    if (lane == 0){
        block_best[warp] = best; 
    }

    __syncthreads(); 

    if (lane == 0 && warp == 0){
        for (int i = 0; i < block_best_size; ++i){
            max(best, block_best[i]); 
        }
        result[blockIdx.x] = best;
    }
}   

void check_cuda(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA argmax failed: ") + cudaGetErrorString(error));
}

}

namespace kernels {

uint32_t argmax(const float* logits, size_t vocabulary_size) {
    if (!logits || vocabulary_size == 0 || vocabulary_size > UINT32_MAX)
        throw std::invalid_argument("Invalid argmax input");

    const int sm_count = detail::get_sm_count();

    DeviceBuffer storage(sizeof(candidate) * sm_count);
    auto* device_result = static_cast<candidate*>(storage.data);

    constexpr int THREADS_PER_BLOCK = threads; 

    size_t elem_per_sm = vocabulary_size / sm_count;
    size_t stride_per_thread = elem_per_sm / THREADS_PER_BLOCK + 1;  

    argmax_kernel<<<sm_count, THREADS_PER_BLOCK>>>(logits, device_result, vocabulary_size, stride_per_thread);
    detail::check_launch();

    std::vector<candidate> results(sm_count);
    check_cuda(cudaMemcpy(results.data(), device_result, sizeof(candidate) * sm_count, cudaMemcpyDeviceToHost));
    
    candidate best = candidate::init(); 
    for (auto& other: results){
        max(best, other);
    }
    
    return best.index;
}

}
