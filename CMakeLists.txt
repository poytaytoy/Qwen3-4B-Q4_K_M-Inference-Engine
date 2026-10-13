cmake_minimum_required(VERSION 3.18)
project(InferenceEngine LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

include(FetchContent)

# Only the GGUF reader and core tensor storage are needed.
# FORCE clears backend settings left over from earlier configurations.
set(GGML_CUDA OFF CACHE BOOL "Enable ggml CUDA backend" FORCE)
set(GGML_CPU OFF CACHE BOOL "Enable ggml CPU backend" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "Enable ggml OpenMP support" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "Build ggml tests" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Build ggml examples" FORCE)

FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(ggml)

# Qwen's pre-tokenization pattern needs Unicode properties and lookahead.
set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(pcre2
    GIT_REPOSITORY https://github.com/PCRE2Project/pcre2.git
    GIT_TAG pcre2-10.45
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(pcre2)
find_package(ICU REQUIRED COMPONENTS uc)

set(MINJA_TEST_ENABLED OFF CACHE BOOL "" FORCE)
set(MINJA_EXAMPLE_ENABLED OFF CACHE BOOL "" FORCE)
set(MINJA_FUZZTEST_ENABLED OFF CACHE BOOL "" FORCE)
set(MINJA_USE_VENV OFF CACHE BOOL "" FORCE)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE
)
FetchContent_Declare(minja
    GIT_REPOSITORY https://github.com/google/minja.git
    GIT_TAG 021c2293c187789ef13d56c6cfd89c9b134fd80f
)
FetchContent_MakeAvailable(minja)

file(GLOB SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_executable(inference_engine ${SOURCES})

target_include_directories(inference_engine PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(inference_engine PRIVATE cxx_std_20)
set_target_properties(inference_engine PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

target_link_libraries(inference_engine PRIVATE ggml-base pcre2-8 ICU::uc CUDA::cudart minja OpenMP::OpenMP_CXX)
