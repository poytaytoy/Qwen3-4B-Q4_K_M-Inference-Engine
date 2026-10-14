cmake_minimum_required(VERSION 3.24)
project(InferenceEngine LANGUAGES C CXX CUDA)

set(MODEL_PATH "${PROJECT_SOURCE_DIR}/models/Qwen3-4B-Q4_K_M.gguf"
    CACHE FILEPATH "Default GGUF model for the application, benchmark and tests")
get_filename_component(MODEL_PATH "${MODEL_PATH}" ABSOLUTE BASE_DIR "${PROJECT_SOURCE_DIR}")

option(BUILD_TESTING "builds the test cases")
option(BUILD_BENCHMARK "builds the benchmark")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

include(CTest)
include(FetchContent)
find_package(pcre2 CONFIG REQUIRED)
find_package(ICU REQUIRED COMPONENTS uc)
find_package(nlohmann_json CONFIG REQUIRED)

# Make minja's FetchContent request reuse the vcpkg JSON package.
FetchContent_Declare(json FIND_PACKAGE_ARGS CONFIG REQUIRED NAMES nlohmann_json)

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

set(MINJA_TEST_ENABLED OFF CACHE BOOL "" FORCE)
set(MINJA_EXAMPLE_ENABLED OFF CACHE BOOL "" FORCE)
set(MINJA_FUZZTEST_ENABLED OFF CACHE BOOL "" FORCE)
set(MINJA_USE_VENV OFF CACHE BOOL "" FORCE)
FetchContent_Declare(minja
    GIT_REPOSITORY https://github.com/google/minja.git
    GIT_TAG 021c2293c187789ef13d56c6cfd89c9b134fd80f
)
FetchContent_MakeAvailable(minja)

file(GLOB SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cpp"
)

list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
add_library(inference_core STATIC ${SOURCES})
target_compile_definitions(inference_core PRIVATE INFERENCE_MODEL_PATH="${MODEL_PATH}")

target_include_directories(inference_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(inference_core PUBLIC cxx_std_20)
set_target_properties(inference_core PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

target_link_libraries(inference_core PUBLIC ggml-base PCRE2::8BIT ICU::uc CUDA::cudart minja OpenMP::OpenMP_CXX)

add_executable(inference_engine src/main.cpp)
target_link_libraries(inference_engine PRIVATE inference_core)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

if (BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()
