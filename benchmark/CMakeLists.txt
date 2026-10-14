add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE inference_core)

add_executable(matmul_benchmark matmul_benchmark.cpp)
target_link_libraries(matmul_benchmark PRIVATE inference_core)
target_compile_definitions(matmul_benchmark PRIVATE MATMUL_MODEL_PATH="${MODEL_PATH}")
