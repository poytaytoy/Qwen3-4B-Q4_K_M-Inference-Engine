# Inference Engine

## Requirements

- CMake 3.25+, Ninja, Git, and a CUDA-compatible C++20 compiler with OpenMP support.
- NVIDIA CUDA toolkit and a CUDA-capable GPU for inference, benchmarks, and GPU tests.
- Model file at `assets/models/Qwen3-4B-Q4_K_M.gguf`.

vcpkg installs PCRE2, ICU, nlohmann JSON, and Catch2 for tests. CMake downloads
GGML and minja through FetchContent. Initial configuration requires internet access.

## Setup and build

Run these commands from the repository root:

Download the model if needed with `bash assets/scripts/download_model.sh`.

```sh
git submodule update --init vcpkg
./vcpkg/bootstrap-vcpkg.sh -disableMetrics
cmake --preset release
cmake --build --preset release -j 4
./build/release/inference_engine
```

To change the model, add `-DMODEL_PATH=/absolute/path/to/model.gguf` when
configuring, then rebuild. This sets the model for the application, tests, and
benchmarks; use a compatible Qwen3 GGUF model.

## Tests

The `dev` preset enables Catch2 tests and builds with Debug settings:

```sh
cmake --preset dev
cmake --build --preset dev -j 4
ctest --preset dev
```

Run selected tests or list available tests:

```sh
ctest --preset dev -R 'Tokenizer|ChatFormatter'
ctest --preset dev -R 'EmbedLookup|[Mm]atmul|CudaStream'
ctest --preset dev -N
```

Tests use the real model. Tokenizer and ChatFormatter tests run without a GPU;
embedding, matmul, and stream tests require GPU access.

## Benchmarks

Enable benchmark executables in the Release build:

```sh
cmake --preset release -DBUILD_BENCHMARK=ON
cmake --build --preset release -j 4
```

Measure inference speed in generated tokens per second:

```sh
# Arguments: tokens per run, number of runs, optional prompt
./build/release/benchmark/benchmark 32 3
./build/release/benchmark/benchmark 32 3 "Explain what a GPU does."
```

The inference benchmark warms up first, reports model loading separately, and
reports throughput for multiple output tokens per run (defaults: 32 tokens, 3 runs).
It reports aggregate tokens/s including prefill, first-token latency, and cached
decode tokens/s excluding the first token. It generates the requested count even
if EOS occurs; model loading and prompt tokenization are excluded from throughput.

Compare v1 (FP32 inputs) and experimental v2 (Q8 inputs + packed integer dots):

```sh
./build/release/benchmark/matmul_benchmark
```

The application defaults to v1. To benchmark generation with v2:

```sh
INFERENCE_MATMUL=v2 ./build/release/benchmark/benchmark 32 3
```

v2 includes input quantization overhead and changes activation precision.


This reports milliseconds per projection pair for 1, 32, and 100 input tokens,
using one stream or separate streams. These are kernel benchmark timings, not
end-to-end generation speed.
