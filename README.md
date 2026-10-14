# Inference Engine

Requires CMake 3.25+, Ninja, a C++20 compiler, CUDA toolkit, and vcpkg.
GGML and minja use FetchContent; vcpkg supplies PCRE2, ICU, nlohmann JSON,
and Catch2 (when the `tests` manifest feature is enabled). minja reuses the
vcpkg JSON target.

```sh
git submodule update --init vcpkg
./vcpkg/bootstrap-vcpkg.sh -disableMetrics
cmake --preset dev
cmake --build --preset dev -j 4
ctest --preset dev
```

To reuse the existing local checkout, add
`-DCMAKE_TOOLCHAIN_FILE="$PWD/.cache/vcpkg/scripts/buildsystems/vcpkg.cmake"`
to the configure command.

Tests use `models/Qwen3-4B-Q4_K_M.gguf` by default. To use another compatible
Qwen3 model, configure with:

```sh
cmake --preset dev -DMODEL_PATH=/absolute/path/to/model.gguf
```

Tokenizer and ChatFormatter checks read real model metadata and do not require
a GPU. EmbedLookup tests validate dimensions, empty/invalid input, and compare
GPU lookup results with GGML dequantization. The GPU check requires CUDA access.
Engine is unfinished and is not tested.

For an application build without tests, use the `release` configure/build presets.

Validated the Debug application and test builds with Clang 18 and CUDA 13.3.
All eight CTest cases pass using the real model, including stream ownership,
Q4_K/Q6_K matmul and GPU embedding comparisons.

The Matmul GPU check compares Q4_K and Q6_K against GGML CPU dequantization
using real model weights, including repeated calls to verify output clearing:

```sh
ctest --preset dev -R Matmul
```

`kernels::matmul(input, weight, output, rows, stream)` queues its
`cudaMemsetAsync` and matrix kernel on the supplied CUDA stream. The stream
argument defaults to the default stream. Independent products may use separate
nonblocking streams and separate outputs; callers must use events or stream
synchronization before consuming results on another stream, and keep all buffers
alive until completion. No synchronization is performed inside matmul.

`kernels::CudaStream` (declared in `Kernels.hpp`, implemented in
`src/cuda/stream.cpp`) owns a nonblocking CUDA stream. It is movable and cannot
be copied; use `get()`, `ready()`, and `synchronize()` explicitly. Destruction
releases the stream without waiting for queued work. For example, once input is
ready:

```cpp
kernels::CudaStream first, second;
kernels::matmul(input, q_weight, q_output, rows, first);
kernels::matmul(input, k_weight, k_output, rows, second);
first.synchronize();
second.synchronize();
```

To build and run the standalone stream benchmark with optimizations:

```sh
cmake --preset release -DBUILD_BENCHMARK=ON
cmake --build --preset release -j 4
./build/release/benchmark/matmul_benchmark
```

The benchmark loads real Q4_K/Q6_K projection matrices, warms up each mode,
and reports wall-clock batch time per pair including submission and final stream
synchronization (50 pairs per sample, three samples). Model loading is excluded.
On the RTX 4060 Laptop GPU, the Release build measured:

| Input rows | One stream, ms/pair | Two streams, ms/pair |
| --- | --- | --- |
| 1 | 0.500 | 0.494 |
| 32 | 3.632 | 3.663 |

These measurements show no meaningful throughput gain for these shapes. Actual
overlap depends on GPU resources and workload size. Engine scheduling still uses
the default stream; the standalone benchmark exercises concurrent scheduling.
