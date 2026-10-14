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
Matmul decodes several weights per thread once, retains them across token rows,
and combines their products before one warp reduction and output update per row.
The helpers stay in `src/cuda/matmul/kernel1.cu`, alongside the single
`constexpr size_t strides_per_warp = 32` value. Edit that constant to change the
chunk count, then rebuild. Each chunk covers 32 input elements per warp. Larger
values retain more decoded floats per thread, so benchmark them rather than
assuming they will be faster. Partial final groups are handled without crossing
into the next output column.

`kernels::v2::matmul(...)` selects the shared-memory tiled implementation in
`src/cuda/matmul/kernel2.cu`, with the same arguments and asynchronous stream
contract. `kernels::matmul(...)` / `kernels::v1::matmul(...)` retain kernel1.
Kernel2 stages packed Q4_K/Q6_K blocks and FP32 input in shared memory, decodes
with the same bit layout into a padded, transposed FP32 tile, and reuses those
weights across token rows. Each lane owns an output column and retains its
output sums in registers across all K tiles; results overwrite output directly
without a clear kernel or atomics.

The K tile is 256 elements and the output tile is 32 columns. With 256 threads,
input tiles contain 8, 16, 32, or 56 token rows, selected by input row count and
the device's opt-in shared-memory limit. On the RTX 4060 Laptop (99 KiB per
block), the largest supported layout contains 32 packed weight rows, their
32 × 256 decoded floats, and 56 × 256 input floats. Including padding, this
uses 95,968 bytes for Q4_K or 98,080 bytes for Q6_K. This is the largest
multiple-of-eight input tile fitting this layout, not a universal optimal tile.
Only one such block fits per SM. Smaller devices use the layouts that fit.
Partial token/output tiles are guarded; input width remains a multiple of 256.

The benchmark compares both kernels at 1, 32, and 100 input tokens. A first
Release comparison on this GPU measured these serial Q4_K + Q6_K pair times
(model loading excluded):

| Input tokens | Kernel1 | Kernel2 |
| --- | --- | --- |
| 1 | 0.264 ms | 0.514 ms |
| 32 | 0.960 ms | 1.073 ms |
| 100 | 2.613 ms | 2.686 ms |

These are projection-pair timings, not end-to-end generation throughput.
A separate smaller-tile trial (at most 16 input rows/block) slowed kernel2 to
1.307 ms at 32 tokens and 4.353 ms at 100 tokens. The larger layouts remain
selected, but kernel1 remains the default because tiling has not established an
overall speedup. GPU clocks and stream scheduling affect these measurements.
Real-model CPU-reference tests cover both formats, all tile sizes, partial
boundaries, repeated output writes, and zero input:

```sh
ctest --preset dev -R 'Tiled matmul' --output-on-failure
```

Engine scheduling still uses the default stream; the standalone benchmark
exercises concurrent scheduling.

The fixed-workload Release sweep used the real model on the RTX 4060 Laptop GPU,
23 input tokens, five single-token runs per setting, and excluded model loading:

| Chunks per warp | Tokens/s |
| --- | --- |
| 4 | 3.084 |
| 8 | 3.781 |
| 16 | 4.240 |
| 32 | 4.296 |
| 64 | 3.739 |

A reversed-order repeat measured 4.297 tokens/s at 32 and 3.667 at 64. The sweep
stopped at this clear slowdown; 32 is the fastest tested count and the default.
Nsight Compute reported zero local/shared spill requests for both Q4_K and Q6_K
at 32 and 64. Q4_K used 66 registers/thread at 32 versus 113 at 64, with achieved
occupancy falling from about 45% to 28%. Q6_K used 90 versus 113 registers/thread.
These results support reduced occupancy from register pressure as a contributor,
without evidence of register spilling.

Logs, saved benchmark binaries, resource reports, and Nsight captures are under
`build/stride-sweep/`; `results.json` contains the initial sweep measurements.

Source-correlated profiling of the restored 32-chunk array kernel is saved in
`build/stream-bench/matmul_array32_source_analysis.ncu-rep` (with CSV and JSON
summary exports alongside it). Across two Q4_K launches and one Q6_K launch,
input load/local accumulation accounts for 57–65% of sampled not-issued stalls,
decoding/array setup for 17–25%, shuffle reduction for 11–12%, and output update
plus conditional/address work for about 2%. These are stall-sample shares, not
elapsed-time percentages. The regular kernel's reduction share was about 49%.
Both formats report zero spilling. The first Q4_K launch executes about 76 million
instructions versus 310 million in the regular kernel capture. The latest
unprofiled five-run inference measurement is 3.806 tokens/s; all eight tests pass.
