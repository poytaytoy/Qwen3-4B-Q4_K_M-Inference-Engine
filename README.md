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
cmake --preset dev -DINFERENCE_TEST_MODEL=/absolute/path/to/model.gguf
```

Tokenizer and ChatFormatter checks read real model metadata and do not require
a GPU. EmbedLookup tests validate dimensions, empty/invalid input, and compare
GPU lookup results with GGML dequantization. The GPU check requires CUDA access.
Engine is unfinished and is not tested.

For an application build without tests, use the `release` configure/build presets.
`Engine::inference()` is unfinished and has no return statement (the compiler
reports `-Wreturn-type`); `Engine::run()` currently samples random tokens.

Validated the Debug application and test builds with Clang 18 and CUDA 13.3.
All six Tokenizer, ChatFormatter, and EmbedLookup CTest cases passed using the
default real model, including the GPU embedding comparison.
