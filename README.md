# Inference Engine

Run these commands from the repository root. You need CMake 3.25+, Ninja, Git,
a CUDA-compatible C++20 compiler with OpenMP support, the NVIDIA CUDA toolkit,
a CUDA-capable GPU with GPU access enabled, and Rust/Cargo for the tokenizer backend.

```sh
git submodule update --init --recursive

bash assets/scripts/download_model.sh

cmake --preset release
cmake --build --preset release -j 4

./build/release/inference_engine
```

This is an inference engine for `Qwen3-4B-Q4_K_M.gguf`.

The download script fetches the model and its matching `tokenizer.json` into
`assets/models/`.

Type a message at the `User:` prompt to chat. Press Ctrl+D to exit.

## Generation performance

Using an NVIDIA GeForce RTX 4060 Laptop GPU and
`Qwen3-4B-Q4_K_M.gguf`. The results are as follows: 

Prompt: "Explain what a GPU does in a few sentences." (23 prompt tokens).

| Run | First token | Generated tokens/s afterwards |
| --- | ---: | ---: |
| 1 | 178.202 ms | 62.105 |
| 2 | 190.730 ms | 60.751 |
| 3 | 190.578 ms | 62.115 |
| Average | 186.504 ms | 61.650 |

## TODO

- Implement a native C++ Qwen3 tokenizer to replace the awkward Rust dependency from hugging face's.
- Implement CUDA Graphs for up to 30% speed up 
- Implement a ChatGPT API-compatible on startup instead of user input