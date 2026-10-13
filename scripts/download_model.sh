#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${root_dir}/models/Qwen3-4B-Q4_K_M.gguf}"
revision="bc640142c66e1fdd12af0bd68f40445458f3869b"
sha256="7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"
url="https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/${revision}/Qwen3-4B-Q4_K_M.gguf?download=true"

mkdir -p "$(dirname "${output}")"
curl --fail --location --continue-at - --output "${output}" "${url}"
printf '%s  %s\n' "${sha256}" "${output}" | sha256sum --check
echo "Downloaded and verified ${output}"
