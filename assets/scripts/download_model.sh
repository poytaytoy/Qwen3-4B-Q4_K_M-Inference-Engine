#!/usr/bin/env bash
set -euo pipefail

assets_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${assets_dir}/models/Qwen3-4B-Q4_K_M.gguf}"

fetch() {
    local destination="$1"
    local checksum="$2"
    local url="$3"

    if [[ -f "${destination}" ]] && printf '%s  %s\n' "${checksum}" "${destination}" | sha256sum --check --status; then
        echo "Already verified ${destination}"
        return
    fi

    mkdir -p "$(dirname "${destination}")"
    curl --fail --location --continue-at - --output "${destination}.part" "${url}"
    printf '%s  %s\n' "${checksum}" "${destination}.part" | sha256sum --check
    mv "${destination}.part" "${destination}"
    echo "Downloaded and verified ${destination}"
}

fetch "${output}" \
    "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5" \
    "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf?download=true"

fetch "$(dirname "${output}")/tokenizer.json" \
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4" \
    "https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/tokenizer.json"
