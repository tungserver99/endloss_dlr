#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

TOKEN_DIR="${TOKEN_DIR:-./cache/tokens}"
mkdir -p "${TOKEN_DIR}"
cd "${TOKEN_DIR}"

download() {
  local url="$1"
  local output="$2"
  if [[ -f "${output}" ]]; then
    echo "[download_calibration] Reusing ${output}"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -O "${output}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${output}"
  else
    echo "Need either wget or curl to download calibration files." >&2
    exit 1
  fi
}

safe_link() {
  local target="$1"
  local link_name="$2"
  if [[ -L "${link_name}" || -f "${link_name}" ]]; then
    return
  fi
  ln -s "${target}" "${link_name}"
}

# Llama-2 (redpajama)
download "https://github.com/snu-mllab/GuidedQuant/releases/download/v1.0.0/Llama-2-7b-hf-redpajama_s1024_blk4096.pt" \
  "Llama-2-7b-hf-redpajama_s1024_blk4096.pt"
safe_link "Llama-2-7b-hf-redpajama_s1024_blk4096.pt" "Llama-2-13b-hf-redpajama_s1024_blk4096.pt"
safe_link "Llama-2-7b-hf-redpajama_s1024_blk4096.pt" "Llama-2-70b-hf-redpajama_s1024_blk4096.pt"
safe_link "Llama-2-7b-hf-redpajama_s1024_blk4096.pt" "Llama-2-7b-redpajama_s1024_blk4096.pt"
safe_link "Llama-2-7b-hf-redpajama_s1024_blk4096.pt" "Llama-2-13b-redpajama_s1024_blk4096.pt"
safe_link "Llama-2-7b-hf-redpajama_s1024_blk4096.pt" "Llama-2-70b-redpajama_s1024_blk4096.pt"

# Llama-3 (redpajama)
download "https://github.com/snu-mllab/GuidedQuant/releases/download/v1.0.0/Meta-Llama-3-8B-redpajama_s1024_blk4096.pt" \
  "Meta-Llama-3-8B-redpajama_s1024_blk4096.pt"
safe_link "Meta-Llama-3-8B-redpajama_s1024_blk4096.pt" "Meta-Llama-3-70B-redpajama_s1024_blk4096.pt"

# Llama-2 (wikitext2)
download "https://github.com/snu-mllab/GuidedQuant/releases/download/v1.0.0/Llama-2-7b-hf-wikitext2_s128_blk2048.pt" \
  "Llama-2-7b-hf-wikitext2_s128_blk2048.pt"
safe_link "Llama-2-7b-hf-wikitext2_s128_blk2048.pt" "Llama-2-13b-hf-wikitext2_s128_blk2048.pt"
safe_link "Llama-2-7b-hf-wikitext2_s128_blk2048.pt" "Llama-2-70b-hf-wikitext2_s128_blk2048.pt"
safe_link "Llama-2-7b-hf-wikitext2_s128_blk2048.pt" "Llama-2-7b-wikitext2_s128_blk2048.pt"
safe_link "Llama-2-7b-hf-wikitext2_s128_blk2048.pt" "Llama-2-13b-wikitext2_s128_blk2048.pt"
safe_link "Llama-2-7b-hf-wikitext2_s128_blk2048.pt" "Llama-2-70b-wikitext2_s128_blk2048.pt"
