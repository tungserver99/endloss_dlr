#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
MODEL_BASENAME="${MODEL##*/}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
SEQ_LEN="${SEQ_LEN:-4096}"

DST_DIR="${ROOT_DIR}/cache/tokens"
DST="${DST_DIR}/${MODEL_BASENAME}-redpajama_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt"

mkdir -p "${DST_DIR}"

if [[ -e "${DST}" ]]; then
  echo "RedPajama token cache already exists at ${DST}"
  exit 0
fi

if [[ "${MODEL_BASENAME}" == "Meta-Llama-3-8B" && "${NUM_EXAMPLES}" == "1024" && "${SEQ_LEN}" == "4096" ]]; then
  URL="https://github.com/snu-mllab/GuidedQuant/releases/download/v1.0.0/Meta-Llama-3-8B-redpajama_s1024_blk4096.pt"
else
  echo "No known prebuilt RedPajama token cache URL for ${MODEL_BASENAME}, num_examples=${NUM_EXAMPLES}, seq_len=${SEQ_LEN}" >&2
  echo "Use REDPAJAMA_SOURCE=raw to build tokens from the dataset instead." >&2
  exit 1
fi

if command -v wget >/dev/null 2>&1; then
  wget -O "${DST}" "${URL}"
elif command -v curl >/dev/null 2>&1; then
  curl -L "${URL}" -o "${DST}"
else
  echo "Neither wget nor curl is available to download the RedPajama token cache." >&2
  exit 1
fi

echo "Downloaded RedPajama token cache to ${DST}"
