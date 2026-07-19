#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/load_env.sh"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
MODEL_BASENAME="${MODEL##*/}"
BITS="${BITS:-3}"
DATASET="${DATASET:-c4}"
SEQ_LEN="${SEQ_LEN:-128}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1028}"
CPU_COUNT="${CPU_COUNT:-8}"
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE:-cache}"
REDPAJAMA_DATASET_REPO="${REDPAJAMA_DATASET_REPO:-togethercomputer/RedPajama-Data-1T}"
FORCE_BASE_REBUILD="${FORCE_BASE_REBUILD:-0}"

TOKENS_PATH="./cache/tokens/${MODEL_BASENAME}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt"
BASE_QUANTIZED_PATH="./cache/quantized/${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
BASE_PACKED_PATH="./cache/packed/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"

OVERWRITE_ARGS=()
if [[ "${FORCE_BASE_REBUILD}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite_gradients --overwrite_quantize --overwrite_pack)
fi

REDPAJAMA_ARGS=()
if [[ "${DATASET}" == "redpajama" ]]; then
  if [[ "${REDPAJAMA_SOURCE}" == "cache" ]]; then
    :
  elif [[ "${REDPAJAMA_SOURCE}" == "raw" ]]; then
    export REDPAJAMA_DATASET_REPO
    REDPAJAMA_ARGS+=(--redpajama_source "${REDPAJAMA_SOURCE}" --redpajama_dataset_repo "${REDPAJAMA_DATASET_REPO}")
    if [[ "${FORCE_BASE_REBUILD}" == "1" ]]; then
      OVERWRITE_ARGS+=(--overwrite_tokens)
    fi
  else
    echo "Unknown REDPAJAMA_SOURCE=${REDPAJAMA_SOURCE}, expected cache or raw" >&2
    exit 1
  fi
  if [[ "${REDPAJAMA_SOURCE}" == "cache" ]]; then
    REDPAJAMA_ARGS+=(--redpajama_source "${REDPAJAMA_SOURCE}" --redpajama_dataset_repo "${REDPAJAMA_DATASET_REPO}")
  fi
fi

ensure_tokens_only() {
  echo "[ensure_squeeze_base] Building missing token cache: ${TOKENS_PATH}"
  python quantize.py "${MODEL}" \
    --seed_precision "${BITS}" \
    --parent_precision "${BITS}" \
    --mode tokens \
    --dataset "${DATASET}" \
    --seq_len "${SEQ_LEN}" \
    --num_examples "${NUM_EXAMPLES}" \
    --cpu_count "${CPU_COUNT}" \
    "${REDPAJAMA_ARGS[@]}"
}

if [[ "${FORCE_BASE_REBUILD}" != "1" ]] && [[ -d "${BASE_QUANTIZED_PATH}" ]] && [[ -d "${BASE_PACKED_PATH}" ]] && [[ -n "$(find "${BASE_QUANTIZED_PATH}" -mindepth 1 -print -quit 2>/dev/null)" ]] && [[ -n "$(find "${BASE_PACKED_PATH}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  if [[ ! -f "${TOKENS_PATH}" ]]; then
    ensure_tokens_only
  fi
  echo "[ensure_squeeze_base] Reusing base squeeze artifacts:"
  echo "  quantized=${BASE_QUANTIZED_PATH}"
  echo "  packed=${BASE_PACKED_PATH}"
  exit 0
fi

echo "[ensure_squeeze_base] Building base squeeze artifacts"
python quantize.py "${MODEL}" \
  --seed_precision "${BITS}" \
  --parent_precision "${BITS}" \
  --mode pack \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --num_examples "${NUM_EXAMPLES}" \
  --cpu_count "${CPU_COUNT}" \
  "${REDPAJAMA_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"
