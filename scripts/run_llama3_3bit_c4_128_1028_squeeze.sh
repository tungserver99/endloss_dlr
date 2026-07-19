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
EVAL_DATASETS="${EVAL_DATASETS:-wikitext2 c4}"
EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_C4_SAMPLES="${EVAL_C4_SAMPLES:-2000}"
EVAL_DTYPE="${EVAL_DTYPE:-float16}"
HF_TOKEN="${HF_TOKEN:-}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-./results}"
CLEANUP_INTERMEDIATE="${CLEANUP_INTERMEDIATE:-1}"
KEEP_FINAL_PACKED="${KEEP_FINAL_PACKED:-1}"
KEEP_RESULTS_ONLY="${KEEP_RESULTS_ONLY:-0}"
FORCE_BASE_REBUILD="${FORCE_BASE_REBUILD:-1}"

FORCE_BASE_REBUILD="${FORCE_BASE_REBUILD}" \
MODEL="${MODEL}" \
BITS="${BITS}" \
DATASET="${DATASET}" \
SEQ_LEN="${SEQ_LEN}" \
NUM_EXAMPLES="${NUM_EXAMPLES}" \
CPU_COUNT="${CPU_COUNT}" \
bash "${SCRIPT_DIR}/ensure_squeeze_base.sh"

QUANTIZED_PATH="./cache/quantized/${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
MODEL_PATH="./cache/packed/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
OUTPUT_FILE="${EVAL_OUTPUT_DIR}/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}.json"

python scripts/eval_nonuquant_style_ppl.py \
  --model-path "${MODEL}" \
  --quantized-path "${QUANTIZED_PATH}" \
  --model-name "anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}" \
  --tokenizer-path "${MODEL}" \
  --datasets ${EVAL_DATASETS} \
  --dtype "${EVAL_DTYPE}" \
  --stride "${EVAL_STRIDE}" \
  --max-length "${EVAL_MAX_LENGTH}" \
  --c4-samples "${EVAL_C4_SAMPLES}" \
  --hf-token "${HF_TOKEN}" \
  --output-file "${OUTPUT_FILE}"

if [[ "${CLEANUP_INTERMEDIATE}" == "1" ]]; then
  rm -rf "${QUANTIZED_PATH}"
fi

if [[ "${KEEP_FINAL_PACKED}" != "1" ]]; then
  rm -rf "${MODEL_PATH}"
fi

if [[ "${KEEP_RESULTS_ONLY}" == "1" ]]; then
  rm -rf "${QUANTIZED_PATH}" "${MODEL_PATH}"
fi
