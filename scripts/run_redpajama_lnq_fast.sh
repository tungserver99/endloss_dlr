#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/load_env.sh"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
MODEL_BASENAME="${MODEL##*/}"
BITS="${BITS:-3}"
DATASET="${DATASET:-redpajama}"
SEQ_LEN="${SEQ_LEN:-4096}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
CPU_COUNT="${CPU_COUNT:-8}"
NUM_GROUPS="${NUM_GROUPS:-1}"
LNQ_NUM_ITERATIONS="${LNQ_NUM_ITERATIONS:-3}"
LNQ_CD_CYCLES="${LNQ_CD_CYCLES:-4}"
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE:-raw}"
REDPAJAMA_DATASET_REPO="${REDPAJAMA_DATASET_REPO:-ZengXiangyu/RedPajama-Data-1T-Sample}"
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
KEEP_BASE_SQUEEZE="${KEEP_BASE_SQUEEZE:-1}"
FORCE_BASE_REBUILD="${FORCE_BASE_REBUILD:-0}"
OVERWRITE_LNQ="${OVERWRITE_LNQ:-1}"
LNQ_COMPACT_LOG="${LNQ_COMPACT_LOG:-1}"

export LNQ_COMPACT_LOG
export TQDM_DISABLE="${TQDM_DISABLE:-${LNQ_COMPACT_LOG}}"

if [[ "${EVAL_DTYPE}" == "float16" ]]; then
  case "${MODEL,,}" in
    *gemma*|*qwen*)
      EVAL_DTYPE="bfloat16"
      echo "[run_redpajama_lnq_fast] Auto-selected EVAL_DTYPE=bfloat16 for MODEL=${MODEL}"
      ;;
  esac
fi

MODEL_NAME="${MODEL}" \
BITS="${BITS}" \
NUM_GROUPS="${NUM_GROUPS}" \
DATASET="${DATASET}" \
SEQ_LEN="${SEQ_LEN}" \
NUM_EXAMPLES="${NUM_EXAMPLES}" \
CPU_COUNT="${CPU_COUNT}" \
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE}" \
REDPAJAMA_DATASET_REPO="${REDPAJAMA_DATASET_REPO}" \
FORCE_BASE_REBUILD="${FORCE_BASE_REBUILD}" \
LNQ_NUM_ITERATIONS="${LNQ_NUM_ITERATIONS}" \
LNQ_CD_CYCLES="${LNQ_CD_CYCLES}" \
OVERWRITE_LNQ="${OVERWRITE_LNQ}" \
bash "${SCRIPT_DIR}/run_lnq.sh" "${MODEL}" "${BITS}" "${NUM_GROUPS}"

BASE_QUANTIZED_PATH="./cache/quantized/${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
BASE_PACKED_PATH="./cache/packed/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
LNQ_QUANTIZED_PATH="./cache/layerwise_quantized/${MODEL_BASENAME}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${NUM_GROUPS}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}_nosal"
TOKENS_PATH="./cache/tokens/${MODEL_BASENAME}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt"
MODEL_PATH="./cache/layerwise_packed/layerwise-${MODEL_BASENAME}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${NUM_GROUPS}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}_nosal"
OUTPUT_FILE="${EVAL_OUTPUT_DIR}/anyprec-lnq-${MODEL_BASENAME}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${NUM_GROUPS}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}.json"

python scripts/eval_nonuquant_style_ppl.py \
  --model-path "${MODEL}" \
  --quantized-path "${LNQ_QUANTIZED_PATH}" \
  --model-name "anyprec-lnq-${MODEL_BASENAME}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${NUM_GROUPS}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}" \
  --tokenizer-path "${MODEL}" \
  --datasets ${EVAL_DATASETS} \
  --dtype "${EVAL_DTYPE}" \
  --stride "${EVAL_STRIDE}" \
  --max-length "${EVAL_MAX_LENGTH}" \
  --c4-samples "${EVAL_C4_SAMPLES}" \
  --hf-token "${HF_TOKEN}" \
  --output-file "${OUTPUT_FILE}"

if [[ "${CLEANUP_INTERMEDIATE}" == "1" ]]; then
  rm -rf "${LNQ_QUANTIZED_PATH}"
  if [[ "${KEEP_BASE_SQUEEZE}" != "1" ]]; then
    rm -rf "${BASE_QUANTIZED_PATH}" "${BASE_PACKED_PATH}"
  fi
  if [[ "${REDPAJAMA_SOURCE}" == "raw" ]]; then
    rm -f "${TOKENS_PATH}"
  fi
fi

if [[ "${KEEP_FINAL_PACKED}" != "1" ]]; then
  rm -rf "${MODEL_PATH}"
fi

if [[ "${KEEP_RESULTS_ONLY}" == "1" ]]; then
  rm -rf "${BASE_QUANTIZED_PATH}" "${BASE_PACKED_PATH}" "${LNQ_QUANTIZED_PATH}" "${MODEL_PATH}"
  if [[ "${REDPAJAMA_SOURCE}" == "raw" ]]; then
    rm -f "${TOKENS_PATH}"
  fi
fi
