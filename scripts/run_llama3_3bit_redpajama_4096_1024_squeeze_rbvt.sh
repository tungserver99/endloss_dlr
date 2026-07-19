#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/load_env.sh"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
MODEL_BASENAME="${MODEL##*/}"
BITS="${BITS:-3}"
DATASET="redpajama"
SEQ_LEN="${SEQ_LEN:-4096}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
CPU_COUNT="${CPU_COUNT:-8}"
N_CALIB="${N_CALIB:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_LAMBDA_TAG="$(python -c 'import sys; print(f"{float(sys.argv[1]):g}")' "${RBVT_LAMBDA}")"
RBVT_TOPK="${RBVT_TOPK:-0}"
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE:-cache}"
REDPAJAMA_DATASET_REPO="${REDPAJAMA_DATASET_REPO:-togethercomputer/RedPajama-Data-1T}"
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
KEEP_SHARED_STATS="${KEEP_SHARED_STATS:-1}"
KEEP_BASE_SQUEEZE="${KEEP_BASE_SQUEEZE:-1}"
OVERWRITE_POSTPROCESS="${OVERWRITE_POSTPROCESS:-1}"
FORCE_BASE_REBUILD="${FORCE_BASE_REBUILD:-0}"

if [[ "${EVAL_DTYPE}" == "float16" ]]; then
  case "${MODEL,,}" in
    *gemma*|*qwen*)
      EVAL_DTYPE="bfloat16"
      echo "[run_llama3_3bit_redpajama_4096_1024_squeeze_rbvt] Auto-selected EVAL_DTYPE=bfloat16 for MODEL=${MODEL}"
      ;;
  esac
fi

FORCE_BASE_REBUILD="${FORCE_BASE_REBUILD}" \
MODEL="${MODEL}" \
BITS="${BITS}" \
DATASET="${DATASET}" \
SEQ_LEN="${SEQ_LEN}" \
NUM_EXAMPLES="${NUM_EXAMPLES}" \
CPU_COUNT="${CPU_COUNT}" \
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE}" \
REDPAJAMA_DATASET_REPO="${REDPAJAMA_DATASET_REPO}" \
bash "${SCRIPT_DIR}/ensure_squeeze_base.sh"

MODE="rbvt" \
MODEL="${MODEL}" \
BITS="${BITS}" \
DATASET="${DATASET}" \
SEQ_LEN="${SEQ_LEN}" \
NUM_EXAMPLES="${NUM_EXAMPLES}" \
N_CALIB="${N_CALIB}" \
BATCH_SIZE="${BATCH_SIZE}" \
RBVT_LAMBDA="${RBVT_LAMBDA}" \
RBVT_TOPK="${RBVT_TOPK}" \
CPU_COUNT="${CPU_COUNT}" \
OVERWRITE_POSTPROCESS="${OVERWRITE_POSTPROCESS}" \
bash "${SCRIPT_DIR}/run_redpajama_postprocess_fast.sh"

BASE_QUANTIZED_PATH="./cache/quantized/${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
BASE_PACKED_PATH="./cache/packed/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
POST_QUANTIZED_PATH="./cache/post_sqllm_quantized/${MODEL_BASENAME}-w${BITS}-sqllm-rbvt-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA_TAG}"
TOKENS_PATH="./cache/tokens/${MODEL_BASENAME}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt"
MODEL_PATH="./cache/post_sqllm_packed/anyprec-sqllm-rbvt-${MODEL_BASENAME}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA_TAG}"
OUTPUT_FILE="${EVAL_OUTPUT_DIR}/$(basename "${MODEL_PATH}").json"

python scripts/eval_nonuquant_style_ppl.py \
  --model-path "${MODEL}" \
  --quantized-path "${POST_QUANTIZED_PATH}" \
  --model-name "$(basename "${MODEL_PATH}")" \
  --tokenizer-path "${MODEL}" \
  --datasets ${EVAL_DATASETS} \
  --dtype "${EVAL_DTYPE}" \
  --stride "${EVAL_STRIDE}" \
  --max-length "${EVAL_MAX_LENGTH}" \
  --c4-samples "${EVAL_C4_SAMPLES}" \
  --hf-token "${HF_TOKEN}" \
  --output-file "${OUTPUT_FILE}"

if [[ "${CLEANUP_INTERMEDIATE}" == "1" ]]; then
  rm -rf "${POST_QUANTIZED_PATH}"
  if [[ "${KEEP_BASE_SQUEEZE}" != "1" ]]; then
    rm -rf "${BASE_QUANTIZED_PATH}" "${BASE_PACKED_PATH}"
  fi
  if [[ "${KEEP_SHARED_STATS}" != "1" ]]; then
    :
  fi
  if [[ "${REDPAJAMA_SOURCE}" == "raw" ]]; then
    rm -f "${TOKENS_PATH}"
  fi
fi

if [[ "${KEEP_FINAL_PACKED}" != "1" ]]; then
  rm -rf "${MODEL_PATH}"
fi

if [[ "${KEEP_RESULTS_ONLY}" == "1" ]]; then
  rm -rf "${BASE_QUANTIZED_PATH}" "${BASE_PACKED_PATH}" "${POST_QUANTIZED_PATH}" "${MODEL_PATH}"
  if [[ "${REDPAJAMA_SOURCE}" == "raw" ]]; then
    rm -f "${TOKENS_PATH}"
  fi
fi
