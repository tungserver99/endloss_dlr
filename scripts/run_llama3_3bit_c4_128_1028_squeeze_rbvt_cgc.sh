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
N_CALIB="${N_CALIB:-1028}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_LAMBDA_TAG="$(python -c 'import sys; print(f"{float(sys.argv[1]):g}")' "${RBVT_LAMBDA}")"
RBVT_TOPK="${RBVT_TOPK:-0}"
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
OVERWRITE_POSTPROCESS="${OVERWRITE_POSTPROCESS:-1}"
FORCE_BASE_REBUILD="${FORCE_BASE_REBUILD:-0}"

FORCE_BASE_REBUILD="${FORCE_BASE_REBUILD}" \
MODEL="${MODEL}" \
BITS="${BITS}" \
DATASET="${DATASET}" \
SEQ_LEN="${SEQ_LEN}" \
NUM_EXAMPLES="${NUM_EXAMPLES}" \
CPU_COUNT="${CPU_COUNT}" \
bash "${SCRIPT_DIR}/ensure_squeeze_base.sh"

python rbvt_squeezellm.py \
  --model "${MODEL}" \
  --mode rbvt_cgc \
  --bits "${BITS}" \
  --dataset "${DATASET}" \
  --seq-len "${SEQ_LEN}" \
  --num-examples "${NUM_EXAMPLES}" \
  --n-calib "${N_CALIB}" \
  --batch-size "${BATCH_SIZE}" \
  --rbvt-lambda "${RBVT_LAMBDA}" \
  --rbvt-topk "${RBVT_TOPK}" \
  --cpu-count "${CPU_COUNT}" \
  $( [[ "${OVERWRITE_POSTPROCESS}" == "1" ]] && echo --overwrite )

BASE_QUANTIZED_PATH="./cache/quantized/${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
BASE_PACKED_PATH="./cache/packed/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
POST_QUANTIZED_PATH="./cache/post_sqllm_quantized/${MODEL_BASENAME}-w${BITS}-sqllm-rbvt-cgc-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA_TAG}"
MODEL_PATH="./cache/post_sqllm_packed/anyprec-sqllm-rbvt-cgc-${MODEL_BASENAME}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA_TAG}"
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
  rm -rf "${BASE_QUANTIZED_PATH}" "${BASE_PACKED_PATH}" "${POST_QUANTIZED_PATH}"
fi

if [[ "${KEEP_FINAL_PACKED}" != "1" ]]; then
  rm -rf "${MODEL_PATH}"
fi

if [[ "${KEEP_RESULTS_ONLY}" == "1" ]]; then
  rm -rf "${BASE_QUANTIZED_PATH}" "${BASE_PACKED_PATH}" "${POST_QUANTIZED_PATH}" "${MODEL_PATH}"
fi
