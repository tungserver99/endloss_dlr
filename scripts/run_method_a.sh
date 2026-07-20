#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

MODEL="meta-llama/Llama-2-7b-hf"
MODEL_BASENAME="${MODEL##*/}"
BITS=3
NUM_OUTPUT_GROUPS=4

DATASET="redpajama"
SEQ_LEN=4096
NUM_EXAMPLES=1024
N_CALIB=1024
BATCH_SIZE=1
STATS_LAYER_CHUNK_SIZE=1
STATS_CHUNK_SIZE=1024
ROW_BATCH_SIZE=64
KL_PROBES=1
CPU_COUNT=8
RANDOM_STATE=0
REDPAJAMA_SOURCE="cache"
TOKENS_PATH="./cache/tokens/Llama-2-7b-hf-redpajama_s1024_blk4096.pt"

RUN_NAME="${MODEL_BASENAME}-w${BITS}-method-a-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
QUANTIZED_PATH="./cache/method_a_quantized/${RUN_NAME}"
PACKED_PATH="./cache/method_a_packed/anyprec-${RUN_NAME}"
EVAL_OUTPUT="./results/anyprec-${RUN_NAME}-ppl.json"
EVAL_STRIDE=512
EVAL_MAX_LENGTH=2048
EVAL_C4_SAMPLES=2000

python method_a_quantize.py "${MODEL}" \
  --stage all \
  --bits "${BITS}" \
  --dataset "${DATASET}" \
  --seq-len "${SEQ_LEN}" \
  --num-examples "${NUM_EXAMPLES}" \
  --n-calib "${N_CALIB}" \
  --batch-size "${BATCH_SIZE}" \
  --stats-layer-chunk-size "${STATS_LAYER_CHUNK_SIZE}" \
  --stats-chunk-size "${STATS_CHUNK_SIZE}" \
  --num-output-groups "${NUM_OUTPUT_GROUPS}" \
  --kl-probes "${KL_PROBES}" \
  --row-batch-size "${ROW_BATCH_SIZE}" \
  --cpu-count "${CPU_COUNT}" \
  --random-state "${RANDOM_STATE}" \
  --redpajama-source "${REDPAJAMA_SOURCE}" \
  --tokens-path "${TOKENS_PATH}" \
  --quantized-path "${QUANTIZED_PATH}" \
  --output-packed-path "${PACKED_PATH}"

python scripts/eval_nonuquant_style_ppl.py \
  --model-path "${MODEL}" \
  --quantized-path "${QUANTIZED_PATH}" \
  --model-name "anyprec-${RUN_NAME}" \
  --tokenizer-path "${MODEL}" \
  --datasets wikitext2 c4 \
  --device cuda \
  --dtype float16 \
  --stride "${EVAL_STRIDE}" \
  --max-length "${EVAL_MAX_LENGTH}" \
  --c4-samples "${EVAL_C4_SAMPLES}" \
  --output-file "${EVAL_OUTPUT}"