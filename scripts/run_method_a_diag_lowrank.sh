#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
MODEL_BASENAME="${MODEL##*/}"
BITS="${BITS:-3}"
NUM_OUTPUT_GROUPS="${NUM_OUTPUT_GROUPS:-4}"
SKETCH_RANK="${SKETCH_RANK:-64}"
SKETCH_SEED="${SKETCH_SEED:-0}"
SKETCH_TOKEN_CHUNK_SIZE="${SKETCH_TOKEN_CHUNK_SIZE:-256}"

DATASET="${DATASET:-redpajama}"
SEQ_LEN="${SEQ_LEN:-4096}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
N_CALIB="${N_CALIB:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STATS_LAYER_CHUNK_SIZE="${STATS_LAYER_CHUNK_SIZE:-1}"
ROW_BATCH_SIZE="${ROW_BATCH_SIZE:-64}"
KL_PROBES="${KL_PROBES:-1}"
CPU_COUNT="${CPU_COUNT:-8}"
RANDOM_STATE="${RANDOM_STATE:-0}"
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE:-cache}"
METHOD_A_STAGE="${1:-${METHOD_A_STAGE:-all}}"
TOKENS_PATH="${TOKENS_PATH:-./cache/tokens/${MODEL_BASENAME}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt}"

RUN_NAME="${MODEL_BASENAME}-w${BITS}-method-a-diag-lowrank-r${SKETCH_RANK}-s${SKETCH_SEED}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
SQ_LLM_GRADIENTS_PATH="./cache/gradients/${MODEL_BASENAME}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt"
SQ_LLM_Q0_PATH="./cache/quantized/${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
METHOD_A_STATS_PATH="./cache/method_a_stats/${MODEL_BASENAME}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_seed${RANDOM_STATE}_n${N_CALIB}_g${NUM_OUTPUT_GROUPS}_klp${KL_PROBES}"
QUANTIZED_PATH="./cache/method_a_quantized/${RUN_NAME}"
PACKED_PATH="./cache/method_a_packed/anyprec-${RUN_NAME}"
EVAL_OUTPUT="./results/anyprec-${RUN_NAME}-ppl.json"
EVAL_STRIDE=512
EVAL_MAX_LENGTH=2048
EVAL_C4_SAMPLES=2000

mkdir -p results

echo "[run_method_a_diag_lowrank] stage=${METHOD_A_STAGE}"
echo "[run_method_a_diag_lowrank] SqueezeLLM gradients: ${SQ_LLM_GRADIENTS_PATH}"
echo "[run_method_a_diag_lowrank] SqueezeLLM q0:        ${SQ_LLM_Q0_PATH}"
echo "[run_method_a_diag_lowrank] Method A DLR stats:   ${METHOD_A_STATS_PATH}"
echo "[run_method_a_diag_lowrank] Method A output:      ${QUANTIZED_PATH}"

python method_a_quantize.py "${MODEL}" \
  --stage "${METHOD_A_STAGE}" \
  --curvature-backend diag-lowrank \
  --sketch-rank "${SKETCH_RANK}" \
  --sketch-seed "${SKETCH_SEED}" \
  --sketch-token-chunk-size "${SKETCH_TOKEN_CHUNK_SIZE}" \
  --bits "${BITS}" \
  --dataset "${DATASET}" \
  --seq-len "${SEQ_LEN}" \
  --num-examples "${NUM_EXAMPLES}" \
  --n-calib "${N_CALIB}" \
  --batch-size "${BATCH_SIZE}" \
  --stats-layer-chunk-size "${STATS_LAYER_CHUNK_SIZE}" \
  --num-output-groups "${NUM_OUTPUT_GROUPS}" \
  --kl-probes "${KL_PROBES}" \
  --row-batch-size "${ROW_BATCH_SIZE}" \
  --cpu-count "${CPU_COUNT}" \
  --random-state "${RANDOM_STATE}" \
  --redpajama-source "${REDPAJAMA_SOURCE}" \
  --tokens-path "${TOKENS_PATH}" \
  --quantized-path "${QUANTIZED_PATH}" \
  --output-packed-path "${PACKED_PATH}"

if [[ "${METHOD_A_STAGE}" == "all" || "${METHOD_A_STAGE}" == "quantize" ]]; then
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
fi
