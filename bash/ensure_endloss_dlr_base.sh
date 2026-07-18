#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CACHE_DIR="${CACHE_DIR:-./cache}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
BITS="${BITS:-3}"
DATASET="${DATASET:-redpajama}"
SEQ_LEN="${SEQ_LEN:-4096}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE:-cache}"
REDPAJAMA_DATASET_REPO="${REDPAJAMA_DATASET_REPO:-togethercomputer/RedPajama-Data-1T}"
RESULTS_DIR="${RESULTS_DIR:-./results/ppl}"
PPL_DATASETS="${PPL_DATASETS:-wikitext2,c4}"

BETA="${BETA:-0.5}"
RANK="${RANK:-4}"
NUM_OUTPUT_GROUPS="${NUM_OUTPUT_GROUPS:-8}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-1}"
FISHER_PROBES="${FISHER_PROBES:-16}"
GRADIENT_NUM_EXAMPLES="${GRADIENT_NUM_EXAMPLES:-}"
STATS_LAYER_CHUNK_SIZE="${STATS_LAYER_CHUNK_SIZE:-8}"
MAX_OUTER_ITERS="${MAX_OUTER_ITERS:-8}"
REL_TOL="${REL_TOL:-1e-7}"
LAMBDA_SAFETY="${LAMBDA_SAFETY:-1.01}"
TIE_TOL="${TIE_TOL:-0.0}"
CPU_COUNT="${CPU_COUNT:-}"

EXTRA_ARGS=()

if [[ "${OVERWRITE_TOKENS:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--overwrite_tokens")
fi
if [[ "${OVERWRITE_GRADIENTS:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--overwrite_gradients")
fi
if [[ "${OVERWRITE_QUANTIZE:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--overwrite_quantize")
fi
if [[ -n "${RANDOM_STATE:-}" ]]; then
  EXTRA_ARGS+=("--random_state" "${RANDOM_STATE}")
fi
if [[ -n "${GRADIENT_NUM_EXAMPLES}" ]]; then
  EXTRA_ARGS+=("--gradient_num_examples" "${GRADIENT_NUM_EXAMPLES}")
fi
if [[ -n "${STATS_LAYER_CHUNK_SIZE}" ]]; then
  EXTRA_ARGS+=("--stats_layer_chunk_size" "${STATS_LAYER_CHUNK_SIZE}")
fi
if [[ -n "${CPU_COUNT}" ]]; then
  EXTRA_ARGS+=("--cpu_count" "${CPU_COUNT}")
fi

MODEL_BASENAME="${MODEL##*/}"
OUTPUT_FILE="${RESULTS_DIR}/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}.json"
mkdir -p "${RESULTS_DIR}"

python quantize.py "${MODEL}" \
  --seed_precision "${BITS}" \
  --parent_precision "${BITS}" \
  --mode quantize \
  --cache_dir "${CACHE_DIR}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --num_examples "${NUM_EXAMPLES}" \
  --redpajama_source "${REDPAJAMA_SOURCE}" \
  --redpajama_dataset_repo "${REDPAJAMA_DATASET_REPO}" \
  --beta "${BETA}" \
  --rank "${RANK}" \
  --num_output_groups "${NUM_OUTPUT_GROUPS}" \
  --calibration_batch_size "${CALIBRATION_BATCH_SIZE}" \
  --fisher_probes "${FISHER_PROBES}" \
  --max_outer_iters "${MAX_OUTER_ITERS}" \
  --rel_tol "${REL_TOL}" \
  --lambda_safety "${LAMBDA_SAFETY}" \
  --tie_tol "${TIE_TOL}" \
  --eval_ppl_datasets "${PPL_DATASETS}" \
  --eval_ppl_output_file "${OUTPUT_FILE}" \
  "${EXTRA_ARGS[@]}"
