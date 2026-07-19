#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/load_env.sh"

mkdir -p logs
LOG_FILE="logs/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"

MODEL="meta-llama/Llama-2-7b-hf"
MODEL_BASENAME="${MODEL##*/}"
DATA_TAG="${MODEL_BASENAME}-redpajama_s1024_blk4096"
BASE_STATS_TAG="fastwgf_v2_${DATA_TAG}_r4_os4_ncalib1024_bs1_fprobe16_gex1024_lchunk8_og8_damp0p0001_seed0"
LEGACY_BASE_STATS_TAG="${DATA_TAG}_r4_os4_ncalib1024_fprobe16_gex1024_lchunk8_og8_damp0p0001_seed0"
BASE_STATS_PATH="cache/endloss_dlr_stats/${BASE_STATS_TAG}"
LEGACY_BASE_STATS_PATH="cache/endloss_dlr_stats/${LEGACY_BASE_STATS_TAG}"
if [[ ! -d "${BASE_STATS_PATH}" && -d "${LEGACY_BASE_STATS_PATH}" ]]; then
  BASE_STATS_PATH="${LEGACY_BASE_STATS_PATH}"
fi
HF_TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARGS=(--hf-token "${HF_TOKEN}")
fi
OVERWRITE_GRADIENT_CACHE="${OVERWRITE_GRADIENT_CACHE:-0}"
GRADIENT_CACHE_ARGS=()
if [[ "${OVERWRITE_GRADIENT_CACHE}" == "1" ]]; then
  GRADIENT_CACHE_ARGS=(--overwrite-gradient-cache)
fi
STATS_TAG="fastwgf_v2_${DATA_TAG}_r4_os4_ncalib1024_bs1_fprobe16_gex1024_lchunk8_og8_damp0p0001_seed0"
SOLVER_TAG="${STATS_TAG}_beta0_iters0_rtol1em07_lambda1p01_sdmin1em08"
RUN_TAG="${MODEL_BASENAME}-w3-endloss-dlr-${SOLVER_TAG}"
QUANTIZED_PATH="cache/endloss_dlr_quantized/${RUN_TAG}"
PACKED_PATH="cache/endloss_dlr_packed/anyprec-${RUN_TAG}"
OUTPUT_FILE="results/anyprec-${RUN_TAG}-wikitext2.json"

python endloss_dlr_quantize.py "${MODEL}" \
  --stage all \
  --bits 3 \
  --dataset redpajama \
  --redpajama-source cache \
  --seq-len 4096 \
  --num-examples 1024 \
  --n-calib 1024 \
  --batch-size 1 \
  --stats-chunk-size 1024 \
  --fisher-probes 16 \
  --stats-layer-chunk-size 8 \
  --num-output-groups 8 \
  --damping-ratio 1e-4 \
  --row-batch-size 64 \
  --rank 4 \
  --oversampling 4 \
  --beta 0.0 \
  --max-outer-iters 0 \
  --rel-tol 1e-7 \
  --lambda-safety 1.01 \
  --device cuda \
  --cpu-count 8 \
  --stats-path "${BASE_STATS_PATH}" \
  "${GRADIENT_CACHE_ARGS[@]}" \
  --quantized-path "${QUANTIZED_PATH}" \
  --output-packed-path "${PACKED_PATH}" \
  --overwrite-quantize \
  --overwrite-pack

python scripts/check_endloss_dlr_cache.py "${QUANTIZED_PATH}" --warn-abs 100 --topk 20

python scripts/eval_nonuquant_style_ppl.py \
  --model-path "${MODEL}" \
  --quantized-path "${QUANTIZED_PATH}" \
  --model-name "anyprec-${RUN_TAG}" \
  --tokenizer-path "${MODEL}" \
  --datasets wikitext2 \
  --dtype float16 \
  --stride 512 \
  --max-length 2048 \
  "${HF_TOKEN_ARGS[@]}" \
  --output-file "${OUTPUT_FILE}"