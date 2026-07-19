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
STATS_TAG="fastwgf_v2_${DATA_TAG}_r4_os4_ncalib1024_bs1_fprobe16_gex1024_lchunk8_og8_damp0p0001_seed0"
SOLVER_TAG="${STATS_TAG}_beta0_iters0_rtol1em07_lambda1p01_sdmin1em08"
RUN_TAG="${MODEL_BASENAME}-w3-endloss-dlr-${SOLVER_TAG}"
QUANTIZED_PATH="cache/endloss_dlr_quantized/${RUN_TAG}"
HF_TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARGS=(--hf-token "${HF_TOKEN}")
fi

WINDOW="${WINDOW:-55}"
TOKEN="${TOKEN:-642}"
python scripts/debug_endloss_layer30_mlp_channels.py \
  --model-path "${MODEL}" \
  --quantized-path "${QUANTIZED_PATH}" \
  --tokenizer-path "${MODEL}" \
  --layer 30 \
  --window "${WINDOW}" \
  --token "${TOKEN}" \
  --channels 3721 7006 \
  --stride 512 \
  --max-length 2048 \
  --dtype float16 \
  "${HF_TOKEN_ARGS[@]}"