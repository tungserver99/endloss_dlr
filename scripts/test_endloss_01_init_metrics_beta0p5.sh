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
python scripts/debug_endloss_init_metrics.py \
  --model "${MODEL}" \
  --stats-path "${BASE_STATS_PATH}" \
  --layer 4 \
  --module self_attn.v_proj \
  --bits 3 \
  --beta 0.5 \
  --device cuda \
  --row-batch-size 64 \
  --topk 30