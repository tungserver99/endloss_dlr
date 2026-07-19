#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="logs/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"

python endloss_dlr_quantize.py "meta-llama/Llama-2-7b-hf" \
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
  --beta 0.5 \
  --max-outer-iters 0 \
  --rel-tol 1e-7 \
  --lambda-safety 1.01 \
  --device cuda \
  --cpu-count 8 \
  --overwrite-quantize \
  --overwrite-pack