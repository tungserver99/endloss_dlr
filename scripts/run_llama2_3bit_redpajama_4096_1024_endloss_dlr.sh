#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

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
  --row-batch-size 64 \
  --rank 4 \
  --oversampling 4 \
  --beta 0.5 \
  --max-outer-iters 8 \
  --rel-tol 1e-7 \
  --lambda-safety 1.01 \
  --device cuda \
  --cpu-count 8 \
  --overwrite-stats \
  --overwrite-quantize \
  --overwrite-pack