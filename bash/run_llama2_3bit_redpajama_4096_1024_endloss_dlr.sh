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

bash "${SCRIPT_DIR}/ensure_endloss_dlr_base.sh"

MODEL_BASENAME="${MODEL##*/}"
MODEL_PATH="${CACHE_DIR}/packed/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}"
OUTPUT_FILE="${RESULTS_DIR}/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}.json"

mkdir -p "${RESULTS_DIR}"

python scripts/eval_ppl_single_model.py \
  --model_path "${MODEL_PATH}" \
  --output_file "${OUTPUT_FILE}" \
  --datasets "${PPL_DATASETS}" \
  --chunk_size "${SEQ_LEN}"
