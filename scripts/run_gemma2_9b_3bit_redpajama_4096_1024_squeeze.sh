#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/load_env.sh"

MODEL="${MODEL:-google/gemma-2-9b}"
BITS="${BITS:-3}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE:-raw}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"

MODEL="${MODEL}" \
BITS="${BITS}" \
SEQ_LEN="${SEQ_LEN}" \
NUM_EXAMPLES="${NUM_EXAMPLES}" \
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE}" \
EVAL_DTYPE="${EVAL_DTYPE}" \
bash "${SCRIPT_DIR}/run_llama3_3bit_redpajama_4096_1024_squeeze.sh"
