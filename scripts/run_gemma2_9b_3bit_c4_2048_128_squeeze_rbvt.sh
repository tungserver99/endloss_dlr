#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/load_env.sh"

MODEL="${MODEL:-google/gemma-2-9b}"
BITS="${BITS:-3}"
DATASET="${DATASET:-c4}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_EXAMPLES="${NUM_EXAMPLES:-128}"
N_CALIB="${N_CALIB:-128}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"

MODEL="${MODEL}" \
BITS="${BITS}" \
DATASET="${DATASET}" \
SEQ_LEN="${SEQ_LEN}" \
NUM_EXAMPLES="${NUM_EXAMPLES}" \
N_CALIB="${N_CALIB}" \
EVAL_DTYPE="${EVAL_DTYPE}" \
bash "${SCRIPT_DIR}/run_llama3_3bit_c4_128_1028_squeeze_rbvt.sh"
