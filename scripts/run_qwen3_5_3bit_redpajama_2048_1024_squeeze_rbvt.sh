#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/load_env.sh"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
BITS="${BITS:-3}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"

MODEL="${MODEL}" \
BITS="${BITS}" \
SEQ_LEN="${SEQ_LEN}" \
NUM_EXAMPLES="${NUM_EXAMPLES}" \
bash "${SCRIPT_DIR}/run_llama3_3bit_redpajama_4096_1024_squeeze_rbvt.sh"
