#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/load_env.sh"

MODEL_NAME="${1:?Usage: bash scripts/run_lnq.sh <model> <bits> <num_groups> [-m mode]}"
BITS="${2:?Usage: bash scripts/run_lnq.sh <model> <bits> <num_groups> [-m mode]}"
NUM_GROUPS="${3:?Usage: bash scripts/run_lnq.sh <model> <bits> <num_groups> [-m mode]}"

MODE="${LNQ_MODE:-pack}"
if [[ "${4:-}" == "-m" && -n "${5:-}" ]]; then
  MODE="${5}"
fi

DATASET="${DATASET:-redpajama}"
SEQ_LEN="${SEQ_LEN:-4096}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
CPU_COUNT="${CPU_COUNT:-8}"
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE:-raw}"
REDPAJAMA_DATASET_REPO="${REDPAJAMA_DATASET_REPO:-ZengXiangyu/RedPajama-Data-1T-Sample}"
FORCE_BASE_REBUILD="${FORCE_BASE_REBUILD:-0}"
LNQ_NUM_ITERATIONS="${LNQ_NUM_ITERATIONS:-3}"
LNQ_CD_CYCLES="${LNQ_CD_CYCLES:-4}"
OVERWRITE_LNQ="${OVERWRITE_LNQ:-1}"
LNQ_COMPACT_LOG="${LNQ_COMPACT_LOG:-1}"

FORCE_BASE_REBUILD="${FORCE_BASE_REBUILD}" \
MODEL="${MODEL_NAME}" \
BITS="${BITS}" \
DATASET="${DATASET}" \
SEQ_LEN="${SEQ_LEN}" \
NUM_EXAMPLES="${NUM_EXAMPLES}" \
CPU_COUNT="${CPU_COUNT}" \
REDPAJAMA_SOURCE="${REDPAJAMA_SOURCE}" \
REDPAJAMA_DATASET_REPO="${REDPAJAMA_DATASET_REPO}" \
bash "${SCRIPT_DIR}/ensure_squeeze_base.sh"

export LNQ_COMPACT_LOG
export TQDM_DISABLE="${TQDM_DISABLE:-${LNQ_COMPACT_LOG}}"

LNQ_ARGS=(
  "${MODEL_NAME}"
  --seed_precision "${BITS}"
  --mode "${MODE}"
  --dataset "${DATASET}"
  --seq_len "${SEQ_LEN}"
  --num_examples "${NUM_EXAMPLES}"
  --cpu_count "${CPU_COUNT}"
  --num_groups "${NUM_GROUPS}"
  --num_iterations "${LNQ_NUM_ITERATIONS}"
  --cd_cycles "${LNQ_CD_CYCLES}"
  --is_nosal true
)

if [[ "${OVERWRITE_LNQ}" == "1" ]]; then
  LNQ_ARGS+=(--overwrite_quantize --overwrite_pack)
fi

python layerwise_nuq.py "${LNQ_ARGS[@]}"
