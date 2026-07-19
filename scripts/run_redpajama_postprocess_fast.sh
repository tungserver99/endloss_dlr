#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/load_env.sh"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
MODEL_BASENAME="${MODEL##*/}"
MODE="${MODE:?MODE is required (cgc|rbvt|cgc_rbvt|rbvt_cgc)}"
BITS="${BITS:-3}"
DATASET="${DATASET:-redpajama}"
SEQ_LEN="${SEQ_LEN:-4096}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
N_CALIB="${N_CALIB:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
CPU_COUNT="${CPU_COUNT:-8}"
DEVICE="${DEVICE:-cuda}"
GPU_DEVICES="${GPU_DEVICES:-auto}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-20000}"
GPU_MAX_DEVICES="${GPU_MAX_DEVICES:-8}"
AUTO_GPU_CANDIDATES="${AUTO_GPU_CANDIDATES:-auto}"
ROW_CHUNK="${ROW_CHUNK:-1024}"
GAP_FLOOR="${GAP_FLOOR:-1e-8}"
ALLOW_OVERSHOOT="${ALLOW_OVERSHOOT:-0}"
OVERWRITE_POSTPROCESS="${OVERWRITE_POSTPROCESS:-1}"
RBVT_LAMBDA_TAG="$(python -c 'import sys; print(f"{float(sys.argv[1]):g}")' "${RBVT_LAMBDA}")"

mode_tag() {
  case "$1" in
    cgc) echo "sqllm-cgc" ;;
    rbvt) echo "sqllm-rbvt" ;;
    cgc_rbvt) echo "sqllm-cgc-rbvt" ;;
    rbvt_cgc) echo "sqllm-rbvt-cgc" ;;
    *)
      echo "Unsupported MODE=$1" >&2
      exit 1
      ;;
  esac
}

INPUT_QUANTIZED_PATH="${INPUT_QUANTIZED_PATH:-./cache/quantized/${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}}"

auto_gpu_devices() {
  local min_free_mb="${1:-${GPU_MIN_FREE_MB}}"
  local candidates="${2:-${AUTO_GPU_CANDIDATES}}"
  local query
  query="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${query}" ]]; then
    return 0
  fi

  local candidate_filter=""
  if [[ -n "${candidates}" && "${candidates}" != "auto" ]]; then
    candidate_filter=" ${candidates//,/ } "
  fi

  local chosen=()
  local line idx free
  while IFS=',' read -r idx free; do
    idx="${idx// /}"
    free="${free// /}"
    [[ -z "${idx}" || -z "${free}" ]] && continue
    if [[ -n "${candidate_filter}" && "${candidate_filter}" != " auto " ]]; then
      [[ "${candidate_filter}" != *" ${idx} "* ]] && continue
    fi
    if (( free >= min_free_mb )); then
      chosen+=("${idx}")
    fi
  done <<< "${query}"

  local limited=()
  local i
  for i in "${!chosen[@]}"; do
    if (( i >= GPU_MAX_DEVICES )); then
      break
    fi
    limited+=("${chosen[$i]}")
  done
  echo "${limited[*]}"
}

resolve_devices() {
  if [[ "${GPU_DEVICES}" == "auto" ]]; then
    auto_gpu_devices "${GPU_MIN_FREE_MB}" "${AUTO_GPU_CANDIDATES}"
  else
    echo "${GPU_DEVICES//,/ }"
  fi
}

primary_device() {
  local devices="$1"
  if [[ -n "${devices}" ]]; then
    read -r -a arr <<< "${devices}"
    if [[ "${#arr[@]}" -gt 0 ]]; then
      echo "cuda:${arr[0]}"
      return
    fi
  fi
  echo "${DEVICE}"
}

count_layers() {
  local weights_dir="${INPUT_QUANTIZED_PATH}/weights"
  if [[ ! -d "${weights_dir}" ]]; then
    echo "Missing weights dir: ${weights_dir}" >&2
    exit 1
  fi
  find "${weights_dir}" -maxdepth 1 -name 'l*.pt' | wc -l | tr -d ' '
}

run_shards() {
  local layer_count="$1"
  local devices="$2"
  shift 2

  if [[ -z "${devices}" ]]; then
    "$@" --device "${DEVICE}"
    return
  fi

  read -r -a arr <<< "${devices}"
  local ngpu="${#arr[@]}"
  if [[ "${ngpu}" -le 1 ]]; then
    "$@" --device "cuda:${arr[0]}"
    return
  fi

  local shard_size=$(( (layer_count + ngpu - 1) / ngpu ))
  local pids=()
  local shard_idx start end
  for shard_idx in "${!arr[@]}"; do
    start=$(( shard_idx * shard_size ))
    end=$(( start + shard_size ))
    if (( start >= layer_count )); then
      continue
    fi
    if (( end > layer_count )); then
      end="${layer_count}"
    fi
    echo "[run_redpajama_postprocess_fast] ${MODE} shard ${start},${end} on cuda:${arr[$shard_idx]}"
    "$@" \
      --device "cuda:${arr[$shard_idx]}" \
      --layer-range "${start}" "${end}" &
    pids+=($!)
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

case "${MODE}" in
  cgc|rbvt|cgc_rbvt|rbvt_cgc) ;;
  *)
    echo "Unsupported MODE=${MODE}" >&2
    exit 1
    ;;
esac

MODE_TAG="$(mode_tag "${MODE}")"
OUTPUT_QUANTIZED_PATH="./cache/post_sqllm_quantized/${MODEL_BASENAME}-w${BITS}-${MODE_TAG}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_lambda${RBVT_LAMBDA_TAG}"

if [[ "${OVERWRITE_POSTPROCESS}" == "1" ]]; then
  rm -rf "${OUTPUT_QUANTIZED_PATH}"
fi

LAYER_COUNT="$(count_layers)"
DEVICES="$(resolve_devices)"
PACK_DEVICE="$(primary_device "${DEVICES}")"

COMMON_ARGS=(
  --model "${MODEL}"
  --mode "${MODE}"
  --stage apply
  --bits "${BITS}"
  --dataset "${DATASET}"
  --seq-len "${SEQ_LEN}"
  --num-examples "${NUM_EXAMPLES}"
  --n-calib "${N_CALIB}"
  --batch-size "${BATCH_SIZE}"
  --rbvt-lambda "${RBVT_LAMBDA}"
  --rbvt-topk "${RBVT_TOPK}"
  --row-chunk "${ROW_CHUNK}"
  --gap-floor "${GAP_FLOOR}"
  --cpu-count "${CPU_COUNT}"
)

if [[ "${ALLOW_OVERSHOOT}" == "1" ]]; then
  COMMON_ARGS+=(--allow-overshoot)
fi

if [[ "${OVERWRITE_POSTPROCESS}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

if [[ -n "${DEVICES}" ]]; then
  echo "[run_redpajama_postprocess_fast] Auto-selected devices: ${DEVICES}"
else
  echo "[run_redpajama_postprocess_fast] Falling back to device: ${DEVICE}"
fi

run_shards "${LAYER_COUNT}" "${DEVICES}" python rbvt_squeezellm.py "${COMMON_ARGS[@]}"

echo "[run_redpajama_postprocess_fast] Finalizing pack on ${PACK_DEVICE}"
PACK_ARGS=(
  --model "${MODEL}"
  --mode "${MODE}"
  --stage pack
  --bits "${BITS}"
  --dataset "${DATASET}"
  --seq-len "${SEQ_LEN}"
  --num-examples "${NUM_EXAMPLES}"
  --n-calib "${N_CALIB}"
  --batch-size "${BATCH_SIZE}"
  --rbvt-lambda "${RBVT_LAMBDA}"
  --rbvt-topk "${RBVT_TOPK}"
  --row-chunk "${ROW_CHUNK}"
  --gap-floor "${GAP_FLOOR}"
  --cpu-count "${CPU_COUNT}"
  --device "${PACK_DEVICE}"
)

if [[ "${ALLOW_OVERSHOOT}" == "1" ]]; then
  PACK_ARGS+=(--allow-overshoot)
fi

if [[ "${OVERWRITE_POSTPROCESS}" == "1" ]]; then
  PACK_ARGS+=(--overwrite)
fi

python rbvt_squeezellm.py "${PACK_ARGS[@]}"
