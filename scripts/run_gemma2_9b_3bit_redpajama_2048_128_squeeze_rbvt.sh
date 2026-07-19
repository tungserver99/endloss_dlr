#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[run_gemma2_9b_3bit_redpajama_2048_128_squeeze_rbvt] Redirecting to 4096/1024 default script."
exec bash "${SCRIPT_DIR}/run_gemma2_9b_3bit_redpajama_4096_1024_squeeze_rbvt.sh" "$@"
