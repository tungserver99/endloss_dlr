#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-redpajama}" bash "${SCRIPT_DIR}/ensure_squeeze_base.sh"
