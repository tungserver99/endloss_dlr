#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

for CANDIDATE in "${ROOT_DIR}/.env" "${PWD}/.env"; do
  if [[ -f "${CANDIDATE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${CANDIDATE}"
    set +a
    echo "[load_env] Loaded ${CANDIDATE}"
    break
  fi
done
