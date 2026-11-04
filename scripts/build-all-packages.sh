#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
PYTHON_BIN="${PYTHON:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Could not find python or python3 on PATH." >&2
    exit 1
  fi
fi

packages=(
  "packages/qcsc-prefect-core"
  "packages/qcsc-prefect-blocks"
  "packages/qcsc-prefect-adapters"
  "packages/qcsc-prefect-executor"
  "packages/qcsc-prefect-qiskit"
  "packages/qcsc-prefect-dice"
  "packages/qcsc-prefect-sbd"
  "packages/qcsc-prefect"
)

rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

for package in "${packages[@]}"; do
  "${PYTHON_BIN}" -m build --outdir "${DIST_DIR}" "${ROOT_DIR}/${package}"
done

"${PYTHON_BIN}" -m twine check "${DIST_DIR}"/*
