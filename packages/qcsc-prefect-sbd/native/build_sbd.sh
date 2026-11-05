#!/usr/bin/env bash
set -euo pipefail

# Always operate in this script directory to avoid accidental cleanup in cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load modules (Miyabi defaults).
module load intel/2023.2.0
module load impi/2021.10.0

REPO_URL="${SBD_REPO_URL:-https://github.com/r-ccs-cms/sbd.git}"
SBD_COMMIT="${SBD_COMMIT:-$(cat "$SCRIPT_DIR/SBD_REVISION")}"
SBD_DIR="${SBD_DIR:-${SCRIPT_DIR}/sbd}"
CXXFLAGS=("-std=c++17" "-axSAPPHIRERAPIDS,CORE-AVX512" "-qopenmp" "-O3")

if command -v mpiicpc >/dev/null 2>&1; then
    CCCOM="mpiicpc"
elif command -v mpiicpx >/dev/null 2>&1; then
    CCCOM="mpiicpx"
else
    echo "Neither mpiicpc nor mpiicpx was found. Load Intel oneAPI modules first." >&2
    exit 1
fi

if [ ! -d "$SBD_DIR" ]; then
    echo "Cloning SBD repo..."
    git clone "$REPO_URL" "$SBD_DIR"
else
    echo "SBD repo already exists: $SBD_DIR"
fi

# Use the revision matching main.cc on every platform, including existing clones.
if ! git -C "$SBD_DIR" cat-file -e "${SBD_COMMIT}^{commit}" 2>/dev/null; then
    git -C "$SBD_DIR" fetch origin "$SBD_COMMIT"
fi
git -C "$SBD_DIR" checkout --detach "$SBD_COMMIT"
echo "Using SBD commit: $(git -C "$SBD_DIR" rev-parse HEAD)"

# Clean previous build
rm -f "$SCRIPT_DIR"/*.o "$SCRIPT_DIR"/diag

# Compile and link
"$CCCOM" "${CXXFLAGS[@]}" -c "$SCRIPT_DIR/main.cc" -o "$SCRIPT_DIR/main.o" -I"$SBD_DIR/include"
"$CCCOM" "${CXXFLAGS[@]}" -o "$SCRIPT_DIR/diag" "$SCRIPT_DIR/main.o" -qmkl=parallel

echo "Build completed: $SCRIPT_DIR/diag"
