#!/usr/bin/env bash
set -euo pipefail

# Always operate in this script directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SBD_DIR="${SBD_DIR:-sbd}"
REPO_URL="${SBD_REPO_URL:-https://github.com/r-ccs-cms/sbd.git}"
CCCOM="${CCCOM:-mpiFCCpx}"
CCFLAGS="${CCFLAGS:--Nclang -std=c++17 -stdlib=libc++ -Kfast,openmp -Xpreprocessor -fopenmp}"
SYSLIB="${SYSLIB:--SSL2}"

if ! command -v "$CCCOM" >/dev/null 2>&1; then
    echo "Compiler '$CCCOM' not found in PATH. Load Fugaku compiler/MPI modules first." >&2
    exit 1
fi

if [ ! -d "$SBD_DIR" ]; then
    echo "Cloning SBD repo..."
    git clone "$REPO_URL" "$SBD_DIR"
else
    echo "SBD repo already exists."
fi

# Clean previous build
rm -f ./*.o ./diag

# Compile and link
echo "$CCCOM $CCFLAGS -c main.cc -o main.o -I$SBD_DIR/include"
$CCCOM $CCFLAGS -c main.cc -o main.o -I"$SBD_DIR/include"
echo "$CCCOM $CCFLAGS $SYSLIB -o diag main.o"
$CCCOM $CCFLAGS $SYSLIB -o diag main.o

echo "Build completed: $SCRIPT_DIR/diag"
