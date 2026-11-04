#!/usr/bin/env bash
set -euo pipefail

# Slurm build assumptions:
# - An MPI C++ compiler wrapper (mpicxx or mpic++) is available in PATH.
# - The compiler supports C++17 and OpenMP.
# - OpenBLAS development headers/libraries are installed and linkable with -lopenblas.
# - Git is available to clone the SBD repository if it is not already present.


# Always operate in this script directory to avoid accidental cleanup in cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_URL="${SBD_REPO_URL:-https://github.com/r-ccs-cms/sbd.git}"
SBD_COMMIT="${SBD_COMMIT:-d3210615ec36581ec3a286b8e339492f20b4b529}"
SBD_DIR="${SBD_DIR:-${SCRIPT_DIR}/sbd}"

# Default portable build flags
CXXFLAGS=("-std=c++17" "-fopenmp" "-O3")

# Optional CPU-specific optimization:
# For systems with Intel Sapphire Rapids CPUs, additional architecture-specific
# optimization flags may improve performance. Choose flags appropriate for the
# compiler used by your MPI wrapper (mpicxx/mpic++).
#
# For example, when the MPI wrapper uses an Intel compiler, adjust the flags
# for the target CPU architecture and compiler as needed:
# CXXFLAGS=("-std=c++17" "-axSAPPHIRERAPIDS,CORE-AVX512" "-qopenmp" "-O3")

if command -v mpicxx >/dev/null 2>&1; then
    CCCOM="mpicxx"
elif command -v mpic++ >/dev/null 2>&1; then
    CCCOM="mpic++"
else
    echo "No MPI C++ compiler wrapper (mpicxx or mpic++) was found." >&2
    exit 1
fi

if [ ! -d "$SBD_DIR" ]; then
    echo "Cloning SBD repo..."
    git clone "$REPO_URL" "$SBD_DIR"
else
    echo "SBD repo already exists: $SBD_DIR"
fi

echo "Checking out SBD commit: $SBD_COMMIT"
git -C "$SBD_DIR" checkout "$SBD_COMMIT"

# Clean previous build
rm -f "$SCRIPT_DIR"/*.o "$SCRIPT_DIR"/diag

# Compile and link
"$CCCOM" "${CXXFLAGS[@]}" -c "$SCRIPT_DIR/main.cc" -o "$SCRIPT_DIR/main.o" -I"$SBD_DIR/include"
"$CCCOM" "${CXXFLAGS[@]}" -o "$SCRIPT_DIR/diag" "$SCRIPT_DIR/main.o" -lopenblas

echo "Build completed: $SCRIPT_DIR/diag"
