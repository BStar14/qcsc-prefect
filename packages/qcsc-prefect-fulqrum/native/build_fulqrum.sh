#!/bin/bash
# Build fulqrum into a dedicated compute-side virtual environment.
#
# Usage: build_fulqrum.sh <venv_dir> [fulqrum_src_dir]
#
# If fulqrum_src_dir is omitted, fulqrum is cloned (with submodules) at
# FULQRUM_REF into ./external/fulqrum.
#
# Environment:
#   PYTHON             interpreter used to create the venv (default: python3)
#   FULQRUM_REPO       git URL (default: upstream GitHub)
#   FULQRUM_REF        commit or tag to build (default: pinned commit below)
#   FQ_BUILD_PARALLEL  parallel compile jobs (default: nproc)
#   FQ_ARCH            optional CPU target; leave unset for a portable build
#   CC / CXX           compilers with OpenMP support (default: toolchain default)
#   FULQRUM_EXTRA_PIP  extra packages to install into the venv (e.g. test deps)
#   FULQRUM_WITH_PRIMME  1 to also build the optional PRIMME eigensolver (default: 0).
#                        Needs a C compiler, Python headers, pkg-config, and a
#                        BLAS/LAPACK with a .pc file (openblas, mkl, blis or lapack).
#   PRIMME_VERSION     primme version to build (default: 3.2.3)
set -euo pipefail

VENV_DIR=${1:?usage: build_fulqrum.sh <venv_dir> [fulqrum_src_dir]}
SRC_DIR=${2:-}
PYTHON=${PYTHON:-python3}
FULQRUM_REPO=${FULQRUM_REPO:-https://github.com/qiskit-community/fulqrum.git}
FULQRUM_REF=${FULQRUM_REF:-52d7c66ab0986580b00ad9382fe4d67bbe4d883e}
export FQ_BUILD_PARALLEL=${FQ_BUILD_PARALLEL:-$(nproc 2>/dev/null || echo 1)}

if [ -z "$SRC_DIR" ]; then
    SRC_DIR="$(pwd)/external/fulqrum"
    rm -rf "$SRC_DIR"
    git clone "$FULQRUM_REPO" "$SRC_DIR"
    git -C "$SRC_DIR" checkout "$FULQRUM_REF"
    git -C "$SRC_DIR" submodule update --init --recursive
fi

if [ ! -d "$SRC_DIR/qiskit-addon-sqd-hpc/include" ]; then
    echo "error: $SRC_DIR is missing submodules (clone with --recursive)" >&2
    exit 1
fi

"$PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install "cython>=3.0.5" numpy setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$SRC_DIR/requirements.txt"
"$VENV_DIR/bin/python" -m pip install --no-build-isolation "$SRC_DIR"
if [ "${FULQRUM_WITH_PRIMME:-0}" = 1 ]; then
    # primme ships only an sdist. Build it against this venv's numpy and Cython;
    # the pkgconfig module lets its setup.py find BLAS/LAPACK (otherwise it falls
    # back to -llapack -lblas, which many sites do not provide).
    blas_pc=""
    for p in openblas mkl blis lapack; do
        if pkg-config --exists "$p" 2>/dev/null; then blas_pc=$p; break; fi
    done
    if [ -z "$blas_pc" ]; then
        echo "error: FULQRUM_WITH_PRIMME=1 but pkg-config finds no openblas/mkl/blis/lapack" >&2
        exit 1
    fi
    echo "primme: BLAS/LAPACK via pkg-config '$blas_pc'"
    "$VENV_DIR/bin/python" -m pip install pkgconfig
    "$VENV_DIR/bin/python" -m pip install --no-build-isolation "primme==${PRIMME_VERSION:-3.2.3}"
fi
if [ -n "${FULQRUM_EXTRA_PIP:-}" ]; then
    # shellcheck disable=SC2086
    "$VENV_DIR/bin/python" -m pip install $FULQRUM_EXTRA_PIP
fi

# Import from outside the source tree so the installed package is checked.
cd /
"$VENV_DIR/bin/python" -c "import fulqrum; print('fulqrum', fulqrum.__version__)"
if [ "${FULQRUM_WITH_PRIMME:-0}" = 1 ]; then
    "$VENV_DIR/bin/python" -c "import primme; from importlib.metadata import version; print('primme', version('primme'))"
fi
