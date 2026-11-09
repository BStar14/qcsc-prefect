#!/bin/bash
# Build fulqrum into a dedicated compute-side virtual environment.
#
# Usage: build_fulqrum.sh [venv_dir] [fulqrum_src_dir]
#
# venv_dir defaults to fulqrum/ next to this script. After fulqrum, the script
# installs qcsc-prefect-fulqrum from this checkout with --no-deps, so the
# compute-node executable is <venv_dir>/bin/qcsc-fulqrum-solve. If
# fulqrum_src_dir is omitted, fulqrum is cloned at FULQRUM_REF into external/fulqrum
# next to this script, with the submodules the build needs; pass "" as venv_dir
# to build a given source into the default venv. README.md next to this script
# has the CPU and GPU recipes.
#
# Environment:
#   PYTHON             interpreter used to create the venv (default: python3)
#   FULQRUM_REPO       git URL (default: upstream GitHub)
#   FULQRUM_REF        commit or tag to build (default: pinned commit below)
#   FQ_BUILD_PARALLEL  parallel compile jobs (default: nproc)
#   FQ_ARCH            optional CPU target; leave unset for a portable build
#   CC / CXX           compilers with OpenMP support (default: toolchain default)
#   FULQRUM_EXTRA_PIP  extra packages to install into the venv (e.g. test deps)
#   FULQRUM_WITH_PRIMME  1 to also build PyPI's PRIMME eigensolver (default: 0).
#                        Needs a C compiler, Python headers, pkg-config, and an
#                        LP64 BLAS/LAPACK with a .pc file (openblas, mkl, blis or lapack).
#   PRIMME_VERSION     primme version to build (default: 3.2.3)
#   FULQRUM_SKIP_CLI   1 to skip installing qcsc-prefect-fulqrum, for example
#                      when it is installed editable later (default: 0)
#
# GPU build: the CUDA kernels, GPUHamiltonian, and the distributed GPU solver
# with fulqrum's own patched PRIMME. Needs the CUDA toolkit (nvcc), an MPI with
# mpicc, and OpenBLAS ILP64.
#   FULQRUM_WITH_GPU   1 to build the GPU stack (default: 0)
#   FQ_CUDA_ARCH       CUDA target, e.g. sm_90 for H100/GH200; set it when the
#                      build host has no GPU (default: detected by fulqrum)
#   FQ_BLAS_LIB        path to libopenblas64_.so (ILP64, _64_ symbols); default:
#                      found by fulqrum, e.g. under $HOME/.local/opt/openblas-ilp64
#   CUDA_HOME          CUDA toolkit root (default: from nvcc on PATH)
#   MPICC              MPI compiler wrapper to build mpi4py with (default: mpicc)
#   FULQRUM_CUPY       CuPy requirement (default: cupy-cuda13x>=14,<15; use
#                      cupy-cuda12x for a CUDA 12 toolkit)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR=${1:-$SCRIPT_DIR/fulqrum}
# Absolute, since the checks below run from /.
case "$VENV_DIR" in /*) ;; *) VENV_DIR="$(pwd)/$VENV_DIR" ;; esac
SRC_DIR=${2:-}
PYTHON=${PYTHON:-python3}
FULQRUM_REPO=${FULQRUM_REPO:-https://github.com/Qiskit/qiskit-fulqrum.git}
FULQRUM_REF=${FULQRUM_REF:-2e2d66e6bafb617e7c685b3591b0a0018202d2cd}
FULQRUM_WITH_GPU=${FULQRUM_WITH_GPU:-0}
FULQRUM_CUPY=${FULQRUM_CUPY:-cupy-cuda13x>=14,<15}
FULQRUM_SKIP_CLI=${FULQRUM_SKIP_CLI:-0}
export FQ_BUILD_PARALLEL=${FQ_BUILD_PARALLEL:-$(nproc 2>/dev/null || echo 1)}

die() {
    echo "error: $*" >&2
    exit 1
}

# Whether an OpenBLAS ILP64 library is where fulqrum's build looks for one.
have_ilp64_openblas() {
    local dir prefix="$HOME/.local/opt/openblas-ilp64"
    # shellcheck disable=SC2046
    for dir in "$prefix/lib" "$prefix/lib64" ${CONDA_PREFIX:+"$CONDA_PREFIX/lib"} \
        $(echo "${LD_LIBRARY_PATH:-}" | tr ':' ' '); do
        compgen -G "$dir/libopenblas*64_*.so*" >/dev/null && return 0
    done
    return 1
}

# The CPU build needs the qiskit-addon-sqd-hpc submodule, the GPU build also
# third-party/primme. third-party/doctest is only for fulqrum's C++ tests.
SUBMODULES=(qiskit-addon-sqd-hpc)

if [ "$FULQRUM_WITH_GPU" = 1 ]; then
    SUBMODULES+=(third-party/primme)
    # Check the toolchain before cloning and compiling anything.
    command -v nvcc >/dev/null 2>&1 || [ -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ] ||
        die "FULQRUM_WITH_GPU=1 needs the CUDA toolkit: put nvcc on PATH or set CUDA_HOME"
    MPICC=${MPICC:-$(command -v mpicc || true)}
    { [ -n "$MPICC" ] && [ -x "$MPICC" ]; } ||
        die "FULQRUM_WITH_GPU=1 needs MPI: load the MPI module or set MPICC to its mpicc"
    if [ -n "${FQ_BLAS_LIB:-}" ]; then
        [ -f "$FQ_BLAS_LIB" ] || die "FQ_BLAS_LIB=$FQ_BLAS_LIB does not exist"
    elif ! have_ilp64_openblas; then
        cat >&2 <<EOF
error: FULQRUM_WITH_GPU=1 needs OpenBLAS ILP64 (64-bit integers, _64_ symbols),
and none was found in \$HOME/.local/opt/openblas-ilp64, \$CONDA_PREFIX/lib or
LD_LIBRARY_PATH. Set FQ_BLAS_LIB=/path/to/libopenblas64_.so, or build one into
the default prefix from an OpenBLAS release tree (fulqrum's GPU_INSTALL.md):
  make -j8 CC=gcc FC=gfortran INTERFACE64=1 SYMBOLSUFFIX=64_ LIBNAMESUFFIX=64_ \\
      DYNAMIC_ARCH=1 USE_THREAD=1 USE_OPENMP=0
  make install PREFIX=\$HOME/.local/opt/openblas-ilp64 INTERFACE64=1 \\
      SYMBOLSUFFIX=64_ LIBNAMESUFFIX=64_ DYNAMIC_ARCH=1 USE_THREAD=1 USE_OPENMP=0
An ordinary LP64 OpenBLAS does not work.
EOF
        exit 1
    fi
    export FQ_BUILD_GPU=1 MPICC
    for var in FQ_CUDA_ARCH FQ_BLAS_LIB CUDA_HOME; do
        if [ -n "${!var:-}" ]; then
            export "${var?}"
            echo "gpu build: $var=${!var}"
        fi
    done
fi

if [ -z "$SRC_DIR" ]; then
    SRC_DIR="$SCRIPT_DIR/external/fulqrum"
    rm -rf "$SRC_DIR"
    git clone "$FULQRUM_REPO" "$SRC_DIR"
    git -C "$SRC_DIR" checkout "$FULQRUM_REF"
    git -C "$SRC_DIR" submodule update --init --recursive "${SUBMODULES[@]}"
fi

if [ ! -d "$SRC_DIR/qiskit-addon-sqd-hpc/include" ]; then
    die "$SRC_DIR is missing the qiskit-addon-sqd-hpc submodule" \
        "(git submodule update --init --recursive qiskit-addon-sqd-hpc)"
fi
if [ "$FULQRUM_WITH_GPU" = 1 ] && [ ! -f "$SRC_DIR/third-party/primme/include/primme.h" ]; then
    die "$SRC_DIR is missing the third-party/primme submodule" \
        "(git submodule update --init --recursive third-party/primme)"
fi
if [ "$FULQRUM_SKIP_CLI" != 1 ] && [ ! -f "$SCRIPT_DIR/../pyproject.toml" ]; then
    die "$SCRIPT_DIR/.. is not the qcsc-prefect-fulqrum package: run this script from" \
        "packages/qcsc-prefect-fulqrum/native of a checkout, or set FULQRUM_SKIP_CLI=1"
fi

"$PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install "cython>=3.0.5" "numpy>=2.0" "setuptools>=77" wheel
# fulqrum's runtime dependencies. They are listed here rather than read from
# its requirements.txt, which at the pinned commit is just "." (resolved
# against the current directory, not the source tree).
"$VENV_DIR/bin/python" -m pip install "scipy>=1.14.1" orjson pyscf psutil
if [ "$FULQRUM_WITH_GPU" = 1 ]; then
    # mpi4py must match the MPI of the launcher, so it is never a binary wheel.
    MPICC="$MPICC" "$VENV_DIR/bin/python" -m pip install \
        --no-binary=mpi4py --no-cache-dir "mpi4py>=4.1,<5"
    "$VENV_DIR/bin/python" -m pip install "$FULQRUM_CUPY"
fi
# Not the gpu extra: it pins the CUDA 13 CuPy, and FULQRUM_CUPY already chose one.
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
        die "FULQRUM_WITH_PRIMME=1 but pkg-config finds no openblas/mkl/blis/lapack"
    fi
    echo "primme: BLAS/LAPACK via pkg-config '$blas_pc'"
    "$VENV_DIR/bin/python" -m pip install pkgconfig
    "$VENV_DIR/bin/python" -m pip install --no-build-isolation "primme==${PRIMME_VERSION:-3.2.3}"
fi
if [ -n "${FULQRUM_EXTRA_PIP:-}" ]; then
    # shellcheck disable=SC2086
    "$VENV_DIR/bin/python" -m pip install $FULQRUM_EXTRA_PIP
fi
if [ "$FULQRUM_SKIP_CLI" != 1 ]; then
    # The compute-node CLI; fulqrum, numpy and setuptools are already in the venv.
    "$VENV_DIR/bin/python" -m pip install --no-deps --no-build-isolation "$SCRIPT_DIR/.."
fi

# Import from outside the source tree so the installed package is checked.
cd /
"$VENV_DIR/bin/python" -c "import fulqrum; print('fulqrum', fulqrum.__version__)"
if [ "${FULQRUM_WITH_PRIMME:-0}" = 1 ]; then
    "$VENV_DIR/bin/python" -c "import primme; from importlib.metadata import version; print('primme', version('primme'))"
fi
if [ "$FULQRUM_WITH_GPU" = 1 ]; then
    # The verification of fulqrum's GPU_INSTALL.md. Only the device count needs
    # a GPU, so a build on a host without one warns there instead of failing.
    "$VENV_DIR/bin/python" - <<'PYCODE'
import cupy
import fulqrum
import fulqrum.gpu.distributed.kernel_launch  # noqa: F401
from fulqrum.gpu.eigensolver import _primme_mpi
from mpi4py import MPI

assert _primme_mpi.gpu_backend_available(), "_primme_mpi was built without the cuBLAS backend"
print("Fulqrum", fulqrum.__version__)
print("CuPy", cupy.__version__)
print("PRIMME cuBLAS backend available")
print("mpi4py on", MPI.Get_library_version().splitlines()[0])
try:
    count = cupy.cuda.runtime.getDeviceCount()
except cupy.cuda.runtime.CUDARuntimeError:
    count = 0
if count:
    print("CUDA devices:", count)
else:
    print("warning: no CUDA device on this host; rerun this check on a GPU node")
PYCODE
fi
if [ "$FULQRUM_SKIP_CLI" != 1 ]; then
    "$VENV_DIR/bin/qcsc-fulqrum-solve" --help >/dev/null
    echo "fulqrum executable: $VENV_DIR/bin/qcsc-fulqrum-solve"
else
    echo "fulqrum venv: $VENV_DIR (qcsc-prefect-fulqrum not installed: FULQRUM_SKIP_CLI=1)"
fi
