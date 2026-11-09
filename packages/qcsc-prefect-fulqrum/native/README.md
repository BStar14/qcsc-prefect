# Building fulqrum for `qcsc-fulqrum-solve`

`build_fulqrum.sh` builds [fulqrum](https://github.com/Qiskit/fulqrum) into a
dedicated virtual environment on the target system, `native/fulqrum` next to
the script by default. It clones the pinned commit `2e2d66e` (fulqrum 1.0.0,
branch `v1.0.0-rc`) into `native/external/fulqrum` with the submodules the build
needs, and then installs this package into the environment with `--no-deps`.
From the repository root:

```bash
bash packages/qcsc-prefect-fulqrum/native/build_fulqrum.sh
```

The compute node then runs
`/path/to/qcsc-prefect/packages/qcsc-prefect-fulqrum/native/fulqrum/bin/qcsc-fulqrum-solve`,
which the script prints at the end. The header of `build_fulqrum.sh` lists its
options. Pass another virtual environment directory as the first argument, and a
fulqrum checkout as the second to build that instead of cloning.
`FULQRUM_SKIP_CLI=1` leaves this package out, for an environment that installs
it later, for example in editable mode.

## Backends

`FulqrumSolverOptions.backend` (the `backend` field of `FulqrumSolverJob`)
selects how a subspace is diagonalized. Subspaces of at most `dense_max_dim`
determinants are always solved densely on the CPU.

| Backend | Processes | Operator | Eigensolver | Result |
| --- | --- | --- | --- | --- |
| `cpu` | 1, OpenMP | `SubspaceHamiltonian` (CSR or matrix-free) | `scipy` or `primme` | energy, occupancies, amplitudes |
| `gpu` | 1, all GPUs it sees | `fulqrum.gpu.GPUHamiltonian` | `scipy` or `primme` | energy, occupancies, amplitudes |
| `gpu-mpi` | 1 MPI rank per GPU, any number of nodes | `fulqrum.gpu.DistributedMPIGPUHamiltonian` | fulqrum's distributed PRIMME | energy, occupancies, no amplitudes |

- `gpu` builds one worker per visible device, so `CUDA_VISIBLE_DEVICES` selects
  the GPUs. The eigensolver, tolerance and start vector are those of `cpu`.
- `gpu-mpi` keeps the state distributed. Each rank folds its tile of the
  ground state into the per-string weights, which are summed over the ranks; the
  occupancies and the carryover come from these weights, so no rank holds the
  full vector. Rank `k` of a node uses GPU `k % device_count`, with the node-local
  rank from `OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID` or `SLURM_LOCALID`.
  `eigensolver` and `hamiltonian_format` do not apply. The solve converges when
  the residual is below `tol` (`tol=0` means `1e-12`), and `max_iter` caps the
  number of matrix-vector products.
- The GPU kernels need at least two electrons per spin. A smaller problem logs a
  warning and is solved on the CPU (rank 0 under `gpu-mpi`).
- Under `gpu-mpi` the process index is the `MPI.COMM_WORLD` rank. A failing rank
  writes `error.rank<k>.json` (`error.json` for rank 0) and calls
  `MPI.COMM_WORLD.Abort(1)`, so the other ranks do not wait in a collective until
  the walltime. The submit side reports the first error file it finds.

## CPU build

Needs a C++17 compiler with OpenMP and Python 3.11 or later:

```bash
bash packages/qcsc-prefect-fulqrum/native/build_fulqrum.sh
```

`FULQRUM_WITH_PRIMME=1` also builds PyPI's `primme` for `eigensolver="primme"`.
It compiles from source against an LP64 BLAS/LAPACK found by `pkg-config`.

## GPU build

`FULQRUM_WITH_GPU=1` builds the CUDA kernels, `GPUHamiltonian`, and the
distributed solver with fulqrum's own patched PRIMME (the `third-party/primme`
submodule). It needs:

- the CUDA toolkit with `nvcc`, cuBLAS and cuSPARSE (`nvcc` on `PATH`, or
  `CUDA_HOME`), and a host compiler that toolkit supports;
- an MPI with `mpicc` (or `MPICC`); mpi4py is built from source against it, so
  load the same MPI that launches the jobs;
- OpenBLAS ILP64: 64-bit integers and `_64_` symbols, as `libopenblas64_.so`.
  The script looks in `$HOME/.local/opt/openblas-ilp64`, `$CONDA_PREFIX/lib` and
  `LD_LIBRARY_PATH`, or takes `FQ_BLAS_LIB`; it stops with a build recipe when
  none is found. An ordinary LP64 OpenBLAS does not work.

```bash
# Build OpenBLAS ILP64 once, from an OpenBLAS release tree:
make -j8 CC=gcc FC=gfortran INTERFACE64=1 SYMBOLSUFFIX=64_ LIBNAMESUFFIX=64_ \
    DYNAMIC_ARCH=1 USE_THREAD=1 USE_OPENMP=0
make install PREFIX=$HOME/.local/opt/openblas-ilp64 INTERFACE64=1 \
    SYMBOLSUFFIX=64_ LIBNAMESUFFIX=64_ DYNAMIC_ARCH=1 USE_THREAD=1 USE_OPENMP=0

# Then, with the site's CUDA and MPI modules loaded:
FULQRUM_WITH_GPU=1 FQ_CUDA_ARCH=sm_90 \
    bash packages/qcsc-prefect-fulqrum/native/build_fulqrum.sh packages/qcsc-prefect-fulqrum/native/fulqrum-gpu
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `FULQRUM_WITH_GPU` | `0` | `1` builds the GPU stack (`FQ_BUILD_GPU=1` for fulqrum) |
| `FQ_CUDA_ARCH` | detected | CUDA target, e.g. `sm_90` for H100 and GH200; set it on a build host without a GPU |
| `FQ_BLAS_LIB` | searched | full path to `libopenblas64_.so` |
| `CUDA_HOME` | from `nvcc` | CUDA toolkit root |
| `MPICC` | `mpicc` | MPI compiler wrapper for mpi4py |
| `FULQRUM_CUPY` | `cupy-cuda13x>=14,<15` | CuPy requirement |

CuPy has one wheel per CUDA major version, and it must match the toolkit fulqrum
is built with. The default is for CUDA 13; with a CUDA 12 toolkit set
`FULQRUM_CUPY="cupy-cuda12x"`. Never install both. Both have aarch64 wheels, so
the same recipe works on Grace Hopper (GH200) nodes.

The build ends with fulqrum's own installation check: it imports CuPy, the CUDA
kernels and the distributed PRIMME extension, checks that PRIMME has its cuBLAS
backend, and prints the MPI library mpi4py uses. On a build host without a GPU
it warns that no device is visible; run it again on a compute node:

```bash
packages/qcsc-prefect-fulqrum/native/fulqrum-gpu/bin/python -c 'import cupy; print(cupy.cuda.runtime.getDeviceCount())'
mpirun -n 2 packages/qcsc-prefect-fulqrum/native/fulqrum-gpu/bin/python -c \
    'from mpi4py import MPI; print(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)'
```

## Launching

- `cpu`: one task with `OMP_NUM_THREADS` cores.
- `gpu`: one task on one node, with the GPUs of that node.
- `gpu-mpi`: one rank per GPU. Under Slurm, `srun --mpi=pmix` with
  `--ntasks-per-node` equal to the GPUs per node and `--gpus-per-task=1` (or
  `--gpus-per-node`). Elsewhere `mpirun -n <total GPUs>` with as many ranks per
  node as it has GPUs; on a system with one GPU per node, such as GH200 nodes,
  that is one rank per node.

Every rank runs `qcsc-fulqrum-solve <job_dir>`; the job directory must be on a
file system all nodes share.

Runtime environment:

| Variable | Purpose |
| --- | --- |
| `FQ_DIST_BACKEND` | Transport between ranks: `auto` (default; NCCL when every rank supports it and has its own GPU, otherwise host MPI), `nccl`, `host_mpi` or `cuda_mpi` (needs CUDA-aware MPI) |
| `CUPY_CACHE_DIR` | Where CuPy caches compiled kernels (default `~/.cupy/kernel_cache`). The first run compiles them; point it at a persistent directory every node can reach when the home directory is not |
| `CUDA_VISIBLE_DEVICES` | Devices the `gpu` backend uses, and those `gpu-mpi` ranks pick from |
| `OMP_NUM_THREADS`, `FQ_BUILD_THREADS` | CPU threads fulqrum uses to build the GPU tables |

## Testing the GPU build

`tests/test_fulqrum_gpu_integration.py` compares `gpu` and `gpu-mpi` with `cpu`
on LiH, N2 and triplet O2: the energy to `1e-8`, the occupancies against the
diagonal of fulqrum's one-body RDM (`compute_rdms(rank=1)`), and the carryover.
It also runs `qcsc-fulqrum-solve` under `gpu-mpi`. It needs PySCF in the build
environment (a fulqrum dependency) plus pytest, and runs only with
`FULQRUM_GPU=1`. From the repository root, on a GPU node:

```bash
packages/qcsc-prefect-fulqrum/native/fulqrum-gpu/bin/python -m pip install pytest

# One process: the gpu backend on every GPU it sees, gpu-mpi on one rank.
FULQRUM_GPU=1 packages/qcsc-prefect-fulqrum/native/fulqrum-gpu/bin/python -m pytest \
    -p no:cacheprovider \
    packages/qcsc-prefect-fulqrum/tests/test_fulqrum_gpu_integration.py

# One rank per GPU; with Slurm, srun --mpi=pmix -n 2 --gpus-per-task=1 instead.
FULQRUM_GPU=1 mpirun -n 2 packages/qcsc-prefect-fulqrum/native/fulqrum-gpu/bin/python -m pytest \
    -p no:cacheprovider \
    packages/qcsc-prefect-fulqrum/tests/test_fulqrum_gpu_integration.py
```

Every rank runs the whole file in the same order, as the distributed solves are
collective; the `gpu` tests run on rank 0. The CLI test creates its job directory
in the current directory, which must be shared by all ranks. The
`eigensolver="primme"` cases of `gpu` skip unless PyPI's `primme` is installed
(`FULQRUM_WITH_PRIMME=1`).
