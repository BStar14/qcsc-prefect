# QCSC Prefect Fulqrum

[Fulqrum](https://github.com/Qiskit/qiskit-fulqrum) subspace eigensolver
integration for qcsc-prefect workflows. It is a drop-in alternative to the DICE
SHCI solver for SQD.

This package provides:

- `solve_fulqrum`: a Prefect-free solver core. It diagonalizes the molecular
  Hamiltonian in the subspace spanned by alpha/beta CI strings on the CPU, on
  the GPUs of one node, or on one MPI rank per GPU across nodes. It returns the
  energy, orbital occupancies, carryover strings and amplitudes.
- `qcsc-fulqrum-solve <work_dir>`: the compute-node executable. It reads
  `input.npz` + `solver.json`, runs `solve_fulqrum`, and writes `output.npz` +
  `result.json`. On failure it writes `error.json` (`error.rank<k>.json` from
  process `k > 0` of a multi-process run) and exits with status 1. It imports
  neither Prefect nor qiskit-addon-sqd.
- `FulqrumSolverJob`: a Prefect block with the same `run()` signature and
  `SCIResult` return type as `DiceSHCISolverJob`. It writes the job directory,
  runs `qcsc-fulqrum-solve` through `run_job_from_blocks`, and reads the result
  back. `spin_sq` is accepted and ignored.
- `create_fulqrum_blocks` / `register_fulqrum_block_types`: create the command,
  execution-profile, HPC-profile and solver blocks for `local`, `slurm`,
  `miyabi` or `fugaku`.

Fulqrum itself is not installed from PyPI. It is built per site with
`native/build_fulqrum.sh`, by default fulqrum 1.0 at a pinned commit, into the
virtual environment `native/fulqrum`. The script installs this package there
with `--no-deps`, so the executable is `native/fulqrum/bin/qcsc-fulqrum-solve`;
the compute node needs only numpy, scipy and fulqrum, plus CuPy and mpi4py for
the GPU backends.
[native/README.md](native/README.md) has the CPU and GPU build recipes, the
launch layout and the runtime variables.

The optional [PRIMME](https://github.com/primme/primme) eigensolver
(`eigensolver="primme"`) is built by the same script with
`FULQRUM_WITH_PRIMME=1`. PyPI only has a source release, so the build needs a C
compiler, Python headers, `pkg-config` and a BLAS/LAPACK that has a `.pc` file
(for example `openblas-devel`). The default `scipy` eigensolver needs none of
this.

## Backends

`FulqrumSolverOptions(backend=...)`, or the `backend` field of
`FulqrumSolverJob`:

- `cpu` (default): one OpenMP process.
- `gpu`: one process on every GPU it sees (`fulqrum.gpu.GPUHamiltonian`), with
  the same `scipy` or `primme` eigensolver as `cpu`.
- `gpu-mpi`: one MPI rank per GPU, across nodes
  (`fulqrum.gpu.DistributedMPIGPUHamiltonian` and fulqrum's distributed PRIMME).
  The state stays distributed, so there are no amplitudes; the occupancies and
  the carryover come from the per-string weights summed over the ranks. Launch
  it with `srun --mpi=pmix` or `mpirun`, one rank per GPU.

The GPU backends need fulqrum built with `FULQRUM_WITH_GPU=1`, CuPy for the
CUDA major version of the toolkit (`cupy-cuda13x` by default, `cupy-cuda12x` for
CUDA 12), and for `gpu-mpi` mpi4py built against the site's MPI. The kernels
need at least two electrons per spin; smaller problems are solved on the CPU
with a warning. Subspaces of at most `dense_max_dim` determinants are solved
densely on the CPU with every backend. Under `gpu-mpi` a rank that fails writes
its error file and aborts `MPI.COMM_WORLD`, so the others do not hang.

## Usage

Direct, in-process:

```python
from qcsc_prefect_fulqrum import FulqrumSolverOptions, solve_fulqrum

solution = solve_fulqrum(
    ci_strings=(ci_strs_a, ci_strs_b),
    one_body_tensor=h1,
    two_body_tensor=eri,  # full 4-index, PySCF chemist notation
    norb=norb,
    nelec=(n_alpha, n_beta),
    options=FulqrumSolverOptions(tol=1e-8),
)
total_energy = solution.energy + nuclear_repulsion_energy
```

Through blocks, on a scheduler:

```python
from qcsc_prefect_fulqrum import FulqrumSolverJob, create_fulqrum_blocks

create_fulqrum_blocks(
    hpc_target="slurm",
    queue="normal",
    root_dir="/shared/fulqrum-jobs",
    fulqrum_executable="/path/to/qcsc-prefect/packages/qcsc-prefect-fulqrum/native/fulqrum/bin/qcsc-fulqrum-solve",
    ompthreads=8,
)

# Inside a flow:
solver = await FulqrumSolverJob.load("fulqrum-solver")
result = await solver.run(ci_strings, h1, eri, norb, nelec)  # SCIResult
```

The returned energy is electronic only, as with the DICE solver.
`solution.amplitudes[ia, ib]` (and `SCIResult.sci_state.amplitudes`) follows the
`qiskit_addon_sqd.fermion.SCIState` layout over `ci_strs_a` × `ci_strs_b`.

The `cpu` backend runs as one OpenMP process. `create_fulqrum_blocks` defaults
to `launcher="single"` with one task. It also exports `OMP_NUM_THREADS` when
`ompthreads` is set. If `OMP_NUM_THREADS` is unset, `qcsc-fulqrum-solve` falls
back to `SLURM_CPUS_PER_TASK`.

## Tests

- The io, CLI, block and solver-job tests run anywhere. The block and
  solver-job tests skip where Prefect is not installed.
- The GPU backend tests (`test_fulqrum_gpu_backends.py`) run anywhere, against
  fake fulqrum, CuPy and mpi4py modules.
- The numerical tests need fulqrum and PySCF, so they skip when those are not
  installed. Run them on a node where fulqrum is built.
- `test_fulqrum_gpu_integration.py` compares the GPU backends with the CPU on
  real GPUs. It needs a GPU build and `FULQRUM_GPU=1`, and runs under `mpirun`
  or `srun`; see [native/README.md](native/README.md#testing-the-gpu-build).
- `test_fulqrum_slurm_integration.py` submits real Slurm jobs. It needs
  `SLURM_INTEGRATION=1` and `sbatch`, and no Prefect server.
