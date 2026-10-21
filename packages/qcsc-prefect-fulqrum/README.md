# QCSC Prefect Fulqrum

[Fulqrum](https://github.com/qiskit-community/fulqrum) subspace eigensolver
integration for qcsc-prefect workflows. It is a drop-in alternative to the DICE
SHCI solver for SQD.

This package provides:

- `solve_fulqrum`: a Prefect-free solver core. It diagonalizes the molecular
  Hamiltonian in the subspace spanned by alpha/beta CI strings and returns the
  energy, orbital occupancies and amplitudes.
- `qcsc-fulqrum-solve <work_dir>`: the compute-node executable. It reads
  `input.npz` + `solver.json`, runs `solve_fulqrum`, and writes `output.npz` +
  `result.json`. On failure it writes `error.json` and exits with status 1. It
  imports neither Prefect nor qiskit-addon-sqd.
- `FulqrumSolverJob`: a Prefect block with the same `run()` signature and
  `SCIResult` return type as `DiceSHCISolverJob`. It writes the job directory,
  runs `qcsc-fulqrum-solve` through `run_job_from_blocks`, and reads the result
  back. `spin_sq` is accepted and ignored.
- `create_fulqrum_blocks` / `register_fulqrum_block_types`: create the command,
  execution-profile, HPC-profile and solver blocks for `local`, `slurm`,
  `miyabi` or `fugaku`.

Fulqrum itself is not installed from PyPI. Version 0.5 is built per site with
`native/build_fulqrum.sh`. Install this package into that environment with
`--no-deps`; the compute node then needs only numpy, scipy and fulqrum.

The optional [PRIMME](https://github.com/primme/primme) eigensolver
(`eigensolver="primme"`) is built by the same script with
`FULQRUM_WITH_PRIMME=1`. PyPI only has a source release, so the build needs a C
compiler, Python headers, `pkg-config` and a BLAS/LAPACK that has a `.pc` file
(for example `openblas-devel`). The default `scipy` eigensolver needs none of
this.

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
    fulqrum_executable="/opt/venvs/fulqrum/bin/qcsc-fulqrum-solve",
    ompthreads=8,
)

# Inside a flow:
solver = await FulqrumSolverJob.load("fulqrum-solver")
result = await solver.run(ci_strings, h1, eri, norb, nelec)  # SCIResult
```

The returned energy is electronic only, as with the DICE solver.
`solution.amplitudes[ia, ib]` (and `SCIResult.sci_state.amplitudes`) follows the
`qiskit_addon_sqd.fermion.SCIState` layout over `ci_strs_a` × `ci_strs_b`.

Fulqrum runs as one OpenMP process. `create_fulqrum_blocks` defaults to
`launcher="single"` with one task. It also exports `OMP_NUM_THREADS` when
`ompthreads` is set. If `OMP_NUM_THREADS` is unset, `qcsc-fulqrum-solve` falls
back to `SLURM_CPUS_PER_TASK`.

## Tests

- The io, CLI, block and solver-job tests run anywhere. The block and
  solver-job tests skip where Prefect is not installed.
- The numerical tests need fulqrum and PySCF, so they skip when those are not
  installed. Run them on a node where fulqrum is built, for example the
  `native/docker` testbed's `c1` node.
- `test_fulqrum_slurm_integration.py` submits real Slurm jobs. It needs
  `SLURM_INTEGRATION=1` and `sbatch`, and no Prefect server. See
  `native/docker/RUNBOOK.md` step 7.
