# Fulqrum Integration

`qcsc-prefect-fulqrum` runs the [fulqrum](https://github.com/qiskit-community/fulqrum)
subspace eigensolver as a qcsc-prefect job. It is a drop-in alternative to the DICE SHCI
solver for SQD:

- `solve_fulqrum`, a Prefect-free solver core
- the `qcsc-fulqrum-solve` compute-node executable
- the `FulqrumSolverJob` Prefect block, with the same `run()` signature and `SCIResult`
  return type as `DiceSHCISolverJob`
- block creation helpers for `local`, `slurm`, `miyabi` and `fugaku`

Like the DICE integration, it does not install the solver itself. fulqrum 0.5 is not on PyPI
and must be built on each target system.

## Installation

The package is installed from the source checkout. It runs in two environments.

**Compute nodes** need only numpy, scipy and fulqrum. Build fulqrum into a dedicated
virtual environment, then install this package into it without dependencies:

```bash
cd /path/to/qcsc-prefect
packages/qcsc-prefect-fulqrum/native/build_fulqrum.sh /opt/venvs/fulqrum
/opt/venvs/fulqrum/bin/python -m pip install --no-deps packages/qcsc-prefect-fulqrum
```

`build_fulqrum.sh` clones fulqrum at a pinned commit unless a source directory is passed as
the second argument. It needs a C++ compiler with OpenMP support. Set `FULQRUM_WITH_PRIMME=1`
to also build the optional PRIMME eigensolver (`eigensolver="primme"`). PRIMME needs a C
compiler, Python headers, `pkg-config` and a BLAS/LAPACK with a `.pc` file, for example
`openblas-devel`. The script header lists the other environment variables.

The path of `/opt/venvs/fulqrum/bin/qcsc-fulqrum-solve` is the executable that the blocks
point to. The compute environment does not need Prefect or qiskit-addon-sqd.

**The workflow host** (where the Prefect process runs and jobs are submitted) installs the
package with its dependencies:

```bash
python -m pip install \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-executor \
  -e packages/qcsc-prefect-fulqrum
```

## Block Configuration

`create_fulqrum_blocks()` creates the command, execution-profile, HPC-profile and solver
blocks in one call:

```python
from qcsc_prefect_fulqrum import create_fulqrum_blocks

create_fulqrum_blocks(
    hpc_target="slurm",
    queue="normal",
    root_dir="/shared/fulqrum-jobs",
    fulqrum_executable="/opt/venvs/fulqrum/bin/qcsc-fulqrum-solve",
    ompthreads=8,
)
```

This saves `cmd-fulqrum-solver`, `exec-fulqrum-slurm`, `hpc-slurm-fulqrum` and the
`fulqrum-solver` `FulqrumSolverJob`. `root_dir` must be visible at the same path from the
workflow host and the compute nodes.

fulqrum runs as one OpenMP process, so the defaults are `launcher="single"`, one node and one
task. When `ompthreads` is set, `OMP_NUM_THREADS` is exported with the same value. If
`OMP_NUM_THREADS` is unset, `qcsc-fulqrum-solve` falls back to `SLURM_CPUS_PER_TASK`.

The solver block also holds the eigensolver settings: `eigensolver` (`scipy` or `primme`),
`hamiltonian_format` (`csr` or `matrix_free`), `tol`, `max_iter`, `dense_max_dim` and
`return_sci_state`.

## Running the Solver

Inside a flow, load the solver block and call it with the SQD subspace:

```python
from qcsc_prefect_fulqrum import FulqrumSolverJob

solver = await FulqrumSolverJob.load("fulqrum-solver")
result = await solver.run(ci_strings, one_body_tensor, two_body_tensor, norb, nelec)
```

The result is a `qiskit_addon_sqd.fermion.SCIResult`. Its energy is electronic only, as with
the DICE solver, so add the nuclear repulsion energy yourself. fulqrum applies no spin
penalty, so `spin_sq` is accepted and ignored.

Each call creates a `job_<timestamp>_<id>` directory under `root_dir` with the rendered
scheduler script, `input.npz`, `solver.json`, the scheduler output, and either `output.npz`
plus `result.json` or, on failure, `error.json`.

To call the solver without Prefect or a scheduler, use `solve_fulqrum()` directly; see the
package [README](https://github.com/qiskit-community/qcsc-prefect/tree/main/packages/qcsc-prefect-fulqrum).

## SQD Workflow

The `sqd_fulqrum` workflow in `algorithms/sqd` runs the SQD experiment of
arXiv:2405.05068 with this solver. Its README covers block creation from a TOML file
(`sqd-fulqrum-create-blocks`), deployment (`sqd-fulqrum-deploy`) and optional QRMI/Slurm
sampling.
