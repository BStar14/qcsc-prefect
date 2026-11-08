# QCSC Prefect SBD

Shared Python-side SBD Davidson solver integration for qcsc-prefect workflows.

This package provides:

- `SBDSolverJob` as a Prefect block (block type slug `sbd_solver_job`)
- `SBDResult`, the parsed solver output
- `create_sbd_blocks` / `register_sbd_block_types`: create the command,
  execution-profile, HPC-profile and solver blocks for `local`, `slurm`, `miyabi` or
  `fugaku`
- SBD input/output file handling and scheduler execution through qcsc-prefect blocks

It is intended to be reused by multiple algorithms. Installing this package does not
build, download, vendor, or install the external SBD executable (`diag`). The executable
must be compiled separately for each target HPC environment.

Source-tree native build helper assets live under:

```text
packages/qcsc-prefect-sbd/native
```

They are not part of the Python package install. See [native/README.md](./native/README.md)
for the build scripts and how to point `sbd_executable` at the separately built binary.

## Usage Example

```python
from qcsc_prefect_sbd import SBDSolverJob, create_sbd_blocks

create_sbd_blocks(
    hpc_target="slurm",
    queue="normal",
    root_dir="/shared/sbd-jobs",
    sbd_executable="/abs/path/to/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag",
    num_nodes=2,
    mpiprocs=4,
    mpi_options=["--mpi=pmix"],
)

# Inside a flow:
solver = await SBDSolverJob.load("sbd-solver")
result = await solver.run(
    ci_strings=(alpha_strings, beta_strings),
    one_body_tensor=h1,
    two_body_tensor=h2,
    norb=norb,
    nelec=nelec,
)
print(result.energy, result.carryover_bitstrings.shape)
```

`create_sbd_blocks` checks the request before it saves any block: local execution takes
no `modules` or `pre_commands`, `gpus_per_node` must fit the target, `solver_mode="gpu"`
needs `resource_class="gpu"`, and `task_comm_size x adet_comm_size x bdet_comm_size` must
divide the MPI rank count (`num_nodes x mpiprocs`). The launcher defaults to `single`
(local), `srun` (Slurm), `mpiexec.hydra` (Miyabi) or `mpiexec` (Fugaku).

## Closed and open shells

`SBDSolverJob.run` takes the same arguments as `DiceSHCISolverJob.run`. SBD diagonalizes
in the product of the alpha and beta determinant sets with the restricted integrals, so it
solves closed-shell and restricted open-shell (ROHF, `nelec[0] != nelec[1]`) problems.

- When the alpha and beta sets are the same, SBD reads `AlphaDets.bin` for both spins.
- When they differ (always for `nelec[0] != nelec[1]`), the beta set goes to
  `BetaDets.bin` (`--bdetfile`), and `SBDResult.carryover_bitstrings_b` holds the beta
  carryover. It is None when one set is shared.

`run` raises `ValueError` (see `check_sbd_input`) when:

- the alpha set does not start with its Hartree-Fock determinant, `(1 << nelec[0]) - 1`,
  or the beta set with `(1 << nelec[1]) - 1`;
- `spin_sq` is not `None` (SBD has no total spin target).

Each determinant must also have `nelec[0]` (alpha) or `nelec[1]` (beta) set bits and a
set must not repeat a determinant. The FCIDUMP has no core energy, so `energy` is the
electronic energy, as with the DICE solver.

The block refers to a `CommandBlock`, an `ExecutionProfileBlock` and an `HPCProfileBlock`
by name. `algorithms/closed_loop/sqd_de/create_blocks.py` creates all four blocks for Miyabi, Fugaku or
Slurm.

Blocks saved before this package existed, when `SBDSolverJob` was defined in
`algorithms/closed_loop/sqd_de`, keep working: the block type name (`SBD Solver Job`) and
slug are unchanged, and `carryover_type`, which they lack, defaults to 1.
