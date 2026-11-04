# QCSC Prefect SBD

Shared Python-side SBD Davidson solver integration for qcsc-prefect workflows.

This package provides:

- `SBDSolverJob` as a Prefect block (block type slug `sbd_solver_job`)
- `SBDResult`, the parsed solver output
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
from qcsc_prefect_sbd import SBDSolverJob

solver = SBDSolverJob.load("davidson-solver")
result = await solver.run(
    ci_strings=(alpha_strings, alpha_strings),
    one_body_tensor=h1,
    two_body_tensor=h2,
    norb=norb,
    nelec=nelec,
)
print(result.energy, result.carryover_bitstrings.shape)
```

`SBDSolverJob.run` takes the same arguments as `DiceSHCISolverJob.run`, but SBD solves a
narrower problem. `run` raises `ValueError` (see `check_sbd_input`) when:

- `nelec` is open-shell (`nelec[0] != nelec[1]`);
- `ci_strings[0]` and `ci_strings[1]` differ (SBD reads only the first set);
- the first determinant is not Hartree-Fock, `(1 << nelec[0]) - 1`;
- `spin_sq` is not `None` (SBD has no total spin target).

The block refers to a `CommandBlock`, an `ExecutionProfileBlock` and an `HPCProfileBlock`
by name. `algorithms/sbd/create_blocks.py` creates all four blocks for Miyabi, Fugaku or
Slurm.

Blocks saved before this package existed, when `SBDSolverJob` was defined in
`algorithms/sbd`, keep working: the block type name (`SBD Solver Job`), slug and fields
are unchanged.
