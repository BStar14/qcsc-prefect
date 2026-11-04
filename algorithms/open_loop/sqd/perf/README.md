# MPI Performance Tuning Script

This module provides the `dice_perf_tuning` workflow, which automates performance tuning for the DICE solver.

The workflow internally runs multiple instances of the `sqd_2405_05068` workflow, each with different combinations of the following parameters:

- `mpiprocs`
- `num_nodes`
- `subspace_dim`

This enables systematic investigation of both **weak** and **strong scaling** behavior of the MPI-parallelized DICE solver in a single batch execution.

The results are automatically summarized and saved as a table artifact named `dice-performance`.

For a practical example of how to optimize MPI parameters for the DICE solver, refer to the notebook: [Tuning MPI Parameters for the DICE Solver](./tuning_mpi_parameters_with_dice.ipynb).

## 🚀 Getting Started

To deploy the `dice_perf_tuning` workflow from the repository root, activate the
virtual environment where the `sqd_dice` package is installed, and run:

```bash
cd /path/to/qcsc-prefect
uv run --project algorithms/open_loop/sqd python algorithms/open_loop/sqd/perf/tune_dice.py
```

This command deploys the `dice_perf_tuning` workflow on your Prefect server.
You can then specify the main flow parameters, along with a list of sub-parameter combinations to sweep:

```text
(mpiprocs, num_nodes, subspace_dim)
```
