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
uv run --project algorithms/sqd python algorithms/sqd/perf/tune_dice.py
```

This command deploys the `dice_perf_tuning` workflow on your Prefect server.
You can then specify the main flow parameters, along with a list of sub-parameter combinations to sweep:

```text
(mpiprocs, num_nodes, subspace_dim)
```

`base_parameters` are the flow parameters of the SQD workflow, and
`base_parameters.solver_block_name` is the DICE solver block to tune. Each
sub-experiment is a separate SQD run, so set `sampling.qiskit_runtime_job_id` (or
`sampling.qrmi_result_dir`) to an earlier sampling job to sample only once, or use
`sampling.quantum_source = "random"`.
