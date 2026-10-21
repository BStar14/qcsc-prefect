# Sample-based Quantum Diagonalization

This package provides implementation for the QCSC algorithm demonstrated in the paper [Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer](https://arxiv.org/abs/2405.05068).

## 🚀 Getting Started

Before starting, make sure:

- You have completed [How to Set Up IBM Quantum Access Credentials for Prefect](../../docs/howto/howto_setup_prefect_qiskit.md).
- You have completed [How to Set Up the MDX Workflow Server for QCSC Execution](../../docs/howto/howto_setup_mdx_server.md).
- You have built the DICE executable under `packages/qcsc-prefect-dice/native` and know its absolute path.

From the repository root, install the local qcsc-prefect packages and the SQD workflow:

```bash
cd /path/to/qcsc-prefect

uv pip install --no-deps \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-executor \
  -e packages/qcsc-prefect-dice

uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/sqd
```

After installation, you can deploy the workflow in a Prefect server:

```bash
sqd-deploy
```

In the Prefect UI, the flow parameters now include:

- `quantum_source`: choose `real-device` or `random`
- `random_seed`: base seed used when `quantum_source = "random"`
- `runner_name`: defaults to the `QuantumRuntime` block `ibm-runner`

Set `quantum_source = "random"` when you want to skip IBM Quantum Runtime and use deterministic pseudo-random bitstrings instead.

Before running on Miyabi or Fugaku, create the DICE-related blocks and
`sampler_options` variable:

```bash
cp algorithms/sqd/sqd_blocks.example.toml algorithms/sqd/sqd_blocks.toml
python algorithms/sqd/create_blocks.py --config algorithms/sqd/sqd_blocks.toml --hpc-target miyabi
```

For Fugaku, start from `algorithms/sqd/sqd_blocks.fugaku.example.toml` and pass
`--hpc-target fugaku` instead.

For a Fugaku build, use `packages/qcsc-prefect-dice/native/build_dice_fugaku.sh`
as the starting point and make sure the same runtime libraries are reflected in
`fugaku_spack_modules`, `modules`, or `environments` in the TOML when needed.

Set `dice_executable` in the TOML file to the built binary, for example:

```toml
dice_executable = "/path/to/qcsc-prefect/packages/qcsc-prefect-dice/native/bin/Dice"
```

On Fugaku, `work_dir` and `dice_executable` should point to a shared path that
compute nodes can read, for example a `/vol...` location.

See the [Run Hybrid Workflow](../../docs/tutorials/run_sqd_workflow.md) tutorial for the end-to-end steps.

## ⚛️ SQD with the fulqrum eigensolver

`sqd_fulqrum` runs the same experiment, but diagonalizes each subspace with
[fulqrum](https://github.com/qiskit-community/fulqrum) through the
`FulqrumSolverJob` block of `packages/qcsc-prefect-fulqrum`. The quantum stage
(LUCJ ansatz, transpilation, `real-device` or `random` sampling) and the
configuration recovery loop are the same as in `sqd_dice`. `sqd_dice` is not
changed or imported.

Differences from `sqd_dice`:

- Each batch is one scheduler job running `qcsc-fulqrum-solve` (one OpenMP
  process). The batches of a recovery round are submitted concurrently.
- The default target is a generic Slurm cluster. `local`, `miyabi` and `fugaku`
  also work, given a fulqrum build on that system.
- fulqrum applies no spin penalty, so `spin_sq` is ignored (the flow logs a
  warning when it is nonzero).
- The flow is `sqd_2405_05068_fulqrum`, its solver block defaults to
  `sqd-fulqrum-solver`, and its artifact key is
  `sqd-fulqrum-configuration-recovery`.

### Runbook (Prefect Cloud)

1. **Compute nodes:** build fulqrum with
   `packages/qcsc-prefect-fulqrum/native/build_fulqrum.sh`, then install
   `qcsc-prefect-fulqrum` into that environment with `--no-deps`. Note the path
   of its `qcsc-fulqrum-solve` entry point. The compute environment needs no
   Prefect.
2. **Submit host** (where the Prefect process runs and `sbatch` works): install
   the workflow with the `fulqrum` extra:

   ```bash
   uv pip install --no-deps \
     -e packages/qcsc-prefect-core -e packages/qcsc-prefect-adapters \
     -e packages/qcsc-prefect-blocks -e packages/qcsc-prefect-executor \
     -e packages/qcsc-prefect-dice -e packages/qcsc-prefect-fulqrum
   uv pip install -e algorithms/qcsc_workflow_utility
   uv pip install -e "algorithms/sqd[fulqrum]"
   ```

3. Log in to Prefect Cloud on the submit host (`prefect cloud login`, or set
   `PREFECT_API_URL` and `PREFECT_API_KEY`).
4. Create the blocks and the `sampler_options` variable:

   ```bash
   cp algorithms/sqd/sqd_fulqrum_blocks.slurm.example.toml algorithms/sqd/sqd_fulqrum_blocks.slurm.toml
   # edit queue, work_dir, fulqrum_executable, ompthreads
   sqd-fulqrum-create-blocks --config algorithms/sqd/sqd_fulqrum_blocks.slurm.toml --dry-run
   sqd-fulqrum-create-blocks --config algorithms/sqd/sqd_fulqrum_blocks.slurm.toml
   ```

   `--dry-run` prints the resolved settings without contacting Prefect.
   `work_dir` must be on a file system that both the submit host and the compute
   nodes see.
5. Serve the deployment: `sqd-fulqrum-deploy`.
6. Start a run from the Prefect UI. Use `quantum_source = "random"` first, then
   `real-device` with the `ibm-runner` `QuantumRuntime` block. For example:

   ```json
   {
     "molecule": {"atom": "N 0 0 0; N 0 0 1.1", "basis": "sto-3g"},
     "sqd": {"subspace_dim": 400, "num_batches": 2, "max_iterations": 2},
     "quantum_source": "random"
   }
   ```

Each solve leaves a `job_*` directory under `work_dir` with the rendered script,
`input.npz`, `solver.json`, `output.npz`, `result.json` (or `error.json`) and the
scheduler output.

### Tests

The tests in `algorithms/sqd/tests` cover only `sqd_fulqrum`. Run them from the
repository root:

```bash
uv run pytest algorithms/sqd/tests
```

Tests skip when their dependencies are missing. The flow tests need `ffsim`,
`prefect-qiskit` and `qcsc_workflow_utility`. The fulqrum comparison needs a
fulqrum build. `test_sqd_fulqrum_slurm_integration.py` needs
`SLURM_INTEGRATION=1` and `sbatch`. See
`packages/qcsc-prefect-fulqrum/native/docker/RUNBOOK.md` for the testbed
commands. No test contacts a Prefect server.

## 🧭 MPI Parameter Tuning

The [perf](./perf/) module also includes example code for implementing a meta workflow.
This workflow internally runs SQD workflows with different flow parameters, and measures the weak and strong scaling of the DICE solver.

## 🧑‍💻 Contribution Guidelines

This package is a reference implementation of the SQD experiment described in the publication.
It is considered feature-complete and is now in maintenance mode.

Further contributions are limited to bug fixes and improvements to Prefect usage patterns.
Workflow developers may use this as a testbed for workflow technology research.
