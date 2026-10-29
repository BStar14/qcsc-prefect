# Sample-based Quantum Diagonalization

This package provides implementation for the QCSC algorithm demonstrated in the paper [Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer](https://arxiv.org/abs/2405.05068).

It contains two variants of the workflow that differ only in the subspace eigensolver:

- `sqd_dice` diagonalizes with the DICE SHCI solver (`qcsc-prefect-dice`). The
  steps below set it up.
- `sqd_fulqrum` diagonalizes with fulqrum (`qcsc-prefect-fulqrum`) and can
  optionally sample through QRMI/Slurm. See "SQD with the fulqrum eigensolver"
  below.

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
configuration recovery loop are the same as in `sqd_dice`, which `sqd_fulqrum`
does not import.

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

### QRMI/Slurm sampling

To sample through QRMI, install the workflow and QRMI integration from this
checkout on the workflow host:

```bash
uv pip install -e 'packages/qcsc-prefect-qiskit[slurm]' \
  -e packages/qcsc-prefect-fulqrum -e algorithms/qcsc_workflow_utility \
  -e 'algorithms/sqd[fulqrum,qrmi]'
```

On the compute node, the Python that runs the QRMI worker needs QRMI and its
`task_runner`, `qcsc-prefect-qiskit` (the worker), and `sqd_fulqrum` from the
same revision. The compilation hook `sqd_fulqrum.transpile:transpile_for_target`
imports only Qiskit and qiskit-ibm-runtime, so ffsim, Prefect-Qiskit and fulqrum
are not needed there:

```bash
uv pip install -e packages/qcsc-prefect-qiskit
uv pip install --no-deps -e algorithms/sqd
```

Use `quantum_source="real-device"` and pass `qrmi_runtime_config` as a top-level
flow argument (alongside `parameters`). For example, in the Prefect run form:

```json
{
  "root_dir": "/shared/qrmi-jobs",
  "scheduler": "slurm",
  "command_block_name": "my-qrmi-command",
  "execution_profile_block_name": "my-qrmi-execution",
  "hpc_profile_block_name": "my-qrmi-hpc",
  "resource": "test_heron"
}
```

The quantum CommandBlock must invoke the QRMI worker, the execution profile
must use `launcher="single"`, and `slurm_qpu` must match `resource`. Configure
QRMI Batch mode and shared storage as described in the
[QRMI/Slurm guide](../../docs/howto/howto_qrmi_slurm_sampler.md).
`FulqrumSolverJob` and its classical execution blocks are unchanged.

`sqd-fulqrum-create-blocks` creates these three blocks when the config has a
`[qrmi]` table with `resource` and `python_executable` (see the commented table
in `sqd_fulqrum_blocks.slurm.example.toml`), or with `--qrmi-resource` and
`--qrmi-python-executable`. The QRMI HPC profile is separate from the solver's,
because it requests the QPU. The command prints the matching
`qrmi_runtime_config`, and `--dry-run` includes it, so it can be pasted into
each run.

With QRMI settings, `runner_name` is ignored. SQD supplies its own preparation
and transpiler hooks, so `qrmi_runtime_config` must not set `prepare`,
`transpiler` or `transpile_config`. On the workflow host, ffsim's `PRE_INIT`
passes run and the remaining ffsim gates are expanded before QPY serialization.
On the compute node, the same pass manager as the IBM Runtime path (preset
stages plus the SABRE layout search of `parameters.circuit`) runs against the
Target acquired inside Slurm, with the transpiler seed the host path would draw.
Both paths therefore compile to the same circuit for the same Target. The
`meas`/`test` registers and reset postselection are preserved. Without
`qrmi_runtime_config`, the existing IBM Runtime path remains in use;
`quantum_source="random"` continues to bypass both runtimes.

The QPU is held by the Slurm job while the node compiles, so give the QRMI
execution profile enough CPU threads for the layout search.

### Tests

The tests in `algorithms/sqd/tests` cover only `sqd_fulqrum`. Run them from the
repository root:

```bash
uv run pytest algorithms/sqd/tests
```

Tests skip when their dependencies are missing. The flow tests need `ffsim`,
`prefect-qiskit` and `qcsc_workflow_utility`. The fulqrum comparison needs a
fulqrum build. `test_qrmi_runtime.py` covers the QRMI hooks and QPY handoff
offline. `test_sqd_fulqrum_slurm_integration.py` and
`test_sqd_fulqrum_qrmi_slurm_integration.py` need `SLURM_INTEGRATION=1` and
`sbatch`; the QRMI one replaces QRMI and `task_runner` with fakes and requests
no QPU. No test contacts a Prefect server.

## 🧭 MPI Parameter Tuning

The [perf](./perf/) module also includes example code for implementing a meta workflow.
This workflow internally runs SQD workflows with different flow parameters, and measures the weak and strong scaling of the DICE solver.

## 🧑‍💻 Contribution Guidelines

This package is a reference implementation of the SQD experiment described in the publication.
It is considered feature-complete and is now in maintenance mode.

Further contributions are limited to bug fixes and improvements to Prefect usage patterns.
Workflow developers may use this as a testbed for workflow technology research.
