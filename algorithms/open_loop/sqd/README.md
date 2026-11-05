# Sample-based Quantum Diagonalization

This package provides implementation for the QCSC algorithm demonstrated in the paper [Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer](https://arxiv.org/abs/2405.05068).

It contains two workflows that differ only in the subspace eigensolver:

| Flow | Package | Solver block | Deploy command |
| --- | --- | --- | --- |
| `sqd_2405_05068_fulqrum` | `sqd_fulqrum` | `FulqrumSolverJob` (`qcsc-prefect-fulqrum`), default `sqd-fulqrum-solver` | `sqd-fulqrum-deploy` |
| `sqd-2405-05068` | `sqd_dice` | `DiceSHCISolverJob` (`qcsc-prefect-dice`), default `sqd-solver` | `sqd-deploy` |

Both run the same stages from `sqd_common`.

## 🧩 Workflow stages

The workflow follows the [Qiskit Patterns](https://quantum.cloud.ibm.com/docs/en/guides/intro-to-patterns)
steps, with one Prefect task per step. Each task carries its step as a tag, so the
run page can be filtered by stage:

```text
compute_molecular_integrals [map] ──────────────┐
   ▼ elec_props                                 │
get_lucj_params [optimize]                      │
   ▼ lucj_params                                │
quantum_sampling [execute]                      │
   ├ build_lucj_circuit                         │
   └ transpile_lucj_circuit (IBM Runtime path)  │
 | load_qrmi_result / load_runtime_job          │
   ▼ bool_array                                 │
sqd_loop [post-process] ◄───────────────────────┘
```

- **Map** (`compute_molecular_integrals`): molecular integrals from a geometry or an
  FCIDUMP file, CCSD amplitudes and the initial orbital occupancies. The CCSD kernel
  restarts `ccsd_maxiter` times from its previous amplitudes.
- **Optimize** (`get_lucj_params`): LUCJ parameters from the CCSD t2 amplitudes,
  restricted to the interactions a heavy-hex lattice implements without SWAP gates.
  `compress` optimizes the truncated factorization and `max_ab_pairs` caps the
  alpha-beta interactions.
- **Execute** (`quantum_sampling`): builds the LUCJ circuit, compiles it and samples
  it. Through IBM Quantum Runtime the workflow host compiles for the backend Target.
  Through QRMI the compute node compiles inside the Slurm job, with the same pass
  manager and seed. `load_qrmi_result` and `load_runtime_job` take this place when a
  run reuses the samples of an earlier job.
- **Post-process** (`sqd_loop`): self-consistent configuration recovery. Every round
  diagonalizes `num_batches` subspaces concurrently, one solver job each, and then
  saves a checkpoint, so the recovery can continue after the last completed round.

Prefect draws an arrow between two tasks when one task's result is passed unchanged
into the other. The run graph therefore shows Map → Optimize → Execute →
Post-process, plus the arrow from Map to Post-process that carries the Hamiltonian.

With `quantum_source = "random"` no circuit runs: the flow draws uniform random
bitstrings and goes from Map straight to Post-process.

## 🎛️ Flow parameters

Each flow takes one argument, `parameters`, grouped by stage:

| Group | Fields |
| --- | --- |
| `molecule` | `{atom, basis, symmetry}` or `{fcidump_file}`, plus `spin_sq` and `ccsd_maxiter` (default 4) |
| `circuit` | `n_lucj_layers`, `compress`, `compress_maxiter`, `max_ab_pairs`, `use_reset_mitigation`, transpiler settings |
| `sampling` | `quantum_source`, `shots`, `random_seed`, `runner_name`, `sampler_options`, `qrmi_runtime_config`, `qiskit_runtime_job_id`, `qrmi_result_dir` |
| `sqd` | `subspace_dim`, `num_batches`, `max_iterations`, `checkpoint` (default on), `resume_checkpoint` |
| `solver_block_name` | Solver block to load |

`default_params/` holds complete examples for N2 (6-31G, 1s cores frozen, 16 orbitals,
32 qubits), whose integrals are in `data/n2_631g_fc.fcidump`
(`data/make_n2_fcidump.py` regenerates it). A relative `fcidump_file` is resolved
from the working directory, then from the repository root.

| File | Sampling |
| --- | --- |
| `fulqrum_n2_random.json` | Random bitstrings, no QPU (a first check of the blocks) |
| `fulqrum_n2.json` | IBM Quantum Runtime through the `ibm-runner` block |
| `fulqrum_n2_qrmi.json` | QRMI on Slurm (fill in `qrmi_runtime_config`) |
| `fulqrum_n2_reload.json` | Samples of an earlier IBM Quantum Runtime job, with a larger subspace |
| `dice_n2.json` | IBM Quantum Runtime, diagonalized with DICE |

The flow loads the solver block before any stage runs, so a wrong block name fails
in seconds.

## ♻️ Resuming and reusing samples

- **Retry a failed run.** The Map, Optimize and Execute results are persisted, and
  `sqd_loop` is never cached. When a run fails in Post-process (for example on a
  walltime limit), *Retry* it from the Prefect UI. The retry reuses the integrals,
  the LUCJ parameters and the samples, and runs only `sqd_loop` again. No new QPU job
  is submitted. The results live in the Prefect result storage of the host that
  serves the deployment.

  With `sqd.checkpoint` on (the default), `sqd_loop` continues from the last
  completed recovery round instead of round 1. The log shows
  `[ckpt] RESUME from …: completed 3 step(s)`, then `SQD iteration 4/5`.
- **Continue a recovery with more rounds.** After every round `sqd_loop` writes
  `ckpt_t00_w0.npz` and `source.json` to
  `<solver block root_dir>/sqd_checkpoints/<flow run id>/`. The `sqd-checkpoint`
  artifact gives the path. To continue that recovery in a new run, copy the path
  into `sqd.resume_checkpoint` and raise `sqd.max_iterations`. A checkpoint with
  32 of 100 rounds done, resumed with `max_iterations = 200`, continues at round
  33/200 from the stored occupancies and best energy.

  `source.json` holds the resume key. The new run must use the same samples (the
  same `qrmi_result_dir`, `qiskit_runtime_job_id` or random seed and shots), the
  same Hamiltonian, `subspace_dim` and `num_batches`. Otherwise it fails before
  any solver job and names what differs. The given checkpoint is only read; the
  new run writes its own. If a run's own checkpoint no longer matches, it is moved
  to `<flow run id>.prev-<time>/` and the recovery starts at round 1.

  Disk use is small: about 16 bytes per orbital plus a few kB per run, rewritten in
  place, under 1 MB even for 1000 orbitals. Nothing removes old run folders, and
  the solver job directories under the same `root_dir` take far more space. Set
  `sqd.checkpoint` to `false` to write nothing.
- **Reuse the samples in a new run.** Every Execute task writes an
  `sqd-sampling-source` artifact that records where the samples came from.
  - For QRMI, copy its `qrmi_result_dir` into `sampling.qrmi_result_dir` of a new run.
  - For IBM Quantum Runtime, copy the job ID from the `job-metrics` artifact into
    `sampling.qiskit_runtime_job_id`.

  The new run then loads those samples instead of sampling, so you can change
  `sqd` (for example `subspace_dim`) or the solver block without QPU time. Keep
  `circuit.use_reset_mitigation` the same as in the sampling run.

## 🚀 Getting Started

Before starting, make sure:

- You have completed [How to Set Up IBM Quantum Access Credentials for Prefect](../../../docs/howto/howto_setup_prefect_qiskit.md).
- You have built the solver for your site: fulqrum with
  `packages/qcsc-prefect-fulqrum/native/build_fulqrum.sh`, or DICE under
  `packages/qcsc-prefect-dice/native`.

### SQD with the fulqrum eigensolver

Each batch is one scheduler job running `qcsc-fulqrum-solve` (one OpenMP process).
The default target is a generic Slurm cluster; `local`, `miyabi` and `fugaku` also
work, given a fulqrum build on that system. fulqrum applies no spin penalty, so the
flow logs a warning when `spin_sq` is nonzero.

1. **Compute nodes:** build fulqrum, then install `qcsc-prefect-fulqrum` into that
   environment with `--no-deps`. Note the path of its `qcsc-fulqrum-solve` entry
   point. The compute environment needs no Prefect.
2. **Submit host** (where the Prefect process runs and `sbatch` works): install the
   workflow with the `fulqrum` extra:

   ```bash
   uv pip install --no-deps \
     -e packages/qcsc-prefect-core -e packages/qcsc-prefect-adapters \
     -e packages/qcsc-prefect-blocks -e packages/qcsc-prefect-executor \
     -e packages/qcsc-prefect-dice -e packages/qcsc-prefect-fulqrum
   uv pip install -e algorithms/qcsc_workflow_utility
   uv pip install -e "algorithms/open_loop/sqd[fulqrum]"
   ```

3. Log in to Prefect Cloud on the submit host (`prefect cloud login`, or set
   `PREFECT_API_URL` and `PREFECT_API_KEY`).
4. Create the blocks:

   ```bash
   cp algorithms/open_loop/sqd/sqd_fulqrum_blocks.slurm.example.toml algorithms/open_loop/sqd/sqd_fulqrum_blocks.slurm.toml
   # edit queue, work_dir, fulqrum_executable, ompthreads
   sqd-fulqrum-create-blocks --config algorithms/open_loop/sqd/sqd_fulqrum_blocks.slurm.toml --dry-run
   sqd-fulqrum-create-blocks --config algorithms/open_loop/sqd/sqd_fulqrum_blocks.slurm.toml
   ```

   `--dry-run` prints the resolved settings without contacting Prefect. `work_dir`
   must be on a file system that both the submit host and the compute nodes see.
5. Serve the deployment and keep the process running: `sqd-fulqrum-deploy`
   (or `python algorithms/open_loop/sqd/deploy.py fulqrum` from a checkout).
6. In Prefect Cloud, open the deployment, choose *Run* → *Custom run* and paste the
   JSON of `default_params/fulqrum_n2_random.json`. Then run `fulqrum_n2.json` or
   `fulqrum_n2_qrmi.json` on a QPU.

To run once without a deployment, from the repository root:

```bash
python algorithms/open_loop/sqd/exec.py fulqrum algorithms/open_loop/sqd/default_params/fulqrum_n2_random.json
```

Each solve leaves a `job_*` directory under `work_dir` with the rendered script,
`input.npz`, `solver.json`, `output.npz`, `result.json` (or `error.json`) and the
scheduler output.

### SQD with the DICE eigensolver

Install the DICE workflow on the submit host:

```bash
uv pip install --no-deps \
  -e packages/qcsc-prefect-core -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks -e packages/qcsc-prefect-executor \
  -e packages/qcsc-prefect-dice
uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/open_loop/sqd
```

Create the DICE blocks for Miyabi or Fugaku:

```bash
cp algorithms/open_loop/sqd/sqd_blocks.example.toml algorithms/open_loop/sqd/sqd_blocks.toml
python algorithms/open_loop/sqd/create_blocks.py --config algorithms/open_loop/sqd/sqd_blocks.toml --hpc-target miyabi
```

For Fugaku, start from `algorithms/open_loop/sqd/sqd_blocks.fugaku.example.toml` and pass
`--hpc-target fugaku`. For a Fugaku build, use
`packages/qcsc-prefect-dice/native/build_dice_fugaku.sh` as the starting point and
reflect the same runtime libraries in `fugaku_spack_modules`, `modules` or
`environments` in the TOML. Set `dice_executable` to the built binary, for example:

```toml
dice_executable = "/path/to/qcsc-prefect/packages/qcsc-prefect-dice/native/bin/Dice"
```

On Fugaku, `work_dir` and `dice_executable` should point to a shared path that
compute nodes can read, for example a `/vol...` location.

Then serve the deployment with `sqd-deploy` and start a run with
`default_params/dice_n2.json`.

### QRMI/Slurm sampling

Both workflows can sample through QRMI. Install the QRMI integration from this
checkout on the workflow host:

```bash
uv pip install -e 'packages/qcsc-prefect-qiskit[slurm]' \
  -e packages/qcsc-prefect-fulqrum -e algorithms/qcsc_workflow_utility \
  -e 'algorithms/open_loop/sqd[fulqrum,qrmi]'
```

On the compute node, the Python that runs the QRMI worker needs QRMI and its
`task_runner`, `qcsc-prefect-qiskit` (the worker), and this package from the same
revision. The compilation hook `sqd_common.transpile:transpile_for_target` imports
only Qiskit and qiskit-ibm-runtime, so ffsim, Prefect-Qiskit and the solvers are not
needed there:

```bash
uv pip install -e packages/qcsc-prefect-qiskit
uv pip install --no-deps -e algorithms/open_loop/sqd
```

Set `sampling.qrmi_runtime_config` in the run parameters, for example:

```json
{
  "root_dir": "/shared/qrmi-jobs",
  "scheduler": "slurm",
  "command_block_name": "cmd-sqd-fulqrum-qrmi",
  "execution_profile_block_name": "exec-sqd-fulqrum-qrmi",
  "hpc_profile_block_name": "hpc-slurm-sqd-fulqrum-qrmi",
  "resource": "test_heron"
}
```

The quantum CommandBlock must invoke the QRMI worker, the execution profile must
use `launcher="single"`, and `slurm_qpu` must match `resource`. Configure QRMI Batch
mode and shared storage as described in the
[QRMI/Slurm guide](../../../docs/howto/howto_qrmi_slurm_sampler.md).

`sqd-fulqrum-create-blocks` creates these three blocks when the config has a `[qrmi]`
table with `resource` and `python_executable` (see the commented table in
`sqd_fulqrum_blocks.slurm.example.toml`), or with `--qrmi-resource` and
`--qrmi-python-executable`. The QRMI HPC profile is separate from the solver's,
because it requests the QPU. The command prints the matching `qrmi_runtime_config`
to paste into each run; `--dry-run` includes it.

With `qrmi_runtime_config`, `runner_name` and `sampler_options` are not used. SQD
supplies its own preparation and transpiler hooks, so `qrmi_runtime_config` must not
set `prepare`, `transpiler` or `transpile_config`. On the workflow host, ffsim's
`PRE_INIT` passes run and the remaining ffsim gates are expanded before QPY
serialization. On the compute node, the same pass manager as the IBM Runtime path
(preset stages plus the SABRE layout search of `parameters.circuit`) runs against
the Target acquired inside Slurm, with the transpiler seed the host path would draw.
Both paths therefore compile to the same circuit for the same Target. The
`meas`/`test` registers and reset postselection are preserved.

The QPU is held by the Slurm job while the node compiles, so give the QRMI execution
profile enough CPU threads for the layout search.

### Tests

Run the tests in `algorithms/open_loop/sqd/tests` from the repository root:

```bash
uv run pytest algorithms/open_loop/sqd/tests
```

Tests skip when their dependencies are missing. The flow tests need `ffsim`,
`prefect-qiskit` and `qcsc_workflow_utility`; the fulqrum comparison needs a fulqrum
build. `test_sqd_common_flow.py` checks that each stage receives the exact object the
previous stage returned, which is what the Prefect graph arrows depend on.
`test_qrmi_runtime.py` covers the QRMI hooks, the QPY handoff and reloading a QRMI
job directory offline. `test_sqd_fulqrum_slurm_integration.py` and
`test_sqd_fulqrum_qrmi_slurm_integration.py` need `SLURM_INTEGRATION=1` and
`sbatch`; the QRMI one replaces QRMI and `task_runner` with fakes and requests no
QPU. No test contacts a Prefect server.

## 🧭 MPI Parameter Tuning

The [perf](./perf/) module also includes example code for implementing a meta workflow.
This workflow internally runs SQD workflows with different flow parameters, and measures the weak and strong scaling of the DICE solver.

## 🧑‍💻 Contribution Guidelines

This package is a reference implementation of the SQD experiment described in the publication.
It is considered feature-complete and is now in maintenance mode.

Further contributions are limited to bug fixes and improvements to Prefect usage patterns.
Workflow developers may use this as a testbed for workflow technology research.
