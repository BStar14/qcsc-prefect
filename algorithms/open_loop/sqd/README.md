# Sample-based Quantum Diagonalization

This package (`sqd` 2.0) implements the QCSC algorithm demonstrated in the paper [Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer](https://arxiv.org/abs/2405.05068).

It provides one Prefect flow, `sqd` (deploy with `sqd-deploy`). The solver block that
diagonalizes each subspace is chosen per run with `solver_block_ref`:

| `solver_block_ref` | Solver block | Install |
| --- | --- | --- |
| `fulqrum_solver_job/<name>` (default `sqd-fulqrum-solver`) | `FulqrumSolverJob` (`qcsc-prefect-fulqrum`) | `sqd` |
| `dice_shci_solver_job/<name>` | `DiceSHCISolverJob` (`qcsc-prefect-dice`) | `sqd[dice]` |
| `sbd_solver_job/<name>` | `SBDSolverJob` (`qcsc-prefect-sbd`) | `sqd[sbd]` |

Quantum sampling goes through a `QrmiRuntimeBlock` (default `qrmi-runtime`), either
from the workflow process (`scheduler = "iqp"`) or in a Slurm job (`scheduler = "slurm"`).

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
   └ build_lucj_circuit                         │
 | load_qrmi_result / load_runtime_job          │
   ▼ bool_array                                 │
sqd_loop [post-process] ◄───────────────────────┘
```

- **Map** (`compute_molecular_integrals`): molecular integrals from a geometry or an
  FCIDUMP file, CCSD amplitudes and the initial orbital occupancies. The CCSD kernel
  restarts `ccsd_maxiter` times from its previous amplitudes.
- **Optimize** (`get_lucj_params`): LUCJ parameters from the CCSD t2 amplitudes,
  restricted to the interactions a heavy-hex lattice implements without SWAP gates
  (alpha-beta pairs every 4th orbital, at most 5).
  `compress` (on by default, `compress_maxiter` 20) optimizes the truncated
  factorization.
- **Execute** (`quantum_sampling`): builds the LUCJ circuit and samples it through the
  QRMI runtime block, which compiles it for the backend Target (see
  [Quantum sampling](#quantum-sampling)). `load_qrmi_result` and `load_runtime_job`
  take this place when a run reuses the samples of an earlier job.
- **Post-process** (`sqd_loop`): self-consistent configuration recovery. Every round
  diagonalizes `num_batches` subspaces concurrently, one solver job each, and then
  saves a checkpoint, so the recovery can continue after the last completed round.

Prefect draws an arrow between two tasks when one task's result is passed unchanged
into the other. The run graph therefore shows Map → Optimize → Execute →
Post-process, plus the arrow from Map to Post-process that carries the Hamiltonian.

With `quantum_source = "random"` no circuit runs: the flow draws uniform random
bitstrings and goes from Map straight to Post-process.

The flow loads the solver block before any stage runs, and checks the molecule
against it right after Map, so a wrong block name, a missing solver extra, or input
the solver cannot solve fails before any quantum job.

## 🎛️ Flow parameters

The flow takes one argument, `parameters`. The Prefect Cloud run form shows its groups
and fields in this order, following the Qiskit Patterns stages:

| Group | Fields |
| --- | --- |
| `molecule` | `{atom, basis, symmetry}` or `{fcidump_file}`, plus `spin_sq` (number or `null`) and `ccsd_maxiter` (default 4) |
| `circuit` | `n_lucj_layers` (1), `compress` (on), `compress_maxiter` (20), `use_reset_mitigation`, `configure_transpiler` (off) and `transpiler` |
| `sampling` | `quantum_source`, `shots` (10000), `runtime_block_name`, `random_seed`, `qrmi_result_dir`, `qiskit_runtime_job_id`, `runner_name` (reload only) |
| `sqd` | `subspace_dim` (90000), `num_batches` (4), `max_iterations` (5), `checkpoint` (on), `resume_checkpoint` |
| `solver_block_ref` | `<block type slug>/<block name>`, default `fulqrum_solver_job/sqd-fulqrum-solver` |

Only `molecule` must be set. The form still marks the three `sqd` fields as required,
so that they stand out. `circuit.transpiler` holds the optimization level and the SABRE
layout search (layout trials 10000, SWAP trials 10, max iterations 8). It applies only
when `configure_transpiler` is on; otherwise these defaults are used.

`default_params/` holds complete examples for N2 (6-31G, 1s cores frozen, 16 orbitals,
32 qubits), whose integrals are in `data/n2_631g_fc.fcidump`
(`data/make_n2_fcidump.py` regenerates it). A relative `fcidump_file` is resolved
from the working directory, then from the repository root.

| File | Solver | Sampling |
| --- | --- | --- |
| `n2_random.json` | fulqrum | Random bitstrings, no QPU (a first check of the blocks) |
| `n2.json` | fulqrum | The `qrmi-runtime` block |
| `n2_reload.json` | fulqrum | Samples of an earlier IBM Quantum job, with a larger subspace |
| `n2_dice.json` | DICE | The `qrmi-runtime` block |
| `n2_sbd.json` | SBD | Random bitstrings, `spin_sq = null` |

### Solver selection

The slug of `solver_block_ref` selects the block class and an unknown slug is
rejected; Prefect's hyphenated spelling (`sbd-solver-job`) is accepted too. DICE and
SBD are imported only when a run selects them; without their extra the run fails
with the `pip install` command to run.

- **fulqrum** has no spin penalty: a nonzero `spin_sq` is ignored with a warning.
- **DICE** uses `spin_sq` as a spin penalty.
- **SBD** solves closed-shell systems only, uses one determinant set for alpha and
  beta, needs the Hartree-Fock determinant first, and has no spin target, so
  `spin_sq` must be `null`. Input SBD cannot solve fails with `ValueError` before
  sampling. For SBD, every subspace puts Hartree-Fock first and keeps
  `int(sqrt(subspace_dim)) - 1` other determinants, so it is not the same subspace
  that fulqrum or DICE gets for the same settings.

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
  same Hamiltonian, `subspace_dim` and `num_batches`, and the same solver type (the
  slug of `solver_block_ref`): SBD draws other subspaces than fulqrum or DICE, so a
  recovery continues only with the solver that started it. The solver block's name
  and job settings, such as its walltime, may change. Otherwise the run fails before
  any solver job and names what differs. The given checkpoint is only read; the
  new run writes its own. If a run's own checkpoint no longer matches, it is moved
  to `<flow run id>.prev-<time>/` and the recovery starts at round 1.

  Disk use is small: about 16 bytes per orbital plus a few kB per run, rewritten in
  place, under 1 MB even for 1000 orbitals. Nothing removes old run folders, and
  the solver job directories under the same `root_dir` take far more space. Set
  `sqd.checkpoint` to `false` to write nothing.
- **Reuse the samples in a new run.** Every Execute task writes an
  `sqd-sampling-source` artifact that records where the samples came from.
  - For `scheduler = "slurm"`, copy its `qrmi_result_dir` into
    `sampling.qrmi_result_dir` of a new run.
  - For `scheduler = "iqp"`, copy its `qiskit_runtime_job_id` into
    `sampling.qiskit_runtime_job_id`. The job is read with the credentials of the
    IBM `QuantumRuntime` block named by `sampling.runner_name` (default `ibm-runner`).

  The new run then loads those samples instead of sampling, so you can change
  `sqd` (for example `subspace_dim`) or the solver without QPU time. Keep
  `circuit.use_reset_mitigation` the same as in the sampling run.

## 🚀 Getting Started

Before starting, make sure:

- For `quantum_source = "real-device"`, you have IBM Quantum credentials for QRMI. See
  the [QRMI Sampler guide](../../../docs/howto/howto_qrmi_slurm_sampler.md).
- You have built the solver for your site: fulqrum with
  `packages/qcsc-prefect-fulqrum/native/build_fulqrum.sh`, DICE under
  `packages/qcsc-prefect-dice/native`, or SBD under `packages/qcsc-prefect-sbd/native`.

### Install

The package is installed from a source checkout. On the submit host (where the
Prefect process runs and the scheduler commands work), from the repository root:

```bash
uv pip install --no-deps \
  -e packages/qcsc-prefect-core -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks -e packages/qcsc-prefect-executor \
  -e 'packages/qcsc-prefect-qiskit[slurm]' -e packages/qcsc-prefect-fulqrum
uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/open_loop/sqd          # fulqrum
```

For DICE or SBD, also install their packages and the extra, for example
`uv pip install --no-deps -e packages/qcsc-prefect-dice` and
`uv pip install -e "algorithms/open_loop/sqd[dice]"` (`[sbd]`, or `[all]` for both).

### SQD with the fulqrum eigensolver

Each batch is one scheduler job running `qcsc-fulqrum-solve` (one OpenMP process).
The default target is a generic Slurm cluster; `local`, `miyabi` and `fugaku` also
work, given a fulqrum build on that system.

1. **Compute nodes:** build fulqrum, then install `qcsc-prefect-fulqrum` into that
   environment with `--no-deps`. Note the path of its `qcsc-fulqrum-solve` entry
   point. The compute environment needs no Prefect.
2. **Submit host:** install the workflow as above.
3. Log in to Prefect Cloud on the submit host (`prefect cloud login`, or set
   `PREFECT_API_URL` and `PREFECT_API_KEY`).
4. Create the blocks:

   ```bash
   cp algorithms/open_loop/sqd/sqd_fulqrum_blocks.slurm.example.toml algorithms/open_loop/sqd/sqd_fulqrum_blocks.slurm.toml
   # edit queue, work_dir, fulqrum_executable, ompthreads, and [qrmi] or [runtime]
   sqd-create-blocks --config algorithms/open_loop/sqd/sqd_fulqrum_blocks.slurm.toml --dry-run
   sqd-create-blocks --config algorithms/open_loop/sqd/sqd_fulqrum_blocks.slurm.toml
   ```

   `--dry-run` prints the resolved settings without contacting Prefect. `work_dir`
   must be on a file system that both the submit host and the compute nodes see.
5. Serve the deployment and keep the process running: `sqd-deploy`
   (or `python algorithms/open_loop/sqd/deploy.py` from a checkout).
6. In Prefect Cloud, open the deployment `sqd`, choose *Run* → *Custom run* and paste
   the JSON of `default_params/n2_random.json`. Then run `n2.json` on a QPU.

To run once without a deployment, from `algorithms/open_loop/sqd`:

```bash
python exec.py default_params/n2_random.json
```

Each solve leaves a `job_*` directory under `work_dir` with the rendered script,
`input.npz`, `solver.json`, `output.npz`, `result.json` (or `error.json`) and the
scheduler output.

### SQD with the DICE eigensolver

Create the DICE blocks for Miyabi or Fugaku:

```bash
cp algorithms/open_loop/sqd/sqd_blocks.example.toml algorithms/open_loop/sqd/sqd_blocks.toml
export QRMI_IBM_QCS_IAM_APIKEY="<your API key>"  # only with a [runtime] table for IQP
sqd-create-blocks --solver dice --config algorithms/open_loop/sqd/sqd_blocks.toml --hpc-target miyabi
```

For Fugaku, start from `algorithms/open_loop/sqd/sqd_blocks.fugaku.example.toml` and pass
`--hpc-target fugaku`. Its `[runtime]` tables are commented out, so it works for
random mode as copied. For a Fugaku build, use
`packages/qcsc-prefect-dice/native/build_dice_fugaku.sh` as the starting point and
reflect the same runtime libraries in `fugaku_spack_modules`, `modules` or
`environments` in the TOML. Set `dice_executable` to the built binary, for example:

```toml
dice_executable = "/path/to/qcsc-prefect/packages/qcsc-prefect-dice/native/bin/Dice"
```

On Fugaku, `work_dir` and `dice_executable` should point to a shared path that
compute nodes can read, for example a `/vol...` location.

Then start a run of the `sqd` deployment with `default_params/n2_dice.json`, whose
`solver_block_ref` is `dice_shci_solver_job/sqd-solver`.

### SQD with the SBD eigensolver

Create an `SBDSolverJob` block with
[`algorithms/closed_loop/sqd_de/create_blocks.py`](../../closed_loop/sqd_de/README.md)
and set `solver_block_ref = "sbd_solver_job/<name>"`, with `spin_sq = null`, as in
`default_params/n2_sbd.json`.

## Quantum sampling

With `quantum_source = "real-device"`, the flow loads the `QrmiRuntimeBlock` named by
`sampling.runtime_block_name` and sends the LUCJ circuit untranspiled in one
`sampler()` call. The ffsim pre-init passes run in the workflow process (the
runtime's `prepare` hook), and the remaining ffsim gates are expanded for QPY. The
SQD compilation hook `sqd_common.transpile:transpile_for_target` then compiles the
circuit against the actual Target: the preset stages plus the SABRE layout search of
`parameters.circuit`, with a fixed transpiler seed. The flow sets these hooks and
their configuration, so the block's `transpiler` and `transpile_config` are not used.
The `meas`/`test` registers and reset postselection are preserved. Where the hook
runs depends on the block's scheduler:

- `scheduler = "iqp"`: in the workflow process. Install QRMI there
  (`pip install "qrmi[ibm]"`, Linux only). The process must reach IBM Quantum.
- `scheduler = "slurm"`: on the compute node, inside a Slurm job that holds the QPU.
  The Python that runs the QRMI worker needs QRMI and its `task_runner`,
  `qcsc-prefect-qiskit` (the worker), and this package from the same revision. The
  hook imports only Qiskit and qiskit-ibm-runtime, so ffsim, Prefect-Qiskit and the
  solvers are not needed there:

  ```bash
  uv pip install -e packages/qcsc-prefect-qiskit
  uv pip install --no-deps -e algorithms/open_loop/sqd
  ```

  The QPU is held while the node compiles, so give the QRMI execution profile enough
  CPU threads for the layout search.

`sqd-create-blocks` saves the runtime block (name `runtime_block_name`, default
`qrmi-runtime`) from one of two config tables:

- `[qrmi]` with `resource` and `python_executable` (or `--qrmi-resource` and
  `--qrmi-python-executable`; fulqrum TOML only): creates the command, execution
  profile and HPC profile blocks of the QRMI Slurm job, separate from the solver's
  because they request the QPU (`slurm_qpu`), plus a `scheduler = "slurm"` runtime
  block that names them. See the commented table in
  `sqd_fulqrum_blocks.slurm.example.toml` and the
  [QRMI Sampler guide](../../../docs/howto/howto_qrmi_slurm_sampler.md) for Batch
  mode and shared storage.
- `[runtime]`: any `QrmiRuntimeBlock` fields. For IQP, `[runtime.qrmi]` holds QRMI's
  settings without the `<resource>_` prefix, and `QRMI_IBM_QCS_IAM_APIKEY` is read
  from the environment; it must not be written in the file. The block stores the API
  key as a secret.

Without either table, an existing block is kept.

## 🔀 Migrating from 1.x

| 1.x (`sqd_dice`) | 2.0 (`sqd`) |
| --- | --- |
| Flows `sqd_2405_05068_fulqrum`, `sqd-2405-05068`; deployments `sqd_2405_05068_fulqrum`, `sqd_2405_05068` | One flow and deployment, `sqd` |
| `solver_block_name: "sqd-fulqrum-solver"` | `solver_block_ref: "fulqrum_solver_job/sqd-fulqrum-solver"` (or `dice_shci_solver_job/…`, `sbd_solver_job/…`) |
| `sampling.qrmi_runtime_config: {…}` | Save a `QrmiRuntimeBlock` with `sqd-create-blocks` (`[qrmi]` or `[runtime]`) and set `sampling.runtime_block_name` |
| IBM `QuantumRuntime` block `ibm-runner` samples by default | The `qrmi-runtime` block samples; `ibm-runner` only reloads a job by `qiskit_runtime_job_id` |
| `sampling.sampler_options` | Removed; QRMI takes `sampling.shots` only |
| `sqd-fulqrum-deploy`, `sqd-deploy` | `sqd-deploy` |
| `sqd-fulqrum-create-blocks`, `sqd-create-blocks` (DICE) | `sqd-create-blocks` (fulqrum), `sqd-create-blocks --solver dice` |
| `pip install "sqd_dice[fulqrum,qrmi]"` | `pip install sqd` (fulqrum and QRMI included); `sqd[dice]`, `sqd[sbd]`, `sqd[all]` |
| `default_params/fulqrum_n2*.json`, `dice_n2.json` | `default_params/n2*.json` |

Old parameter files fail validation with a message that names the replacement.
The resume key of a checkpoint now includes the solver type, so a checkpoint
written before 2.0 is not resumed: a Retry moves it aside and starts at round 1,
and `resume_checkpoint` refuses it.

## Tests

Run the tests in `algorithms/open_loop/sqd/tests` from the repository root:

```bash
uv run pytest algorithms/open_loop/sqd/tests
```

Tests skip when their dependencies are missing. The flow tests need `ffsim`,
`prefect-qiskit` and `qcsc_workflow_utility`; the fulqrum comparison needs a fulqrum
build. `test_sqd_flow.py` runs the flow with each solver stand-in.
`test_sqd_common_flow.py` checks that each stage receives the exact object the
previous stage returned, which is what the Prefect graph arrows depend on.
`test_qrmi_runtime.py` covers the QRMI hooks, the QPY handoff and reloading a QRMI
job directory offline. `test_sqd_fulqrum_slurm_integration.py` and
`test_sqd_fulqrum_qrmi_slurm_integration.py` need `SLURM_INTEGRATION=1` and
`sbatch`; the QRMI one replaces QRMI and `task_runner` with fakes and requests no
QPU. No test contacts a Prefect server.

## 🧭 MPI Parameter Tuning

The [perf](./perf/) module also includes example code for implementing a meta workflow.
This workflow internally runs SQD workflows with different flow parameters, and measures the weak and strong scaling of the DICE solver.
