# Sample-based Quantum Diagonalization

This package (`sqd` 2.0) implements the QCSC algorithm demonstrated in the paper [Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer](https://arxiv.org/abs/2405.05068).

It provides one Prefect flow, `sqd` (deploy with `sqd-deploy`). The solver that
diagonalizes each subspace is chosen once, with `solver` in `sqd_blocks.toml`:
`sqd-create-blocks` saves that solver's blocks and records its solver block in the
Advanced SQD Settings block (`sqd-advanced-settings`), which every run reads. The run
form stays the same for every solver.

| `solver` | Solver block (default name) | Install |
| --- | --- | --- |
| `fulqrum` (default) | `FulqrumSolverJob` (`sqd-fulqrum-solver`, `qcsc-prefect-fulqrum`) | `sqd` |
| `dice` | `DiceSHCISolverJob` (`sqd-dice-solver`, `qcsc-prefect-dice`) | `sqd[dice]` |
| `sbd` | `SBDSolverJob` (`sqd-sbd-solver`, `qcsc-prefect-sbd`) | `sqd[sbd]` |

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
  restarts CCSD Restarts times (Advanced SQD Settings, default 10) from its previous
  amplitudes.
- **Optimize** (`get_lucj_params`): LUCJ parameters from the CCSD t2 amplitudes,
  restricted to the interactions the QPU implements without SWAP gates. With
  `circuit.qpu_topology` `heavy-hex` (IBM Heron, the default), alpha-beta pairs sit
  on every 4th orbital, at most 5; with `square` (IBM Nighthawk), on every orbital.
  `compress` (on by default, `compress_maxiter` 50) optimizes the truncated
  factorization.
- **Execute** (`quantum_sampling`): builds the LUCJ circuit and samples it through the
  QRMI runtime block, which compiles it for the backend Target (see
  [Quantum sampling](#quantum-sampling)). `load_qrmi_result` and `load_runtime_job`
  take this place when a run reuses the samples of an earlier job.
- **Post-process** (`sqd_loop`): self-consistent configuration recovery. Every round
  diagonalizes `num_batches` subspaces concurrently, one solver job each, and then
  saves a checkpoint, so the recovery can continue after the last completed round.
  The heaviest `carryover_fraction` of each spin's CI strings in the round's best
  ground state is kept in every subspace of the next round, within `subspace_dim`.

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
| `molecule` | `{atom, basis, symmetry}` or `{fcidump_file}`, plus `spin_sq` (number or `null`, default `null`) |
| `circuit` | `n_lucj_layers` (1), `compress` (on), `compress_maxiter` (50), `qpu_topology` (`heavy-hex`) |
| `sampling` | `quantum_source`, `shots` (100000), `runtime_block_name`, `random_seed`, `qrmi_result_dir`, `qiskit_runtime_job_id`, `runner_name` (reload only) |
| `sqd` | `subspace_dim` (1000000), `num_batches` (4), `max_iterations` (5), `carryover_fraction` (0.5), `checkpoint` (on), `resume_checkpoint` |
| `advanced_block_name` | Advanced SQD Settings block, default `sqd-advanced-settings` |
| `solver_block_ref` | Solver Block Override, empty by default: `<block type slug>/<block name>` to run with another solver block than the advanced block names |

Only `molecule` must be set. The form still marks the three `sqd` fields as required,
so that they stand out. Set `circuit.qpu_topology` to `square` for an IBM Nighthawk QPU.

The rarely changed settings are in the **Advanced SQD Settings** block, which you can
edit in Prefect Cloud; `sqd-create-blocks` saves it from `[sqd_advanced]`. Without the
block, a run uses its defaults:

| Field | Default | Used by |
| --- | --- | --- |
| Solver Block | the solver block `sqd-create-blocks` created, else `fulqrum_solver_job/sqd-fulqrum-solver` | Post-process |
| CCSD Restarts | 10 | Map |
| Reset Mitigation | off | Execute, and reloads of earlier samples |
| Optimization Level, SABRE Layout Trials, SABRE SWAP Trials, SABRE Max Iteration | 3, 10000, 10, 8 | Execute (the transpiler) |

Each subspace has at most `subspace_dim` determinants. A closed-shell batch takes
`int(sqrt(subspace_dim))` strings for both spins. An open-shell batch splits
`subspace_dim` between the spins in proportion to their full CI string counts,
`C(norb, n_alpha)` and `C(norb, n_beta)`. A spin cannot take more strings than were
sampled for it, and the other spin then takes the rest of `subspace_dim`. When all
sampled strings fit, a round solves one subspace that holds them all. Carryover
strings, sampled strings, and repeats among them count once.

`sqd.carryover_fraction` sets how many CI strings carry over: of the best ground
state of a round, each spin's `int(fraction × n)` strings with the largest marginal
weight are kept in every subspace of the next round. They count toward
`subspace_dim`, and the rest of a subspace is sampled as before. `0` turns carryover
off. fulqrum and SBD select the strings on the compute node; SBD returns one list for
both spins of a closed-shell run and one per spin of an open-shell run. For DICE they
come from its ground state, so keep the DICE block's `return_sci_state` on. The
fraction is a flow parameter only; the flow sets it on the solver block of each run
(the SBD block's `carryover_ratio` too). The carryover is saved in the checkpoint, and
a run can resume with another fraction.

`default_params/` holds complete examples for N2 (6-31G, 1s cores frozen, 16 orbitals,
32 qubits), whose integrals are in `data/n2_631g_fc.fcidump`
(`data/make_n2_fcidump.py` regenerates it). A relative `fcidump_file` is resolved
from the working directory, then from the repository root.

| File | Solver | Sampling |
| --- | --- | --- |
| `n2_random.json` | fulqrum | Random bitstrings, no QPU (a first check of the blocks) |
| `n2.json` | fulqrum | The `qrmi-runtime` block |
| `n2_reload.json` | fulqrum | Samples of an earlier IBM Quantum job |
| `n2_dice.json` | DICE (with the DICE blocks created) | The `qrmi-runtime` block, with `spin_sq = 0.0` as the spin penalty |
| `n2_sbd.json` | SBD | Random bitstrings, `spin_sq = null` |

### Solver selection

The slug of the Solver Block (`<block type slug>/<block name>`) selects the block class
and an unknown slug is rejected; Prefect's hyphenated spelling (`sbd-solver-job`) is
accepted too. DICE and
SBD are imported only when a run selects them; without their extra the run fails
with the `pip install` command to run.

- **fulqrum** has no spin penalty: a nonzero `spin_sq` is ignored with a warning.
- **DICE** uses `spin_sq` as a spin penalty.
- **SBD** has no spin target, so `spin_sq` must be `null`; input SBD cannot solve
  fails with `ValueError` before sampling. It solves closed-shell and open-shell
  (ROHF) systems and needs the Hartree-Fock string of each spin first, which the flow
  adds:
  - Closed shell: one string list for both spins, Hartree-Fock first and
    `int(sqrt(subspace_dim)) - 1` other strings.
  - Open shell: an alpha and a beta list, each with its own Hartree-Fock string
    first. The subsampled lists are kept when their product, with Hartree-Fock,
    fits in `subspace_dim`; otherwise both are cut by the same factor.

  The carryover strings stay in the cut lists first, so SBD gets other subspaces
  than fulqrum or DICE for the same settings.

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
  33/200 from the stored occupancies, carryover strings and best energy.

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
  `sqd` (for example `subspace_dim`) or the solver without QPU time. The reload
  uses Reset Mitigation of the Advanced SQD Settings block, so keep it as it was in
  the sampling run.

## 🚀 Getting Started

Before starting, make sure:

- On a Slurm cluster, you have followed
  [Set up qcsc-prefect on a Slurm cluster](https://quantum.cloud.ibm.com/docs/en/guides/qcsc-prefect-setup): the checkout, both Python
  environments, the Prefect login, `qcsc-qrmi login`, and a shared work directory.
- For `quantum_source = "real-device"`, you have IBM Quantum credentials for QRMI. See
  the [QRMI Sampler guide](../../../docs/howto/howto_qrmi_slurm_sampler.md).
- You have built the solver for your site (see [Python environments](#python-environments)).

### Python environments

The workflow uses two Python environments, and the solvers run outside both:

| Environment | Where | Holds |
| --- | --- | --- |
| `~/venv/prefect` | workflow host | Prefect, the qcsc-prefect packages and this workflow (see [Install](#install)) |
| `~/venv/sqd-node` | compute nodes, for `scheduler = "slurm"` | the QRMI worker: QRMI with `task_runner`, `qcsc-prefect-qiskit` and `sqd_common` |

The [setup guide](https://quantum.cloud.ibm.com/docs/en/guides/qcsc-prefect-setup) creates both. To create the workflow host environment
yourself, run `uv venv ~/venv/prefect -p 3.12` and install as in [Install](#install). For
QRMI sampling in Slurm, create the compute-node environment on a path the compute nodes
see, from the repository root:

```bash
uv venv ~/venv/sqd-node -p 3.12
source ~/venv/sqd-node/bin/activate
uv pip install "qrmi[ibm]"                            # QRMI and task_runner
uv pip install -e packages/qcsc-prefect-qiskit
uv pip install --no-deps -e algorithms/open_loop/sqd
```

Its `bin/python` is the top-level `python_executable` of `sqd_blocks.toml`
(`--python-executable` on the command line).

Each solver is built on its own, and its block points to the built executable:

| Solver | Build | Executable in `sqd_blocks.toml` |
| --- | --- | --- |
| fulqrum | `bash packages/qcsc-prefect-fulqrum/native/build_fulqrum.sh`, into its own venv `native/fulqrum` | `[fulqrum] fulqrum_executable`: `/path/to/qcsc-prefect/packages/qcsc-prefect-fulqrum/native/fulqrum/bin/qcsc-fulqrum-solve` |
| SBD | `packages/qcsc-prefect-sbd/native/build_sbd_slurm.sh` (or `build_sbd.sh`, `build_sbd_gpu.sh`, `build_sbd_fugaku.sh`) | `[sbd] sbd_executable`: `/path/to/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag` |
| DICE | the `build_dice*.sh` scripts in `packages/qcsc-prefect-dice/native` | `[dice] dice_executable`: `/path/to/qcsc-prefect/packages/qcsc-prefect-dice/native/bin/Dice` |

### Install

The package is installed from a source checkout. On the submit host (where the
Prefect process runs and the scheduler commands work), in `~/venv/prefect`, from the
repository root:

```bash
uv pip install --no-deps \
  -e packages/qcsc-prefect-core -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks -e packages/qcsc-prefect-executor \
  -e 'packages/qcsc-prefect-qiskit[slurm]' -e packages/qcsc-prefect-fulqrum
uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/open_loop/sqd          # fulqrum
```

For SBD or DICE, also install their packages and the extra, for example
`uv pip install --no-deps -e packages/qcsc-prefect-sbd` and
`uv pip install -e "algorithms/open_loop/sqd[sbd]"` (`[dice]`, or `[all]` for both).

### SQD with the fulqrum eigensolver

Each batch is one scheduler job running `qcsc-fulqrum-solve` (one OpenMP process).
The default target is a generic Slurm cluster; Miyabi and Fugaku also work, given a
fulqrum build on that system.

1. **Compute nodes:** run `bash packages/qcsc-prefect-fulqrum/native/build_fulqrum.sh`.
   It builds fulqrum into a venv (default `native/fulqrum`), installs
   `qcsc-prefect-fulqrum` there and prints the path of its `qcsc-fulqrum-solve`. The
   compute environment needs no Prefect.
2. **Submit host:** install the workflow as above.
3. Log in to Prefect Cloud on the submit host (`prefect cloud login`, or set
   `PREFECT_API_URL` and `PREFECT_API_KEY`).
4. Create the blocks (see [Block configuration](#block-configuration)):

   ```bash
   cp algorithms/open_loop/sqd/sqd_blocks.example.toml algorithms/open_loop/sqd/sqd_blocks.toml
   # edit queue, work_dir, python_executable, fulqrum_executable and the [qrmi] resource
   sqd-create-blocks --config algorithms/open_loop/sqd/sqd_blocks.toml --dry-run
   sqd-create-blocks --config algorithms/open_loop/sqd/sqd_blocks.toml
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

### SQD with the SBD eigensolver

Each batch is one MPI job of the SBD `diag` executable.

1. **Compute nodes:** build `diag` with the script for your system under
   `packages/qcsc-prefect-sbd/native/`: `build_sbd_slurm.sh` (Slurm, MPI and
   OpenBLAS), `build_sbd.sh` (Miyabi-C), `build_sbd_gpu.sh` (`diag-gpu`, Miyabi-G) or
   `build_sbd_fugaku.sh`. The CPU scripts write
   `packages/qcsc-prefect-sbd/native/diag`, the default `sbd_executable` of the
   example, and `build_sbd_gpu.sh` writes `diag-gpu` beside it.
2. **Submit host:** install `qcsc-prefect-sbd` and `sqd[sbd]` (see [Install](#install)).
3. Set `solver = "sbd"` and `[sbd] sbd_executable` in `sqd_blocks.toml`, then create
   the blocks:

   ```bash
   sqd-create-blocks --config algorithms/open_loop/sqd/sqd_blocks.toml --dry-run
   sqd-create-blocks --config algorithms/open_loop/sqd/sqd_blocks.toml
   ```

   `--solver sbd` selects SBD for a file with another `solver`. The MPI launcher,
   ranks and threads come from the profile's `[sbd]` table; the `[sbd]` table of your
   file sets the Davidson tuning (`block`, `iteration`, `tolerance`, `carryover_type`,
   `solver_mode` and the three comm sizes). The product of `task_comm_size`, `adet_comm_size`
   and `bdet_comm_size` must divide the MPI ranks, and `solver_mode = "gpu"` needs a
   GPU profile (`resource_class = "gpu"`); `--dry-run` checks both.

This saves `sqd-sbd-solver` with `cmd-sqd-sbd-solver`, `exec-sqd-sbd-<system>` and
`hpc-<system>-sqd-sbd`, and records `sbd_solver_job/sqd-sbd-solver` as the Solver
Block of the Advanced SQD Settings block. Runs need `spin_sq = null`, as in
`default_params/n2_sbd.json`. The flow puts the Hartree-Fock string of each spin
first, so closed-shell and open-shell (ROHF) molecules both work.

### SQD with the DICE eigensolver

Create the DICE blocks for Miyabi or Fugaku from the same example. Set
`solver = "dice"`, `profile = "miyabi"` (or `"fugaku"`), `project` (your group), and
`dice_executable` in the `[dice]` table:

```bash
cp algorithms/open_loop/sqd/sqd_blocks.example.toml algorithms/open_loop/sqd/sqd_blocks.toml
export QRMI_IBM_QCS_IAM_APIKEY="<your API key>"  # only with a [runtime] table for IQP
sqd-create-blocks --config algorithms/open_loop/sqd/sqd_blocks.toml
```

The `miyabi` and `fugaku` profiles set the DICE MPI launcher and ranks per node, and
`mpi_options` defaults to `-np` (Miyabi) or `-n` (Fugaku) with `num_nodes x mpiprocs`. The
`[qrmi]` table is for a Slurm cluster, so on these systems replace it with `[runtime]`
(IQP from the workflow host), or remove both for random mode. For a Fugaku build, use
`packages/qcsc-prefect-dice/native/build_dice_fugaku.sh` as the starting point and
reflect the same runtime libraries in `fugaku_spack_modules`, `modules` or
`environments`.

On Fugaku, `work_dir` and `dice_executable` should point to a shared path that
compute nodes can read, for example a `/vol...` location.

This saves `sqd-dice-solver` and its job blocks, and records
`dice_shci_solver_job/sqd-dice-solver` as the Solver Block of the Advanced SQD
Settings block. Then start a run of the `sqd` deployment with
`default_params/n2_dice.json`; nothing in the run form names the solver.

### Block configuration

`sqd_blocks.example.toml` is laid out by topic:

| Section | Keys |
| --- | --- |
| 1. Solver and system | `python_executable` (the QRMI worker's compute-node Python), `profile`, `solver` |
| 2. Cluster | `queue`, `project`, `work_dir` |
| 3. Solver | `[fulqrum]` (`fulqrum_executable`, `eigensolver`, the eigensolver tuning), `[sbd]` (`sbd_executable`, the SBD tuning), `[dice]` (`dice_executable`, the DICE tuning); only the selected solver's table is read |
| 4. Quantum sampling | `[qrmi]` (optional `resource`, `instance`, `queue`) or `[runtime]` |
| 5. Advanced SQD settings | `[sqd_advanced]`: the Advanced SQD Settings block |
| 6. Block overrides | `[advanced]`: block names, `runtime_block_name`, `advanced_block_name`, `timeout_seconds` |

The system profile, `sqd_common/profiles/<profile>.toml` or your own file given by path,
holds the scheduler settings: `hpc_target`, `resource_class`, `launcher`, `num_nodes`,
`mpiprocs`, `ompthreads`, `gpus_per_node`, `walltime`, `modules`, `pre_commands`,
`environments`, the fulqrum `backend`, the `fugaku_*` keys, the DICE and SBD MPI
settings (`[dice]`, `[sbd]`) and the QRMI job's `walltime`, `ompthreads` and
`timeout_seconds` (`[qrmi]`). The SBD blocks read only the keys they use, so the
fulqrum `backend` and tuning of a profile do not reach them.

Block names follow the solver and system:

| Block | Default name |
| --- | --- |
| Solver, command | `sqd-<solver>-solver`, `cmd-sqd-<solver>-solver` |
| Execution profile, HPC profile | `exec-sqd-<solver>-<system>`, `hpc-<system>-sqd-<solver>` |
| QRMI job (any solver) | `cmd-sqd-qrmi`, `exec-sqd-qrmi`, `hpc-slurm-sqd-qrmi` |
| QRMI runtime, advanced settings | `qrmi-runtime`, `sqd-advanced-settings` |

Later layers override earlier ones: the profile, then `[advanced]`, then the top
level of your file, then its `[<solver>]` table. Any profile key can be set in your
file to override it, and any setting can also be given as a CLI flag. A config
without `profile` or `solver` is read as one flat table.

The bundled profiles are `slurm`, `miyabi` and `fugaku` (CPU), plus two for the GPU
builds of fulqrum (`backend = "gpu-mpi"`) and SBD (`solver_mode = "gpu"`, `diag-gpu`),
which run one MPI rank per GPU:

- `slurm-gpu`: `srun --mpi=pmix`, 4 GPUs and 4 ranks per node
  (`#SBATCH --gpus-per-node`). Set `queue` to a GPU partition. The QRMI sampler job
  uses the same queue unless your `[qrmi]` table sets its own.
- `miyabi-gpu`: Miyabi-G, `mpirun`, 1 GPU and 1 rank per node, 72 threads. The job
  goes to the HPC profile's GPU queue. For SBD it also unsets
  `OMPI_MCA_mca_base_env_list`, sets `MIYABI=G` and loads no modules.

On the CPU profiles SBD runs 4 MPI ranks with one thread each (`srun --mpi=pmix` on
Slurm, Intel MPI `mpiexec.hydra` on Miyabi), or 2 ranks with `mpiexec` on Fugaku.
On Miyabi and Fugaku, `-n` or `-np` in `mpi_options` must equal `num_nodes x mpiprocs`,
and on Fugaku `max-proc-per-node` in `fugaku_mpi_options_for_pjm` must equal `mpiprocs`;
`sqd-create-blocks` stops otherwise. To choose the nodes, ranks, threads and solver grid
for a subspace size, see
[Size the SQD diagonalization for your HPC resources](../../../docs/tutorials/configure_sqd_solver_resources.md).

`num_nodes` adds nodes, and with them ranks and GPUs, to each solver job. See
`packages/qcsc-prefect-fulqrum/native/README.md` for the GPU build.

## Quantum sampling

With `quantum_source = "real-device"`, the flow loads the `QrmiRuntimeBlock` named by
`sampling.runtime_block_name` and sends the LUCJ circuit untranspiled in one
`sampler()` call. The ffsim pre-init passes run in the workflow process (the
runtime's `prepare` hook), and the remaining ffsim gates are expanded for QPY. The
SQD compilation hook `sqd_common.transpile:transpile_for_target` then compiles the
circuit against the actual Target: the preset stages plus the SABRE layout search of
the Advanced SQD Settings block, with a fixed transpiler seed. The flow sets these hooks and
their configuration, so the block's `transpiler` and `transpile_config` are not used.
The `meas`/`test` registers and reset postselection are preserved. Where the hook
runs depends on the block's scheduler:

- `scheduler = "iqp"`: in the workflow process. Install QRMI there
  (`pip install "qrmi[ibm]"`, Linux only). The process must reach IBM Quantum.
- `scheduler = "slurm"`: on the compute node, inside a Slurm job that holds the QPU.
  The Python that runs the QRMI worker needs QRMI and its `task_runner`,
  `qcsc-prefect-qiskit` (the worker, which imports Prefect), and this package from the
  same revision. The hook imports only Qiskit and qiskit-ibm-runtime, so ffsim,
  Prefect-Qiskit and the solvers are not needed there. Install them into that Python,
  `~/venv/sqd-node` in [Python environments](#python-environments), check the
  imports, and set its path as the top-level `python_executable`. If the QRMI worker
  environment already has `qcsc-prefect-qiskit`, only the last three commands are
  needed:

  ```bash
  uv pip install -e packages/qcsc-prefect-qiskit
  uv pip install --no-deps -e algorithms/open_loop/sqd
  python -c "import qrmi, qcsc_prefect.integrations.qiskit.qrmi.worker, sqd_common.transpile"
  which python task_runner
  ```

  The QPU is held while the node compiles, so give the QRMI execution profile enough
  CPU threads for the layout search.

### IBM Quantum access: `qcsc-qrmi login`

Run `qcsc-qrmi login` once, before `sqd-create-blocks`. It:

1. asks for the IBM Cloud API key,
2. finds the IBM Quantum instances the key can use and their backends,
3. writes `qrmi_config.json` (default `~/.qiskit/qrmi_config.json`, mode 600), with
   the key once, one instance list and one backend list, and
4. saves the `qrmi-runtime` block with an **IBM Quantum Instance** and a **QPU
   Backend** dropdown.

Choose the instance and the backend in that block (Prefect Cloud); the block is the
only place they are set. Its **Service CRN** field overrides the instance's CRN. The
API key is not in the block. For Slurm, install the written file as the
`qrmi_config.json` of the compute nodes, the path in `plugstack.conf` (on a test
cluster, pass that path as `--config`). The run checks, before it starts, that the
instance can use the backend; otherwise it fails at once and names the instances
that can. `qcsc-qrmi refresh` updates the lists, `qcsc-qrmi list` prints them, and
`qcsc-qrmi check` runs the check. See the
[QRMI Sampler guide](../../../docs/howto/howto_qrmi_slurm_sampler.md) for the file
layout.

`sqd-create-blocks` saves the rest of the runtime block (name `runtime_block_name`,
default `qrmi-runtime`) from one of two config tables, and keeps the instance and the
backend that `login` saved:

- `[qrmi]`, even empty, with the top-level `python_executable` (or
  `--python-executable`): creates the command, execution profile and HPC profile
  blocks of the QRMI Slurm job, separate from the solver's because the job holds the
  QPU, plus the `scheduler = "slurm"` settings of the runtime block. The job requests
  the block's backend as `--qpu`. This is the default: `sqd_blocks.example.toml` has
  the table active. `resource` and `instance` (a name) are optional and replace the
  block's choice; `service_crn`, `endpoint` and `iam_endpoint` are optional block
  fields. `[qrmi]` rejects `api_key`.
- `[runtime]`: any `QrmiRuntimeBlock` fields, such as `scheduler = "iqp"`. For IQP the
  workflow process reads the API key from its `qrmi_config.json` or the
  `QRMI_IBM_QCS_IAM_APIKEY` environment variable; it must not be written in the file.
  `[runtime.qrmi]` takes other QRMI settings without the `<resource>_` prefix.

A change of the backend or the instance is printed.
Without either table, an existing block is kept.

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
