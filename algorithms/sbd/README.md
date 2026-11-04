# SBD Closed-Loop Workflow on qcsc-prefect (Miyabi/Fugaku)

This workflow performs SQD closed-loop optimization and runs SBD diagonalization
jobs on Miyabi or Fugaku through the current `qcsc-prefect` architecture.

The old `prefect-sbd` / `prefect-miyabi` runtime path is replaced by:

- `qcsc_prefect_blocks` (`CommandBlock`, `ExecutionProfileBlock`, `HPCProfileBlock`)
- `qcsc_prefect_executor.run_job_from_blocks(...)`
- `SBD Solver Job` block (`qcsc_prefect_sbd.SBDSolverJob`, from `packages/qcsc-prefect-sbd`)

Compatibility notes from old `prefect_sbd.sbd_job`:

- keeps the same block slug: `sbd_solver_job`
- keeps the same solver arguments (`task_comm_size`, `adet_comm_size`, `bdet_comm_size`, `block`, `iteration`, `tolerance`, `carryover_ratio`, `solver_mode`)
- keeps the same expected output files (`davidson_energy.txt`, `occ_a.txt`, `occ_b.txt`, `carryover.bin`)

## Prerequisites

- You can submit HPC jobs on your target system (`qsub` / `qstat` for Miyabi, `pjsub` / `pjstat` for Fugaku).
- Prefect API is reachable (On-Prem or Cloud profile is active).
- For `quantum_source = "real-device"`, IBM Quantum credentials for QRMI. See
  [Quantum sampling](#quantum-sampling).
- SBD executable (`diag`) is already built on your target HPC system and you know its absolute path.

Build examples:

```bash
# Miyabi
cd /work/gz00/z12345/qcsc-prefect/packages/qcsc-prefect-sbd/native
bash ./build_sbd.sh
realpath ./diag

# Fugaku
cd /work/gz00/z12345/qcsc-prefect/packages/qcsc-prefect-sbd/native
bash ./build_sbd_fugaku.sh
realpath ./diag
```

## Install

From repository root:

```bash
cd /work/gz00/z12345/qcsc-prefect
source ~/venv/prefect/bin/activate

# qcsc-prefect local packages
uv pip install --no-deps \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-executor \
  -e packages/qcsc-prefect-dice \
  -e packages/qcsc-prefect-qiskit \
  -e packages/qcsc-prefect-sbd

# algorithm dependencies
uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/sbd
```

## Create SBD Blocks (Miyabi/Fugaku)

Use the config file that matches your target:

```bash
cd /work/gz00/z12345/qcsc-prefect
```

Miyabi:

```bash
cp algorithms/sbd/sbd_blocks.example.toml algorithms/sbd/sbd_blocks.toml
vim algorithms/sbd/sbd_blocks.toml
```

Fugaku:

```bash
cp algorithms/sbd/sbd_blocks.fugaku.example.toml algorithms/sbd/sbd_blocks.toml
vim algorithms/sbd/sbd_blocks.toml
```

The `[runtime]` and `[runtime.qrmi]` tables at the end of the Miyabi example create the
`QrmiRuntimeBlock` for `quantum_source = "real-device"`. In the Fugaku and Slurm examples
they are commented out:

- For `random` or `qfw` only, delete (or keep commented) both tables. No QRMI settings or
  API key are needed, and an existing runtime block is left unchanged.
- For `real-device` with `scheduler = "iqp"`, set `resource` and the `[runtime.qrmi]`
  values, and export the API key before running the generator. It must not be written in
  the file:

  ```bash
  export QRMI_IBM_QCS_IAM_APIKEY="<your API key>"
  ```

Run block generator:

Miyabi:

```bash
python algorithms/sbd/create_blocks.py --config algorithms/sbd/sbd_blocks.toml --hpc-target miyabi
```

Fugaku:

```bash
python algorithms/sbd/create_blocks.py --config algorithms/sbd/sbd_blocks.toml --hpc-target fugaku
```

`sbd_executable` should point to your built binary, for example:

```toml
sbd_executable = "/work/project/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag"
```

For memory stability on Miyabi, start from:

```toml
mpiprocs = 4
mpi_options = ["-np", "4"]
task_comm_size = 1
adet_comm_size = 1
bdet_comm_size = 1
block = 4
iteration = 1
tolerance = 0.01
carryover_ratio = 0.1
```

Then increase only if your target case runs without OOM.

For Fugaku, `algorithms/sbd/sbd_blocks.fugaku.example.toml` already matches the tutorial baseline
(`num_nodes = 2`, `mpiprocs = 2`, `mpi_options = ["-n", "2"]`, `script_filename = "sbd_solver.pjm"`),
so in most cases you only need to update `group`, `queue`, `work_dir`, and `sbd_executable`.
For backward compatibility, `project` is still accepted as a legacy alias in Fugaku mode.

This creates:

- `CommandBlock` (default: `cmd-sbd-diag`)
- `ExecutionProfileBlock` (default: `exec-sbd-mpi` for Miyabi, `exec-sbd-fugaku` for Fugaku)
- `HPCProfileBlock` (default: `hpc-miyabi-sbd` for Miyabi, `hpc-fugaku-sbd` for Fugaku)
- `SBD Solver Job` block (default: `davidson-solver`)
- Prefect Variable `sqd_options`
- `QrmiRuntimeBlock` `qrmi-runtime`, only when the config has a `[runtime]` table

## Run flow

Local execution from JSON parameters:

```bash
python algorithms/sbd/exec.py algorithms/sbd/default_params/test_n2.json
```

Note:
- The example JSON files use intentionally low `circ_params.sabre_layout_trials` values so Fugaku runs do not appear stuck in SABRE layout search before IBM Quantum submission.
- These are conservative starting points only. Tune them for your own backend, qubit count, and circuit specification.
- Set `"quantum_source": "random"` in the JSON if you want to skip quantum sampling and use deterministic pseudo-random bitstrings instead. `"random_seed"` controls reproducibility in that mode.

Deploy entrypoint:

```bash
# For Fugaku + Ray stability (recommended)
export PREFECT_RAY_NUM_CPUS=2

sbd-deploy
```

## Quantum sampling

`quantum_source` selects how each walker obtains bitstrings:

- `real-device`: sample through `QrmiRuntime`, using the `QrmiRuntimeBlock` named
  `qrmi-runtime`. The block's `scheduler` selects where QRMI runs (see below).
- `random`: deterministic pseudo-random bitstrings from `random_seed`.
- `qfw`: the QFw-SLURM path through the `QFwResource` block `qfw-runner`, unchanged. See
  [the QFw-SLURM tutorial](../../docs/tutorials/run_sbd_closed_loop_workflow_qfw-slurm.md).

With `real-device`, each walker sends its LUCJ circuit untranspiled in one `sampler()`
call. The ffsim pre-init stage runs in the workflow process. The runtime then searches
the SABRE layout and transpiles against the actual Target with the hook
`sbd.transpilation:transpile`, using `optimization_level` and the SABRE settings in
`circ_params`. The flow sets these hooks and their configuration, so the block's
`transpiler` and `transpile_config` are not used. Where the hook runs depends on the
block's scheduler:

- `scheduler = "iqp"`: in the workflow process. Install QRMI there
  (`pip install "qrmi[ibm]"`, Linux only). The process must reach IBM Quantum.
- `scheduler = "slurm"`: on the compute node. Install this package (`algorithms/sbd`)
  in the compute-node Python as well, so that the hook can be imported. Set `root_dir`
  and the three block names described in the
  [QRMI Sampler guide](../../docs/howto/howto_qrmi_slurm_sampler.md) in `[runtime]`, and
  leave out `[runtime.qrmi]`.

The `sqd_options` variable is passed to `QrmiRuntime` unchanged. On the `real-device`
path it may contain only `{"params": {"shots": <positive integer>}}`; other primitive
options (for example twirling) and non-integer shots are rejected before submission
instead of being ignored. The `random` and `qfw` paths read only the shots, as before.

Each walker searches the layout for its own circuit and submits its own job, so a run
submits (iterations x walkers) jobs. With `iqp`, each job waits in the IBM Quantum queue
by itself. The sampling task carries the Prefect tag `res: quantum`, which can be used
with a Prefect concurrency limit.

## Solver selection

`solver_block_ref` selects the eigensolver of every walker:

- `sbd_solver_job/<name>` (default `sbd_solver_job/davidson-solver`): SBD. Each solve
  returns carryover determinants, which are kept in the next generation's subspace.
- `dice_shci_solver_job/<name>`: DICE, **without carryover**. The walker returns an
  empty carryover, so each generation subsamples without it. This is a different
  variant from SBD with carryover; results are not expected to match.

The selection is logged at the start of the flow (`solver=..., carryover=...`) and is
recorded in every row of the `sqd-telemetry` artifact. In both cases the subspace starts
with the Hartree-Fock determinant and uses one set for alpha and beta.

## Optional integrations

- If `s3-sqd` block is configured (`prefect-aws` S3 block), intermediate arrays
  are saved to object storage.
- If not configured, data falls back to Prefect local storage.
