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
- For a QRMI source Block, IBM Quantum credentials. See
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
uv pip install -e algorithms/closed_loop/sqd_de
```

## Create SBD Blocks (Miyabi/Fugaku)

Use the config file that matches your target:

```bash
cd /work/gz00/z12345/qcsc-prefect
```

Miyabi:

```bash
cp algorithms/closed_loop/sqd_de/sbd_blocks.example.toml algorithms/closed_loop/sqd_de/sbd_blocks.toml
vim algorithms/closed_loop/sqd_de/sbd_blocks.toml
```

Fugaku:

```bash
cp algorithms/closed_loop/sqd_de/sbd_blocks.fugaku.example.toml algorithms/closed_loop/sqd_de/sbd_blocks.toml
vim algorithms/closed_loop/sqd_de/sbd_blocks.toml
```

The `[runtime]` and `[runtime.qrmi]` tables at the end of the Miyabi example create the
`QrmiRuntimeBlock` referenced by the source Block. In the Fugaku and Slurm examples
they are commented out:

- For source kind `random` or `qfw` only, delete (or keep commented) both tables. No QRMI settings or
  API key are needed, and an existing runtime block is left unchanged.
- For source kind `qrmi` with `scheduler = "iqp"`, set `resource` and the `[runtime.qrmi]`
  values, and export the API key before running the generator. It must not be written in
  the file:

  ```bash
  export QRMI_IBM_QCS_IAM_APIKEY="<your API key>"
  ```

Run block generator:

Miyabi:

```bash
python algorithms/closed_loop/sqd_de/create_blocks.py --config algorithms/closed_loop/sqd_de/sbd_blocks.toml --hpc-target miyabi
```

Fugaku:

```bash
python algorithms/closed_loop/sqd_de/create_blocks.py --config algorithms/closed_loop/sqd_de/sbd_blocks.toml --hpc-target fugaku
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

For Fugaku, `algorithms/closed_loop/sqd_de/sbd_blocks.fugaku.example.toml` already matches the tutorial baseline
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
- `QuantumSourceBlock` `quantum-source`, when the config has a `[source]` table

## Run flow

Local execution from JSON parameters:

```bash
python algorithms/closed_loop/sqd_de/exec.py algorithms/closed_loop/sqd_de/default_params/test_n2.json
```

Note:
- The example JSON files use intentionally low `circ_params.sabre_layout_trials` values so Fugaku runs do not appear stuck in SABRE layout search before IBM Quantum submission.
- These are conservative starting points only. Tune them for your own backend, qubit count, and circuit specification.
- `"quantum_source"` is a saved source Block name. The examples use `"quantum-source"`. To use synthetic samples, set the Block's `config.kind` to `"random"`; `"random_seed"` controls reproducibility.

Deploy entrypoint:

```bash
# For Fugaku + Ray stability (recommended)
export PREFECT_RAY_NUM_CPUS=2

sbd-deploy
```

## Quantum sampling

`quantum_source` is the name of a saved
[`QuantumSourceBlock`](../../../docs/howto/howto_use_quantum_source.md), defaulting to
`quantum-source`. The library loads that Block inside each walker and constructs the
configured source. SQD passes its width, walker seed and compilation policy, then calls
`source.sample(circuit, shots=shots)`. It has no platform registry or platform-specific
block-loading branch.

The block generator saves a source Block from the `[source]` table, independently of
its solver and runtime configuration:

```toml
[source]
name = "quantum-source"

[source.config]
kind = "random"
```

To use QRMI, set `kind = "qrmi"`, `block_name = "qrmi-runtime"` and
`tags = ["res: quantum"]` in `[source.config]`. The referenced runtime selects IQP or
Slurm. The Miyabi example has this preset; Fugaku and Slurm start with `random`.

To use QFw, replace the source settings with the following. The sampler path must be
an absolute path in the workflow environment; the existing launcher passes it to the
node as before:

```toml
[source]
name = "quantum-source"

[source.config]
kind = "qfw"
block_name = "qfw-runner"
sampler = "/path/to/qcsc-prefect/algorithms/closed_loop/sqd_de/sbd/qfw_slurm_sqd_sampler.py"
sampler_arg_names = [
  ["optimization_level", "--optimization-level"],
  ["sabre_max_iterations", "--sabre-max-iterations"],
  ["sabre_swap_trials", "--sabre-swap-trials"],
  ["sabre_layout_trials", "--sabre-layout-trials"],
]
```

Save several source Blocks under different names and switch only `quantum_source`,
or edit the Block under the same name to change its source. No code change in SQD is
needed. Existing values `random`, `real-device` and `qfw` have no special meaning;
create source Blocks with those names if retaining those flow parameters. Random
also needs a source Block; it loads no runtime/resource Block and builds no circuit.
The width remains `2 * num_orbitals` and the seed remains `random_seed` plus a pairing
of trial and walker indices.

The QFw node script's move into the library remains PR2; it still lives at the path
above. See [the QFw-SLURM tutorial](../../../docs/tutorials/run_sbd_closed_loop_workflow_qfw-slurm.md).

For sources requiring a circuit, the walker applies reset mitigation itself: when
`circ_params.use_reset_mitigation` is set, it keeps the shots whose
`test` register is all zeros, and records the fraction kept as `shot_retention_rate`.
A result without a `test` register, or with no shot left, is an error. `random` has no
`test` register and records no `shot_retention_rate`.

With a QRMI source Block, each walker sends its LUCJ circuit untranspiled in one `sampler()`
call. The ffsim pre-init stage runs in the workflow process. The runtime then searches
the SABRE layout and transpiles against the actual Target with the hook
`sbd.transpilation:transpile`, using `optimization_level` and the SABRE settings in
`circ_params`. The flow sets these hooks and their configuration, so the block's
`transpiler` and `transpile_config` are not used. Where the hook runs depends on the
block's scheduler:

- `scheduler = "iqp"`: in the workflow process. Install QRMI there
  (`pip install "qrmi[ibm]"`, Linux only). The process must reach IBM Quantum.
- `scheduler = "slurm"`: on the compute node. Install this package (`algorithms/closed_loop/sqd_de`)
  in the compute-node Python as well, so that the hook can be imported. Set `root_dir`
  and the three block names described in the
  [QRMI Sampler guide](../../../docs/howto/howto_qrmi_slurm_sampler.md) in `[runtime]`, and
  leave out `[runtime.qrmi]`.

The `sqd_options` variable sets the shots of every walker, with the same rules for
every `quantum_source`. Each walker reads it before loading any block:

| `sqd_options` | Result |
| --- | --- |
| not saved, `{}` or `{"params": {}}` | 100,000 shots |
| `{"params": {"shots": <positive integer>}}` | that many shots |
| `shots` is `null`, a float, `0`, negative, a boolean or a string | `ValueError` |
| the value, or `params`, is not a mapping (also a saved `null`) | `TypeError` |
| any other option, for example `twirling` or `params.seed` | `ValueError` |

The walker reads the Variable named `sqd_options`. `create_blocks.py` can save it
under another name (`options_variable_name`), but the walker does not read that name:
with another name, the walker samples the default 100,000 shots.

### Behavior changes

Sampling through quantum sources changes these cases. Check saved `sqd_options` values
and QFw results before upgrading:

- `quantum_source` now names a `QuantumSourceBlock`, defaulting to `quantum-source`.
  Save that Block, or save Blocks named `random`, `real-device` and `qfw` to keep
  existing explicit parameter values. Random also loads the source configuration
  Block, though it does not load a runtime/resource Block.
- `random` and `qfw` now reject options other than `params.shots`, as `real-device`
  already did. Before, they ignored them.
- Invalid shots are rejected in every mode. Before, `random` and `qfw` truncated floats
  (`4.9` became `4`) and replaced `0` and `null` with 100,000.
- A saved `{}` or `{"params": {}}` gives 100,000 shots in every mode. Before,
  `real-device` sampled 100 shots then; a Variable that was not saved already gave
  100,000.
- A saved `null` Variable raises `TypeError`, and the Variable is checked before the
  solver block is loaded.
- `qfw` keeps leading zeros. Before, when every kept bitstring started with `0`, the
  bitstrings were shorter than `2 * num_orbitals`.
- `qfw` rejects counts that do not add up to the shots. Its `shot_retention_rate` is the
  fraction of the shots returned, as for `real-device`.
- When reset mitigation keeps no shot, `real-device` and `qfw` both raise
  `RuntimeError("Reset mitigation removed all samples.")`.
- `qfw` reports a missing QFw-SLURM launcher when it samples, not when it loads the
  `QFwResource` block, which no longer imports QFw.

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
