# Sample-based Krylov Quantum Diagonalization

This package provides implementation for the QCSC algorithm demonstrated in the paper  [Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization](https://arxiv.org/abs/2501.09702).

## 🚀 Getting Started

Before starting, make sure:

- For `quantum_source = "real-device"`, you have IBM Quantum credentials for QRMI. See the
  [QRMI Sampler guide](../../../docs/howto/howto_qrmi_slurm_sampler.md).
- You have completed [How to Set Up the MDX Workflow Server for QCSC Execution](../../../docs/howto/howto_setup_mdx_server.md).
- You have built the DICE executable under `packages/qcsc-prefect-dice/native` and know its absolute path.

From the repository root, install the local qcsc-prefect packages and the SKQD workflow:

```bash
cd /path/to/qcsc-prefect

uv pip install --no-deps \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-executor \
  -e packages/qcsc-prefect-dice \
  -e packages/qcsc-prefect-qiskit

uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/open_loop/skqd
```

After installation, you can deploy the workflow in a Prefect server:

```bash
skqd-deploy
```

In the Prefect UI, the flow parameters now include:

- `quantum_source`: choose `real-device` or `random`
- `random_seed`: base seed used when `quantum_source = "random"`
- `runner_name`: defaults to the `QrmiRuntimeBlock` `qrmi-runtime`

Set `quantum_source = "random"` when you want to skip quantum sampling and use deterministic pseudo-random bitstrings instead.

Before running on Miyabi or Fugaku, create the DICE-related blocks, the
`sampler_options` variable and, for `quantum_source = "real-device"`, the
`QrmiRuntimeBlock`. Start by copying the example config:

```bash
cp algorithms/open_loop/skqd/skqd_blocks.example.toml algorithms/open_loop/skqd/skqd_blocks.toml
```

The `[runtime]` and `[runtime.qrmi]` tables at the end of the file describe the
`QrmiRuntimeBlock`:

- For `quantum_source = "random"` only, delete both tables. No QRMI settings or API key
  are needed, and an existing runtime block is left unchanged.
- For `quantum_source = "real-device"` with `scheduler = "iqp"`, set `resource` and the
  `[runtime.qrmi]` values, and put the API key in the environment. It must not be
  written in the file. See [Quantum sampling](#quantum-sampling) for the details and for
  `scheduler = "slurm"`.

Then create the blocks:

```bash
export QRMI_IBM_QCS_IAM_APIKEY="<your API key>"  # only with a [runtime] table for IQP
python algorithms/open_loop/skqd/create_blocks.py --config algorithms/open_loop/skqd/skqd_blocks.toml --hpc-target miyabi
```

For Fugaku, start from `algorithms/open_loop/skqd/skqd_blocks.fugaku.example.toml` and pass
`--hpc-target fugaku` instead. Its `[runtime]` tables are commented out, so it works
for random mode as copied; uncomment them for `quantum_source = "real-device"`.

For a Fugaku build, use `packages/qcsc-prefect-dice/native/build_dice_fugaku.sh`
as the starting point and make sure the same runtime libraries are reflected in
`fugaku_spack_modules`, `modules`, or `environments` in the TOML when needed.

Set `dice_executable` in the TOML file to the built binary, for example:

```toml
dice_executable = "/path/to/qcsc-prefect/packages/qcsc-prefect-dice/native/bin/Dice"
```

On Fugaku, `work_dir` and `dice_executable` should point to a shared path that
compute nodes can read, for example a `/vol...` location.

## Quantum sampling

With `quantum_source = "real-device"`, the flow loads the `QrmiRuntimeBlock` named by
`runner_name` and sends all `krylov_dim` Trotter circuits in one `sampler()` call. Result
`i` belongs to circuit `i`.

The circuits are sent before transpilation. The runtime transpiles each one against the
actual Target with the SKQD hook `skqd_dice.transpilation:transpile`, using the
optimization level and SABRE settings in the `circuit` parameters. The flow sets this
hook and its configuration, so the block's `transpiler` and `transpile_config` are not
used. Where the hook runs depends on the block's scheduler:

- `scheduler = "iqp"`: in the workflow process. Install QRMI there
  (`pip install "qrmi[ibm]"`, Linux only). The process must reach IBM Quantum.
- `scheduler = "slurm"`: on the compute node. Install this package (`algorithms/open_loop/skqd`)
  in the compute-node Python as well, so that the hook can be imported.

`create_blocks.py` creates the block from the `[runtime]` table of the TOML file, as
shown in [Getting Started](#-getting-started). Other keys of the table are
`QrmiRuntimeBlock` fields. For IQP, `[runtime.qrmi]` holds QRMI's settings without the
`<resource>_` prefix, and `QRMI_IBM_QCS_IAM_APIKEY` is read from the environment. The
block stores the API key as a secret. For Slurm, set `scheduler = "slurm"`, `root_dir`
and the three block names described in the
[QRMI Sampler guide](../../../docs/howto/howto_qrmi_slurm_sampler.md), and leave out
`[runtime.qrmi]`; no API key is read.

## Solver selection

The flow uses DICE by default (`DiceSHCISolverJob` named by `solver_name`). To use SBD,
set `solver_block_ref` to `sbd_solver_job/<name>`; `dice_shci_solver_job/<name>` selects
DICE explicitly. The slug selects the block class, and an unknown slug is rejected.

DICE and SBD take the same arguments but do not solve the same problem. In this SKQD flow, SBD:

- solves closed-shell systems only, and uses one determinant set for alpha and beta;
- needs the Hartree-Fock determinant first;
- has no total spin target, so `spin_sq` must be `null` in the molecule settings.

The SBD solver itself also supports restricted open-shell systems, but this flow uses
the closed-shell subspace conversion.

Input that SBD cannot solve fails with `ValueError` before sampling. For SBD, the flow
puts the Hartree-Fock determinant first and keeps the subspace size limit: one slot for
Hartree-Fock and `int(sqrt(subspace_dim)) - 1` others. The subspace therefore always
includes Hartree-Fock and is not the same as the one DICE gets for the same settings.
DICE settings, including the default `spin_sq = 0.0`, are unchanged.

[`default_params/n2_sbd.json`](./default_params/n2_sbd.json) is an N2 example for SBD with
`spin_sq = null` and random sampling.
