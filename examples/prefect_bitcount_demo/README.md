# Prefect BitCount Demo

This example provides two execution styles:

- `flow_optimized.py`: block-driven execution (works on Miyabi, Fugaku, and Slurm)
- `flow_tutorial_style.py`: legacy `counter.get(bitstrings)` style with optional HPC profile override

## Files

- `/Users/hitomi/Project/qcsc-prefect/examples/prefect_bitcount_demo/create_blocks.py`
- `/Users/hitomi/Project/qcsc-prefect/examples/prefect_bitcount_demo/bitcount_blocks.example.toml`
- `/Users/hitomi/Project/qcsc-prefect/examples/prefect_bitcount_demo/flow_optimized.py`
- `/Users/hitomi/Project/qcsc-prefect/examples/prefect_bitcount_demo/flow_tutorial_style.py`
- `/Users/hitomi/Project/qcsc-prefect/examples/prefect_bitcount_demo/quantum_sampling.py`
- `/Users/hitomi/Project/qcsc-prefect/examples/prefect_bitcount_demo/get_counts_integration.py`
- `/Users/hitomi/Project/qcsc-prefect/examples/prefect_bitcount_demo/build_on_miyabi.sh`
- `/Users/hitomi/Project/qcsc-prefect/examples/prefect_bitcount_demo/build_on_fugaku.sh`

For a local Slurm walkthrough using `slurm-docker-cluster`, see
[`docs/tutorials/create_qcsc_workflow_for_local_slurm.md`](../../docs/tutorials/create_qcsc_workflow_for_local_slurm.md).

## Build executable

Miyabi:

```bash
cd /Users/hitomi/Project/qcsc-prefect
./examples/prefect_bitcount_demo/build_on_miyabi.sh
```

Fugaku:

```bash
cd /Users/hitomi/Project/qcsc-prefect
./examples/prefect_bitcount_demo/build_on_fugaku.sh
```

## Create blocks

```bash
cd /Users/hitomi/Project/qcsc-prefect
cp examples/prefect_bitcount_demo/bitcount_blocks.example.toml \
   examples/prefect_bitcount_demo/bitcount_blocks.toml
```

Miyabi defaults:

```bash
python examples/prefect_bitcount_demo/create_blocks.py \
  --config examples/prefect_bitcount_demo/bitcount_blocks.toml
```

This default setup creates the optimized-flow assets only:

- `cmd-bitcount-hist`
- `exec-bitcount-mpi`
- `hpc-miyabi-bitcount`
- `miyabi-bitcount-options`

Fugaku:

```bash
python examples/prefect_bitcount_demo/create_blocks.py \
  --config examples/prefect_bitcount_demo/bitcount_blocks.toml \
  --hpc-target fugaku
```

Local Slurm:

The Slurm execution path is supported by `flow_optimized.py`, but the current
`create_blocks.py` helper does not yet generate Slurm assets automatically.
For now, create the Slurm blocks manually as described in
[`docs/tutorials/create_qcsc_workflow_for_local_slurm.md`](../../docs/tutorials/create_qcsc_workflow_for_local_slurm.md).

## Run optimized flow

Miyabi example:

```bash
python examples/prefect_bitcount_demo/flow_optimized.py \
  --quantum-source real-device \
  --runtime-block qrmi-runtime \
  --command-block cmd-bitcount-hist \
  --execution-profile-block exec-bitcount-mpi \
  --hpc-profile-block hpc-miyabi-bitcount \
  --options-variable miyabi-bitcount-options
```

Fugaku example:

```bash
python examples/prefect_bitcount_demo/flow_optimized.py \
  --quantum-source real-device \
  --runtime-block qrmi-runtime \
  --command-block cmd-bitcount-hist \
  --execution-profile-block exec-bitcount-fugaku \
  --hpc-profile-block hpc-fugaku-bitcount \
  --options-variable fugaku-bitcount-options
```

`flow_optimized.py` resolves the base work directory in this order:
1. `--work-dir` (if provided)
2. `work_dir` in the options variable (`miyabi-bitcount-options` / `fugaku-bitcount-options`)
3. fallback: `./work/prefect_bitcount_optimized`

The scheduler script suffix is resolved from the selected `HPCProfileBlock`, so the same flow can use `.pbs` on Miyabi and `.pjm` on Fugaku without changing the flow code.

If you want to run the tutorial without quantum sampling, switch the quantum source:

```bash
python examples/prefect_bitcount_demo/flow_optimized.py \
  --quantum-source random \
  --random-seed 24 \
  --command-block cmd-bitcount-hist \
  --execution-profile-block exec-bitcount-mpi \
  --hpc-profile-block hpc-miyabi-bitcount \
  --options-variable miyabi-bitcount-options
```

In `random` mode, the flow skips `QrmiRuntimeBlock.aload_runtime(...)` and generates deterministic pseudo-random bitstrings using the requested shot count.

## Quantum runtime block

With `--quantum-source real-device`, the flows load the `QrmiRuntimeBlock` named by
`--runtime-block` (default `qrmi-runtime`) and sample the GHZ circuit untranspiled. The
runtime transpiles it against the actual Target with its default transpiler at
optimization level 3 and seed 123, as the previous host-side transpilation did. The
flow sets these, so the block's `transpiler` and `transpile_config` are not used. The
block's `scheduler` selects where QRMI runs:

- `scheduler = "iqp"`: in the flow process, without Slurm.
- `scheduler = "slurm"`: in a Slurm job, through the block names stored in the runtime block.

Create the block as described in the
[QRMI Sampler guide](../../docs/howto/howto_qrmi_slurm_sampler.md), for example:

```python
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

QrmiRuntimeBlock(
    scheduler="iqp",
    resource="ibm_kingston",
    resource_config={
        "QRMI_IBM_QCS_ENDPOINT": "https://quantum.cloud.ibm.com/api/v1",
        "QRMI_IBM_QCS_IAM_ENDPOINT": "https://iam.cloud.ibm.com",
        "QRMI_IBM_QCS_IAM_APIKEY": "<your API key>",
        "QRMI_IBM_QCS_SERVICE_CRN": "<your service instance CRN>",
    },
).save("qrmi-runtime", overwrite=True)
```

The sampler options variable may contain only `{"params": {"shots": <positive integer>}}`
(plus `work_dir` for `flow_optimized.py`); other primitive options are rejected.

## Run legacy tutorial-style flow

The legacy tutorial assets are opt-in. Create them first on Miyabi:

```bash
python examples/prefect_bitcount_demo/create_blocks.py \
  --config examples/prefect_bitcount_demo/bitcount_blocks.toml \
  --hpc-target miyabi \
  --create-legacy-tutorial-assets
```

```bash
python examples/prefect_bitcount_demo/flow_tutorial_style.py \
  --quantum-source real-device
```

That creates these backward-compatible names:

- BitCounter block: `miyabi-tutorial`
- Prefect Variable: `miyabi-tutorial`

In the legacy flow, you can also override only the `HPCProfileBlock` at runtime
when the stored execution profile is already compatible with the target:

```bash
python examples/prefect_bitcount_demo/flow_tutorial_style.py \
  --bitcounter-block miyabi-tutorial \
  --options-variable miyabi-tutorial \
  --quantum-source random \
  --random-seed 24 \
  --hpc-profile-block-override hpc-fugaku-bitcount
```

## Merged Demo with `flow_optimized.py`

If you want to show Miyabi/Fugaku switching on the recommended path, use `flow_optimized.py`.
`flow_tutorial_style.py` is the legacy-compatible path, so it is not the recommended demo route for backend switching.

Use target-specific `ExecutionProfileBlock` and `HPCProfileBlock` names, while keeping the command block shared.
You can also keep `options_variable_name` shared when sampler options and `work_dir` are compatible across both targets.

Example config naming:

```toml
# Miyabi config
execution_profile_block_name = "exec-bitcount-miyabi"
hpc_profile_block_name = "hpc-miyabi-bitcount"
options_variable_name = "bitcount-options"

# Fugaku config
execution_profile_block_name = "exec-bitcount-fugaku"
hpc_profile_block_name = "hpc-fugaku-bitcount"
options_variable_name = "bitcount-options"
```

Then create the target-specific execution/HPC profile pairs:

```bash
python examples/prefect_bitcount_demo/create_blocks.py \
  --config examples/prefect_bitcount_demo/bitcount_blocks.miyabi.toml \
  --hpc-target miyabi

python examples/prefect_bitcount_demo/create_blocks.py \
  --config examples/prefect_bitcount_demo/bitcount_blocks.fugaku.toml \
  --hpc-target fugaku
```

If `bitcount-options` stores different `work_dir` values per target, pin a common demo directory with `--work-dir`.

After that, the optimized flow demo switches by changing the execution/HPC profile pair at runtime:

```bash
python examples/prefect_bitcount_demo/flow_optimized.py \
  --quantum-source real-device \
  --runtime-block qrmi-runtime \
  --command-block cmd-bitcount-hist \
  --execution-profile-block exec-bitcount-miyabi \
  --hpc-profile-block hpc-miyabi-bitcount \
  --options-variable bitcount-options \
  --work-dir /path/to/shared/bitcount_demo

python examples/prefect_bitcount_demo/flow_optimized.py \
  --quantum-source real-device \
  --runtime-block qrmi-runtime \
  --command-block cmd-bitcount-hist \
  --execution-profile-block exec-bitcount-fugaku \
  --hpc-profile-block hpc-fugaku-bitcount \
  --options-variable bitcount-options \
  --work-dir /path/to/shared/bitcount_demo
```

In these two commands, the changing runtime parameters are `--execution-profile-block` and `--hpc-profile-block`.
