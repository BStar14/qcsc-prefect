# Bell sampling with QRMI and Slurm

This example creates a two-qubit Bell circuit, measures both qubits, and submits
it through `QrmiSlurmRuntime`. It uses the installed compute-node worker and
default compiler. It needs no custom compiler module, test fixture, or source-path
modification. The Python flow returns a dictionary with width, shots, counts and
Slurm metadata; the CLI prints it as JSON.

## Prerequisites

- Install the workspace Qiskit integration and executor on the workflow host.
- Install the same Qiskit integration revision and QRMI on each compute node.
- Provide the `task_runner` executable beside the selected compute-node Python.
- Mount the work directory at the same absolute path on both hosts.
- Configure SPANK to acquire the selected resource in Batch mode before Python starts.
- Use the intended Prefect workspace or server for the three saved blocks below.

See the [library guide](../../docs/howto/howto_qrmi_slurm_sampler.md) for installation,
request/result contracts, diagnostics and optional hooks.

## Register three quantum execution blocks

For a new setup, adapt the Python path, partition and QRMI resource below to the
cluster. Executing this snippet saves three blocks in the selected Prefect
workspace. Existing suitable blocks can instead be selected using CLI arguments.

```python
from qcsc_prefect_blocks.common.blocks import (
    CommandBlock, ExecutionProfileBlock, HPCProfileBlock,
)

CommandBlock(
    command_name="qrmi-bell",
    executable_key="python",
    default_args=["-m", "qcsc_prefect.integrations.qiskit.qrmi.worker"],
).save("qrmi-bell-command", overwrite=True)
ExecutionProfileBlock(
    profile_name="qrmi-bell",
    command_name="qrmi-bell",
    num_nodes=1,
    mpiprocs=1,
    ompthreads=1,
    walltime="00:10:00",
    launcher="single",
).save("qrmi-bell-execution", overwrite=True)
HPCProfileBlock(
    hpc_target="slurm",
    queue_cpu="debug",
    slurm_qpu="test_heron",
    slurm_ntasks=1,
    executable_map={"python": "/shared/venv/qrmi/bin/python"},
).save("qrmi-bell-hpc", overwrite=True)
```

The HPC profile's `slurm_qpu` must match the flow's `--resource`. The selected
Python must import the installed worker from an arbitrary job directory.
A read-only installation check on each compute node is:

```bash
/shared/venv/qrmi/bin/python -m qcsc_prefect.integrations.qiskit.qrmi.worker --help
```

## Run

From the source checkout on the workflow host, in the configured Python environment:

```bash
python -m examples.qrmi_slurm_bell.flow \
  --shots 100 --work-dir /shared/qrmi-bell --resource test_heron
```

This command submits a real Slurm and quantum workload. Each call submits a new
job; the flow and sampling task disable automatic retries. To select other blocks:

```bash
python -m examples.qrmi_slurm_bell.flow \
  --shots 100 --work-dir /shared/qrmi-bell --resource my_resource \
  --command-block-name my-command \
  --execution-profile-block-name my-execution \
  --hpc-profile-block-name my-hpc
```

Successful transport returns `num_bits: 2`, `num_shots: 100`, a count total of 100,
and Slurm state `COMPLETED` with exit status 0. `slurm.work_dir` identifies the
QPY, compiler report, CLI input, raw result and scheduler logs.
An ideal Bell state yields only `00` and `11`; real hardware may also return
other bitstrings. The example reports the observed distribution without applying
a fidelity threshold.

## Offline verification

```bash
uv run --no-sync pytest packages/qcsc-prefect-executor/tests/test_qrmi_slurm_bell.py -q
```

The test replaces the scheduler and QRMI service, runs the installed worker code,
and invokes a fake CLI subprocess. The fake CLI verifies the actual compiled Bell
state and produces encoded samples with the requested shots. No network connection
or QPU allocation is made. Hardware execution of this example is a separate check.
