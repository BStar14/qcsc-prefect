# QRMI/Slurm Sampler

`qcsc_prefect.integrations.qiskit.qrmi.QrmiRuntime` samples one bound,
measured circuit through a Slurm allocation and a QRMI IBM Quantum Batch.
The workflow host writes QPY and JSON to shared storage. The compute node
acquires the QRMI Target, transpiles the circuit, and launches `task_runner`.
The caller receives a native `PrimitiveResult`.

Select this execution path with `scheduler="slurm"` (the default).
Direct QRMI execution without Slurm (`scheduler="iqp"`) is not implemented
and raises `NotImplementedError` at construction; other scheduler values
raise `ValueError`.

## Responsibilities

| Library | Application |
| --- | --- |
| QPY/JSON transport, Slurm submission and monitoring | Circuit construction |
| Batch environment checks and compute-node Target acquisition | Preparation of custom gates for QPY |
| Compilation hook invocation and ISA validation | Application-specific compilation passes |
| CLI payload, process invocation and result decoding | Interpretation and postprocessing of samples |
| Validation of every classical register's name, width and shots | Resource and compilation settings |

The library uses installed modules and does not import the examples or test
fixtures. QRMI is imported only inside the compute-node entrypoint. The HPC
executor is imported when a sampling task runs, keeping it optional for native
Qiskit users. The Bell circuit example in `examples/qrmi_slurm_bell` uses the
default compiler and only the public sampler API.

## Installation and cluster configuration

Install the same library revision on the workflow host and every compute node.
The workflow host also needs the executor, declared by the `slurm` extra.
For this unreleased feature, install from the source checkout rather than an
older published wheel:

```bash
python -m pip install \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-executor \
  -e 'packages/qcsc-prefect-qiskit[slurm]'
```

To update this library in an already configured environment without upgrading
runtime dependencies:

```bash
python -m pip install --no-deps -e packages/qcsc-prefect-qiskit
python -m pip check
```

Retain the cluster's QRMI, SPANK and `task_runner` installation. The Python
selected by the CommandBlock must import the installed library. Configure its
default arguments to run the worker module:

```python
["-m", "qcsc_prefect.integrations.qiskit.qrmi.worker"]
```

The installation also provides the `qcsc-qrmi-sampler` console script.
Set `launcher="single"` on the execution profile, `hpc_target="slurm"` on the HPC
profile, and `slurm_qpu` to the same ID as the runtime's `resource`.
The work directory must be readable and writable at the same absolute path on
both the workflow host and compute nodes.

SPANK must acquire a Batch before Python starts. Configure the selected resource
with `QRMI_IBM_QCS_SESSION_MODE="batch"` on every eligible node. The worker checks
the resource-prefixed mode and acquisition-token presence before execution.
It does not change the mode or print the token.

## Sampling

```python
from pathlib import Path
from qiskit import QuantumCircuit
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

runtime = QrmiRuntime(
    root_dir=Path("/shared/quantum-jobs"),
    scheduler="slurm",
    command_block_name="my-qrmi-command",
    execution_profile_block_name="my-qrmi-execution",
    hpc_profile_block_name="my-qrmi-hpc",
    resource="test_heron",
)
result = await runtime.sampler([(circuit,)], shots=100)
counts = result[0].data.meas.get_counts()
```

Only one `(circuit,)` PUB is accepted. Bind parameters before calling the runtime.
The circuit must contain measurements and named classical registers covering all
classical bits. The default is 100 shots. The equivalent supported options form is:

```python
result = await runtime.sampler(
    sampler_pubs=[(circuit,)], options={"params": {"shots": 100}}
)
```

Conflicting `shots=` and `options["params"]["shots"]` values fail before submission.
Shots must be positive integers; booleans, strings and floats are rejected.
Other options, parameter-value PUBs and PUB-specific shots are rejected explicitly.
This facade returns results directly; it does not provide Runtime Block loading,
workflow-host Target acquisition or a Qiskit primitive job handle.

Each call creates a new directory and submits a new job. Sampling-task caching
and automatic retries are disabled. Calling again after failure submits again.
`watch_poll_interval` and `timeout_seconds` are passed to the existing executor.
`timeout_seconds` limits monitoring: expiry raises `WaitTimeout` without calling
`scancel`. The Slurm job and quantum workload may continue after the caller times out.
Single-job monitoring uses the controller while its matching record exists and
falls back to accounting when necessary; bulk monitoring is unchanged.

## Compilation options

The default compiler uses optimization level 1 and needs no configuration.
Pass preset pass manager options through `transpile_config` on the runtime:

```python
runtime = QrmiRuntime(
    root_dir=Path("/shared/quantum-jobs"),
    command_block_name="my-qrmi-command",
    execution_profile_block_name="my-qrmi-execution",
    hpc_profile_block_name="my-qrmi-hpc",
    resource="test_heron",
    transpile_config={
        "initial_layout": [12, 13],
        "optimization_level": 3,
        "seed_transpiler": 42,
        "routing_method": "sabre",
    },
)
result = await runtime.sampler([(circuit,)], shots=100)
```

`initial_layout[i]` is the physical qubit index for logical qubit `i` in the
prepared circuit. Use a list of distinct nonnegative integers, rather than a
Qiskit `Layout` object, so the setting can cross the JSON/QPY boundary. The list
must have exactly one entry per prepared circuit qubit and fit the compute-node
Target. Routing can change the final placement; the compiled circuit's layout is
retained in `isa.qpy`.

The supported keys match the corresponding Qiskit preset pass manager arguments:

- `initial_layout`, `optimization_level` (0–3), `seed_transpiler`.
- `layout_method`, `routing_method`, `translation_method`, `scheduling_method`.
- `init_method`, `optimization_method`, `approximation_degree`.
- `unitary_synthesis_method`, `unitary_synthesis_plugin_config`, `qubits_initially_zero`.

All values must be JSON serializable. Unknown keys, invalid optimization levels
and seeds, and malformed, duplicate or incorrectly sized layouts fail before
submission. Method options must be strings or `None`; `approximation_degree`
must be a number in [0, 1] or `None` (booleans are rejected).
`qubits_initially_zero` must be a boolean, and `unitary_synthesis_plugin_config`
must be a dictionary or `None`. Validation uses the prepared circuit and merged
configuration, including any overrides from `PreparedCircuit.config`.
Qiskit validates target-dependent constraints and installed plugin names on the
compute node. Target/backend, basis-gate and coupling-map overrides are excluded:
the compiler always uses the actual Target acquired inside the Slurm allocation.
`sampler(options=...)` continues to configure shots only.

Every resulting non-barrier instruction is checked against the actual Target
before `task_runner` is called.

## Optional compilation hooks

For application-specific preparation, supply
`prepare(circuit) -> PreparedCircuit`. This function runs before serialization
on a circuit copy and returns a QPY-safe circuit plus JSON configuration.
The caller's original circuit is not modified. QPY preserves circuit definitions;
applications using custom Python gate attributes should prepare those gates
before transport.

For a custom compute-node compiler, supply `transpiler="module:function"`.
The installed function is called as `function(circuit, target, config)` and must
return a bound, measured `QuantumCircuit`. The module must be importable in the
compute-node Python environment. Both hooks must preserve classical register
names and widths. `PreparedCircuit.config` overrides matching `transpile_config`
keys. This applies to both the default compiler's preset options and a custom
compiler's application-specific configuration.

To customize a PassManager, build and run it inside the hook using the supplied
Target. For example, install this module on both hosts as
`my_algorithm/transpilation.py`:

```python
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import RemoveBarriers


def transpile(circuit, target, config):
    pm = generate_preset_pass_manager(
        target=target, **{"optimization_level": 1, **config}
    )
    pm.pre_init = PassManager([RemoveBarriers()])
    return pm.run(circuit)
```

Select it when constructing the runtime:

```python
runtime = QrmiRuntime(
    root_dir=Path("/shared/quantum-jobs"),
    command_block_name="my-qrmi-command",
    execution_profile_block_name="my-qrmi-execution",
    hpc_profile_block_name="my-qrmi-hpc",
    resource="test_heron",
    transpiler="my_algorithm.transpilation:transpile",
    transpile_config={"initial_layout": [12, 13], "seed_transpiler": 42},
)
```

The custom hook owns its configuration validation; it can also consume its own
keys. A PassManager instance or local callable is not transported to the compute
node. Use `prepare` for workflow-host circuit preparation and the importable
`transpiler` hook for Target-dependent compilation.

Treat request and configuration JSON as trusted workflow-generated input.
Applications own their hook implementations and interpretation of results.

## Files and diagnostics

Each job retains:

- `circuit.qpy`: the circuit to transport.
- `sampler_request.json`: schema version 1, resource, shots, registers and compiler hook.
- `transpile_config.json`: optional compiler configuration.
- `sampler.slurm` and scheduler stdout/stderr.
- `isa.qpy` and `transpile_report.json`: compiled circuit and Target information.
- `sampler_input.json` and raw `result.json`: CLI input and encoded output.
- `slurm_job.json`: scheduler state and execution record when monitoring returns.

Both worker and caller require one result PUB and validate every classical
register's name, width, scalar shape and shots. A zero CLI exit code alone is
not sufficient. Failed jobs and malformed results retain their diagnostic files.

Update the workflow and compute-node installations together. The worker accepts
`--request sampler_request.json`; verify imports and worker `--help` output on
every eligible node before submitting work. The default CLI executable is
`task_runner` beside the worker's Python; `--task-runner` can override it.

## Local tests

From the repository root, with the workspace dependencies installed:

```bash
uv run --no-sync pytest \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_slurm_runtime.py \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_slurm_task.py \
  packages/qcsc-prefect-executor/tests/test_qrmi_slurm_bell.py \
  packages/qcsc-prefect-adapters/tests/test_slurm_runtime.py -q
```

Unit tests replace Slurm and QRMI service calls and block network access. The Bell
example test passes the circuit through real QPY serialization, default Target
compilation, QASM export, a fake CLI process and result decoding. The fake CLI
checks the compiled Bell state and returns deterministic encoded samples.
The separate QASM-import boundary test requires the optional
`qiskit-qasm3-import` package and is skipped when it is not installed.
A separate Prefect integration test starts a temporary local server with
`prefect_test_harness`. It runs a flow and the real sampling task with a
`QuantumCircuit` and a runtime containing a preparation callable, then checks the
completed task state and returned `BitArray`. Only the Slurm submission boundary
is replaced; the test makes no QRMI or QPU calls and rejects external connections.
These local tests do not establish hardware fidelity or cluster deployment.
