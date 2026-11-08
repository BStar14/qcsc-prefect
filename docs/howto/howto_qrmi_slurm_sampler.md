# QRMI Sampler

`qcsc_prefect.integrations.qiskit.qrmi.QrmiRuntime` samples one or more bound,
measured circuits on IBM Quantum through QRMI and returns a native
`PrimitiveResult`. It has two schedulers. Both connect to IBM Quantum through
QRMI; they differ in where the code runs:

| | `scheduler="slurm"` (default) | `scheduler="iqp"` |
| --- | --- | --- |
| Runs in | a Slurm job on a compute node | the workflow (Prefect worker) process |
| Session | the Batch that SPANK acquires | none: one job per `sampler()` call |
| QRMI settings | the runtime's CRN, endpoints and session mode, through the job environment | the runtime's settings |
| API key | `qrmi_config.json` on the compute nodes | `qrmi_config.json` on the workflow host, or `QRMI_IBM_QCS_IAM_APIKEY` |
| Needs | shared `root_dir`, three execution blocks, SPANK | network access to IBM Quantum |
| Resource types | as configured for SPANK | `ibm-quantum-compute-service` only |

With `scheduler="slurm"`, the workflow host writes QPY and JSON to shared
storage. The compute node acquires the QRMI Target, transpiles each circuit,
and launches `task_runner` once for all circuits. Direct execution with
`scheduler="iqp"` is described in [Direct IQP execution](#direct-iqp-execution).
Other scheduler values raise `ValueError`.

## Responsibilities

| Library | Application |
| --- | --- |
| QPY/JSON transport, Slurm submission and monitoring | Circuit construction |
| Batch environment checks and compute-node Target acquisition | Preparation of custom gates for QPY |
| Compilation hook invocation and ISA validation | Application-specific compilation passes |
| CLI payload, process invocation and result decoding | Interpretation and postprocessing of samples |
| Validation of every classical register's name, width and shots | Resource and compilation settings |

The library uses installed modules and does not import the examples or test
fixtures. QRMI is imported only when sampling: inside the compute-node
entrypoint for Slurm, or inside the sampling task for IQP. The HPC executor is
imported when a Slurm sampling task runs, keeping it optional for native Qiskit
users. The Bell circuit examples in `examples/qrmi_slurm_bell` and
`examples/qrmi_iqp_bell` use the default compiler and only the public sampler API.
The SQD workflow in `algorithms/open_loop/sqd` is a full application of both hooks:
an ffsim `prepare` step on the workflow host and an importable `transpiler` (see its
[README](https://github.com/qiskit-community/qcsc-prefect/tree/main/algorithms/open_loop/sqd)).

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
Set `launcher="single"` on the execution profile and `hpc_target="slurm"` on the HPC
profile. The job requests the runtime's `resource` as `--qpu`, so the HPC profile
needs no `slurm_qpu`.
The work directory must be readable and writable at the same absolute path on
both the workflow host and compute nodes.

SPANK must acquire a Batch before Python starts. Configure the selected resource
with `QRMI_IBM_QCS_SESSION_MODE="batch"` on every eligible node. The worker checks
the resource-prefixed mode and acquisition-token presence before execution.
It does not change the mode or print the token.

The service CRN, the endpoints and the session mode come from the runtime.
`QrmiRuntime(job_settings={...})` takes them by their QRMI names without the
`<resource>_` prefix (`QrmiRuntimeBlock` builds it from its instance and fields).
They are passed to `sbatch` in its environment as `<resource>_QRMI_IBM_QCS_*`, not
written into the script, and SPANK sets them with priority over the file. The
API key cannot travel this way: SPANK reads per-job settings only from the job's
environment, which Slurm records, and QRMI has no key-file setting. So
`QRMI_IBM_QCS_IAM_APIKEY` stays in `qrmi_config.json` on the compute nodes, and a
Slurm runtime with an API key is rejected. An API key in the workflow process's
environment is removed from the `sbatch` environment.

### Set up access with `qcsc-qrmi login`

`qcsc-qrmi login` asks for the IBM Cloud API key, finds the IBM Quantum instances
it can use and their backends (through qiskit-ibm-runtime), and writes
`qrmi_config.json`: `--config PATH`, else `$QRMI_CONFIG`, else
`~/.qiskit/qrmi_config.json`, with mode 600. `--no-discover` asks for each
instance's name, plan, CRN and backends instead; `--api-key-env VAR` reads the key
from an environment variable instead of the prompt.

```json
{
  "environment": {"QRMI_IBM_QCS_IAM_APIKEY": "..."},
  "default_instance": "main-instance",
  "instances": [
    {"name": "main-instance", "plan": "premium", "crn": "crn:v1:...",
     "backends": ["ibm_fez", "ibm_kingston"]},
    {"name": "open-instance", "plan": "open", "crn": "crn:v1:...",
     "backends": ["ibm_kingston"]}
  ],
  "resources": [
    {"name": "ibm_fez", "type": "ibm-quantum-compute-service",
     "environment": {"QRMI_IBM_QCS_IAM_APIKEY": "..."}},
    {"name": "ibm_kingston", "type": "ibm-quantum-compute-service",
     "environment": {"QRMI_IBM_QCS_IAM_APIKEY": "..."}}
  ]
}
```

- The API key is set once, in `environment`.
- `instances` lists each instance once, with its CRN and backends.
- `resources` lists each backend once, whichever instances can use it.
- SPANK reads only an entry's own `environment`, so each entry carries a copy of
  the key; `login` and `refresh` keep the copies equal to the top-level key.
- The entries hold nothing else: the CRN, endpoints and session mode come from the
  runtime block for every job.
- QRMI and SPANK ignore the other top-level keys, and `login` keeps entries of
  other resource types.

SPANK reads the file named in `plugstack.conf` on each compute node, so install the
written file there (often `/etc/slurm/qrmi_config.json`) or pass that path as
`--config`. `login` also saves the QRMI Runtime block (`--block`, default
`qrmi-runtime`) with your instances, their backends, and dropdowns for IBM Quantum
Instance and QPU Backend, and prints the `[qrmi]` lines of `sqd_blocks.toml` that
keep the choice. `qcsc-qrmi refresh` repeats the discovery with the stored key,
`qcsc-qrmi list` prints the instances and backends (never the key), and
`qcsc-qrmi check` checks the block's instance and backend.

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

Each PUB is a one-element `(circuit,)` tuple. Bind parameters before calling the
runtime. Each circuit must contain measurements and named classical registers
covering all classical bits. The default is 100 shots. The equivalent supported
options form is:

```python
result = await runtime.sampler(
    sampler_pubs=[(circuit,)], options={"params": {"shots": 100}}
)
```

Several bound circuits can be sampled in one Slurm job and one QRMI sampler
call. Result `i` corresponds to PUB `i`, and all PUBs use the same shots:

```python
result = await runtime.sampler([(circuit_a,), (circuit_b,)], shots=100)
counts_a = result[0].data.meas.get_counts()
counts_b = result[1].data.meas.get_counts()
```

Conflicting `shots=` and `options["params"]["shots"]` values fail before submission.
Shots must be positive integers; booleans, strings and floats are rejected.
Other options, parameter-value PUBs and PUB-specific shots are rejected explicitly.
If any PUB is invalid, the whole call fails before a job directory is created.
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

With several PUBs, `transpile_config` is the base configuration for every PUB.
`prepare` runs once per circuit, and each `PreparedCircuit.config` overrides the
base for that circuit only. Each PUB's final configuration is validated against
its own prepared circuit, and is passed to the compiler with that circuit.
A non-`None` `initial_layout` in the base configuration therefore requires every
prepared circuit to have the same number of qubits; return a per-circuit
`initial_layout` from `prepare` when the circuits differ. Without a layout,
circuits of different widths are accepted. Errors name the failing PUB index.
Qiskit validates target-dependent constraints and installed plugin names on the
compute node. Target/backend, basis-gate and coupling-map overrides are excluded:
the compiler always uses the actual Target acquired inside the Slurm allocation.
`sampler(options=...)` continues to configure shots only.

Every resulting non-barrier instruction is checked against the actual Target
before `task_runner` is called.

## Optional compilation hooks

For application-specific preparation, supply
`prepare(circuit) -> PreparedCircuit`. This function runs before serialization
on a copy of each circuit and returns a QPY-safe circuit plus JSON configuration.
The caller's original circuit is not modified. QPY preserves circuit definitions;
applications using custom Python gate attributes should prepare those gates
before transport.

For a custom compute-node compiler, supply `transpiler="module:function"`.
The installed function is called once per circuit as
`function(circuit, target, config)` and must return a bound, measured
`QuantumCircuit`. The module must be importable in the
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

- `circuit.qpy`: the circuits to transport, in PUB order.
- `sampler_request.json`: schema version 2, resource, shots, a list with the
  registers of each PUB, and the compiler hook.
- `transpile_config.json`: a list with the final compiler configuration of each PUB.
- `sampler.slurm` and scheduler stdout/stderr.
- `isa.qpy` and `transpile_report.json`: compiled circuits in PUB order, Target
  information, and a `pubs` list with the configuration, operation counts and
  registers of each PUB.
- `sampler_input.json` and raw `result.json`: CLI input and encoded output.
- `slurm_job.json`: scheduler state and execution record when monitoring returns.

Both worker and caller require one result PUB per input PUB, in the same order,
and validate every classical register's name, width, scalar shape and shots.
A zero CLI exit code alone is not sufficient. Failed jobs and malformed results
retain their diagnostic files.

Update the workflow and compute-node installations together. The worker rejects
a request whose schema version differs from its own, including schema version 1
requests written by an older workflow-host installation. The worker accepts
`--request sampler_request.json`; verify imports and worker `--help` output on
every eligible node before submitting work. The default CLI executable is
`task_runner` beside the worker's Python; `--task-runner` can override it.

## Direct IQP execution

`scheduler="iqp"` runs QRMI in the workflow process and needs no Slurm cluster,
shared directory or compute-node installation. Install QRMI with its IBM extra
on that host (`pip install "qrmi[ibm]"`; QRMI publishes Linux wheels). The host
must reach the IBM Quantum and IBM Cloud IAM endpoints.

With a `QrmiRuntimeBlock`, the key comes from the workflow host's
`qrmi_config.json` that `qcsc-qrmi login` writes, or from the
`QRMI_IBM_QCS_IAM_APIKEY` environment variable. Built directly, the runtime takes
the settings and the key in `resource_config`:

```python
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime

runtime = QrmiRuntime(
    scheduler="iqp",
    resource="ibm_kingston",
    resource_config={
        "QRMI_IBM_QCS_ENDPOINT": "https://quantum.cloud.ibm.com/api/v1",
        "QRMI_IBM_QCS_IAM_ENDPOINT": "https://iam.cloud.ibm.com",
        "QRMI_IBM_QCS_IAM_APIKEY": "<your API key>",
        "QRMI_IBM_QCS_SERVICE_CRN": "<your service instance CRN>",
    },
)
result = await runtime.sampler([(circuit_a,), (circuit_b,)], shots=100)
```

`resource_config` holds QRMI's IBM Quantum Compute Service settings, using the
QRMI environment variable names without the `<resource>_` prefix. It is passed
to `QuantumResource.from_config` and is not read from or written to the process
environment, so several flows in one worker can use different credentials.
The runtime hides it from its `repr`. `root_dir`, the Slurm block names and
`resource_config` with `scheduler="slurm"` are rejected.

The PUB, shots, `prepare`, `transpile_config` and `transpiler` rules are the same
as for Slurm. Each call:

1. Creates a QRMI resource for this call only. It is never shared, and its
   methods are called one at a time.
2. Fetches the Target with `qrmi.primitives.ibm.get_backend(resource).target`
   and compiles each circuit with the same code as the Slurm worker.
3. Submits all circuits as one job with `QuantumResource.task_start`, using the
   same SamplerV2 input as the Slurm worker, polls its status every
   `watch_poll_interval` seconds, and validates the result against the input PUBs.
   The job is tracked with QRMI's `RuntimeJobV2(..., delete_job=False)`, so
   discarding the job object never stops the job.

No session is used: the runtime never calls `acquire()` or `release()`, and
`resource_config` must not contain `QRMI_IBM_QCS_SESSION_ID` or
`QRMI_JOB_ACQUISITION_TOKEN`, which would attach the job to an existing session.
Each `sampler()` call is a separate job in the IQP queue.

The result's `metadata["qcsc_qrmi"]` records the resource, job ID, submission
time, final status, Target width and the per-PUB compilation report.

### Failures and cancellation

- A job that ends as `ERROR` or `CANCELLED` raises `IqpJobFailed`.
- Cancellation is accepted between status polls. This holds for anyio
  cancellation, `Task.cancel()` and `asyncio.timeout()`. A QRMI call that is
  already running in its worker thread is awaited to completion before anything
  else uses the resource, even if cancellation is requested again; its wait
  follows QRMI's own network timeouts. Direct IQP execution requires an asyncio
  event loop.
- If the job ID is known and the call fails or is cancelled before the job ends,
  the runtime calls `job.cancel()` once, in a worker thread. This includes a
  cancellation that arrives while submission is running: the job ID it returns is
  kept and the job is stopped. A failed cancel is logged and not retried.
- A job that completed, or that ended on its own, is never stopped.
- If submission itself raises, it is unknown whether a job was created.
  The runtime raises `IqpSubmissionUnknown` with the resource, time and error,
  and does **not** resubmit. Check the IBM Quantum dashboard before retrying.
  If the call was cancelled while submission ran and submission then raised,
  the cancellation propagates with that `IqpSubmissionUnknown` as its cause, and
  the submission error is also logged.
- Prefect task retries are disabled, as for Slurm.

### Runtime block

`QrmiRuntimeBlock` stores the runtime settings in Prefect. Its form shows, in order:

| Field | Meaning |
| --- | --- |
| Scheduler | `slurm` or `iqp` |
| IBM Quantum Instance | the instance, as `name (plan)`; a dropdown after `qcsc-qrmi login` |
| QPU Backend (`resource`) | the backend; a dropdown of every instance's backends, each once |
| Service CRN | empty: the instance's CRN; set it to use another |
| Instances | `name (plan)` → CRN, filled by `login` |
| IBM Quantum Endpoint, IAM Endpoint, Session Mode | IBM Quantum's URLs and `batch` by default |
| Backend Access | the backends each instance can use, filled by `login` and `refresh` |

The block holds no API key. `check_backend_access()` raises `BackendAccessError`
when the instance cannot use the backend: from the Backend Access list, or, when
the host has the API key, by asking IBM Quantum, whose answer wins over a stale
list. `qrmi_runtime_block_class(instances, backends)` returns the block class with
the two dropdowns; save with it, and load with `QrmiRuntimeBlock`, which accepts
any value. `resource_config` (secret, IQP only) remains for blocks saved earlier.
`prepare` is a Python callable and cannot be stored; pass it when building the
runtime:

```python
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

QrmiRuntimeBlock(
    scheduler="iqp",
    resource="ibm_kingston",
    instance="main-instance (premium)",
    instances={"main-instance (premium)": "<your service instance CRN>"},
).save("my-qrmi-runtime", overwrite=True)

runtime = await QrmiRuntimeBlock.aload_runtime("my-qrmi-runtime", prepare=prepare)
```

`QrmiRuntimeBlock.load(name)` keeps Prefect's behavior and returns the block;
call `get_runtime(prepare)` on it, or use `aload_runtime` (or `load_runtime` in
synchronous code). The same block type stores Slurm settings (`root_dir` and the
three block names) with `scheduler="slurm"`.

## Local tests

From the repository root, with the workspace dependencies installed:

```bash
uv run --no-sync pytest \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_slurm_runtime.py \
  packages/qcsc-prefect-qiskit/tests/test_qrmi_iqp_runtime.py \
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
The IQP tests replace the `qrmi` module with a fake that records every call and
fails if one QRMI resource is used from two threads at the same time. They cover
success, job failure, cancellation while submitting and while waiting, unknown
submission results, session settings and the runtime block.
These local tests do not establish hardware fidelity or cluster deployment.
