# Quantum Sources

`qcsc_prefect.integrations.qiskit.sources` gives applications one synchronous call for
bitstrings, whatever produces them:

```python
result = source.sample(circuit, shots=shots)   # a PrimitiveResult with one PUB
bits = result[0].data.meas
```

| Source | Produces | `backend_name` | `requires_circuit` |
| --- | --- | --- | --- |
| `RandomSource` | uniform random bits, without a circuit | `"random"` | `False` |
| `QrmiSource` | samples from an existing `QrmiRuntime` (IQP or Slurm) | the QRMI resource | `True` |
| `QFwSource` | samples from the QFw-SLURM launcher | the QFw device ID | `True` |

All three implement the abstract base class `QuantumSource`. The API uses Qiskit's
`QuantumCircuit` and `PrimitiveResult`, and has no application concepts.

## Contract

- `sample()` is synchronous and returns one PUB.
- Sources that run circuits return every measured shot, with one `BitArray` per
  classical register of the circuit. They do not filter the shots, for example for
  reset mitigation; the application does that.
- Shots must be a positive integer. Sources that need a circuit also reject a missing
  circuit, unbound parameters and a circuit without measurements. Both are checked
  before any work starts.
- `requires_circuit` only says whether `sample()` needs a circuit. Use it to skip
  building a circuit for `RandomSource`; it says nothing about error mitigation.
- Importing the package does not import QRMI, QFw or DEFw.

## Responsibilities

| Package | Application |
| --- | --- |
| The interface, sources and common `QuantumSourceBlock` | Choosing a saved source Block |
| Loading source blocks and constructing the selected source | Deployment settings (block names, tags and node script arguments) |
| Validating shots and circuits | Building circuits |
| Converting platform results into a `PrimitiveResult` | Algorithm-specific preparation and compilation settings |
| Saving the QFw device settings (`QFwResource`) | Random width and seed |
| | Interpreting and filtering results, telemetry |

## Select any platform with one saved Block

`QuantumSourceBlock` is a common Prefect Block in this library, with no dependency on
SBD or any other algorithm.

To make the source and QFw resource types available in the UI before saving an
instance, register them with:

```bash
prefect block register -m qcsc_prefect.integrations.qiskit.sources
```

Saving an instance also registers its type automatically.

Save one Block per sampling configuration:

```python
from qcsc_prefect.integrations.qiskit.sources import QuantumSourceBlock, QuantumSourceConfig

QuantumSourceBlock(
    config=QuantumSourceConfig(kind="random"), num_bits=20, seed=42,
).save("samples-local", overwrite=True)

QuantumSourceBlock(
    config=QuantumSourceConfig(
        kind="qrmi", block_name="my-runtime", tags=("app:qpu",),
    ),
).save("samples-cloud", overwrite=True)

QuantumSourceBlock(
    config=QuantumSourceConfig(
        kind="qfw", block_name="my-qfw-resource",
        sampler="/shared/node_sampler.py",
        sampler_arg_names=(("optimization_level", "--optimization-level"),),
    ),
    transpile_config={"optimization_level": 2},
).save("samples-reserved", overwrite=True)
```

The QRMI and QFw settings reference existing `QrmiRuntimeBlock` and `QFwResource`
Blocks. Credentials stay in those Blocks. For QRMI, the referenced runtime's scheduler
selects IQP or Slurm. QFw's node script, argument mapping and work directory are in
`config`, so the algorithm does not contain any platform selection or CLI handling.

Unknown configuration keys are rejected. QRMI and QFw require a nonempty
`config.block_name`, and QFw also requires a nonempty `config.sampler`. These checks
run when the settings or source Block are constructed, without loading a backend.

Every algorithm uses the same call, changing only the saved Block name:

```python
source = QuantumSourceBlock.load_source(source_block_name)
circuit = build_circuit() if source.requires_circuit else None
result = source.sample(circuit, shots=1000)
```

`source_block_name` can be `"samples-local"`, `"samples-cloud"` or `"samples-reserved"`;
these names have no special meaning to the library or algorithm. Editing the saved
configuration under the same name also switches the platform without changing the
flow parameter. Standard `QuantumSourceBlock.load(name)` returns the Block; call
`block.get_source()` to construct a source from an already loaded Block.

Create the source inside the sampling worker. The parent passes only the Block name,
not live sources, runtimes or Blocks. Loading the source Block does not initialize a
platform; `get_source()` loads only its selected runtime/resource Block and does not
submit work. `sample()` is the submission boundary.

### Algorithm-owned policy

Applications can supply dynamic width/seed and their preparation or compilation
policy without knowing the selected platform:

```python
source = QuantumSourceBlock.load_source(
    source_block_name,
    num_bits=required_bit_width,
    seed=sample_seed,
    prepare=my_prepare,  # optional: QuantumCircuit -> PreparedCircuit
    transpiler="my_application.compile:transpile",
    transpile_config=my_compilation_settings,
)
```

The Block can store default `num_bits`, `seed`, `transpiler` and `transpile_config`.
Explicit application values override those defaults. An explicit compilation
configuration replaces the saved configuration; if omitted in both places, QRMI
keeps the referenced runtime's configuration. Random ignores device policy.

`transpiler` is the QRMI compiler hook. QFw uses the configured node script instead;
do not save a `transpiler` on a QFw source Block. An algorithm can still pass its
QRMI hook through the common call above while using QFw's script for the same policy.

QFw takes the circuit from `PreparedCircuit`, with preparation applied once before
serialization, and maps compilation values to `config.sampler_arg_names` in order,
without changing their types. QFw rejects nonempty `PreparedCircuit.config` overrides
before submission. Fixed CLI arguments can be passed through `config.sampler_args`;
`config.work_dir` defaults to `~/qfw_jobs`.

Every supplied QFw compilation setting must have a `sampler_arg_names` entry;
unmapped values raise an error before a platform Block is loaded. Mapping names and
flags must be unique, and fixed `sampler_args` cannot repeat a mapped flag (including
the `--flag=value` form). Saved settings are checked on Block construction; settings
supplied dynamically by an algorithm are checked when its source is constructed.
When a circuit contains gates with Python attributes lost by QPY, the application
must supply `prepare` to lower those gates before serialization.

The library also exposes `load_quantum_source(config, **policy)` for applications that
already have a `QuantumSourceConfig` and do not need to save a source Block. The config
is serializable with `dataclasses.asdict` and JSON. Missing source or platform Blocks
raise `SourceLoadError`, retaining the underlying exception as the cause. Direct
construction of the source classes below remains available.

### Closed-loop SQD migration

The closed-loop SQD flow's `quantum_source` parameter now names a saved
`QuantumSourceBlock`, defaulting to `"quantum-source"`. It is no longer an enum with
built-in `random`, `real-device` and `qfw` behavior. Existing parameter values still
work if source Blocks with those exact names are saved. Random mode also needs a
source Block, though it never loads a runtime/resource Block or a platform SDK.

The algorithm's `[source]` configuration creates the common Block. See its
[README](https://github.com/qiskit-community/qcsc-prefect/tree/main/algorithms/closed_loop/sqd_de#quantum-sampling)
for configuration. Other algorithms can adopt the same Block API independently;
the open-loop flows retain their current parameter behavior.

## `RandomSource`

```python
from qcsc_prefect.integrations.qiskit.sources import RandomSource

source = RandomSource(num_bits=20, seed=42)
result = source.sample(shots=1000)
bits = result[0].data.meas         # 1000 shots of 20 bits; no "test" register
```

The same seed, shots and width always give the same bits. They are the bits of
`qiskit_addon_sqd.counts.generate_bit_array_uniform`, produced with NumPy only. A
circuit, if passed, is ignored: this is synthetic input, not a simulation.

## `QrmiSource`

```python
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock
from qcsc_prefect.integrations.qiskit.sources import QrmiSource

runtime = QrmiRuntimeBlock.load_runtime("my-qrmi-runtime")
source = QrmiSource(runtime, tags=("res: quantum",))
result = source.sample(circuit, shots=1000)
```

`QrmiSource` calls `runtime.sampler([(circuit,)], shots=shots)` and uses the runtime as
given: its scheduler, `prepare`, `transpiler` and `transpile_config`. Set application
defaults on the runtime first, for example with `dataclasses.replace`. `tags` are
applied to the runtime's Prefect sampling task, so a Prefect concurrency limit can use
them; pass a tuple or list, since a single string is rejected. The resource ID is checked when the source is built.

Each call runs the async sampler in a new event loop in the calling thread. Do not call
`sample()` from a thread that already runs an event loop, such as an `async` flow or
task: it then raises `RuntimeError` before submitting. In async code, call
`await runtime.sampler(...)` directly. Separate calls, in threads or processes, share
no QRMI object.

See [QRMI Sampler](howto_qrmi_slurm_sampler.md) for the runtime and its schedulers.

## `QFwSource`

```python
from qcsc_prefect.integrations.qiskit.sources import QFwResource, QFwSource

resource = QFwResource.load("qfw-runner")
source = QFwSource(
    resource.device_id,
    sampler="/shared/path/to/sampler_script.py",
    sampler_args=["--optimization-level", 3],
    prepare=my_prepare,          # optional: QuantumCircuit -> QuantumCircuit
    work_dir="~/qfw_jobs",       # the default
)
result = source.sample(circuit, shots=1000)
```

`QFwSource` calls the QFw-SLURM launcher
`qfw_slurm_sampler.run_sampler_job(circuit, shots=..., qpu=device_id, work_dir=...,
sampler=..., sampler_args=...)` directly, in the calling thread. It imports the launcher
on the first `sample()`, from `QFW_SLURM_BIN` if that is set, and raises `RuntimeError`
if it is not installed. `sampler` is the node-side script and `sampler_args` its
arguments, passed unchanged.

`prepare` runs once, on a copy of the circuit, before the launcher serializes it, and
must keep the classical registers. Use it for steps that need gate objects QPY does not
keep.

The launcher's counts become one `BitArray` per classical register:

- Count keys list the registers in reverse order, in binary, with or without spaces.
  Hexadecimal keys are rejected.
- Every register of a shot comes from the same count entry, so `test[i]` and `meas[i]`
  belong to the same measured shot. The original order of the shots is not kept.
- Leading zeros and register widths are kept, and the registers are in the circuit's
  order, as Qiskit samplers return them.
- The counts must add up to `shots`; otherwise `sample()` raises `RuntimeError`.

The rest of the launcher record, for example the job ID, is in
`result.metadata["qcsc_qfw"]`.

### `QFwResource`

`QFwResource` (block type slug `qfwresource`) stores `device_id`,
`device_access_config` and `interface`. Importing or loading it does not import QFw or
DEFw, so it can be loaded in a worker thread. `QFwSource` uses only `device_id`. See
[QFw Resource Block](howto_setup_qfw_resource_block.md) to register and create it.

## Writing a source

Subclass `QuantumSource` and implement `backend_name` and `requires_circuit` as
`@property`, and `sample()`:

```python
from qcsc_prefect.integrations.qiskit.sources import QuantumSource


class MySource(QuantumSource):
    @property
    def backend_name(self) -> str:
        return "my-backend"

    @property
    def requires_circuit(self) -> bool:
        return True

    def sample(self, circuit=None, *, shots):
        ...
```

A dataclass field without a default does not implement an abstract property: the class
then cannot be instantiated.
