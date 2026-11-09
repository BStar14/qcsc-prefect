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
| The interface and the three sources | Choosing and configuring a source |
| Validating shots and circuits | Building circuits |
| Converting platform results into a `PrimitiveResult` | Algorithm-specific preparation and compilation settings |
| Saving the QFw device settings (`QFwResource`) | Random width and seed |
| | Interpreting and filtering results, telemetry |

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
