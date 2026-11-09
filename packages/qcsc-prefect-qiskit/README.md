# QCSC Prefect Qiskit

Native Qiskit Runtime integration utilities for QCSC Prefect workflows.

QRMI sampling on IBM Quantum is available as
`qcsc_prefect.integrations.qiskit.qrmi.QrmiRuntime`, with settings that can be
saved as a `QrmiRuntimeBlock`.

- `scheduler="slurm"` (the default) runs in a Slurm job. Install the `slurm` extra
  on the workflow host to include the HPC executor. The compute-node entrypoint is
  `python -m qcsc_prefect.integrations.qiskit.qrmi.worker` or `qcsc-qrmi-sampler`.
- `scheduler="iqp"` runs in the workflow process, without Slurm or a session.
  Install `qrmi[ibm]` on that host.

`qcsc_prefect.integrations.qiskit.sources` gives applications one synchronous
`source.sample(circuit, shots=shots)` call over `RandomSource` (uniform random bits,
no circuit), `QrmiSource` (an existing `QrmiRuntime`) and `QFwSource` (the QFw-SLURM
launcher, with the `QFwResource` block). See
[Quantum Sources](../../docs/howto/howto_use_quantum_source.md).

See [QRMI Sampler](../../docs/howto/howto_qrmi_slurm_sampler.md) for configuration,
supported calls, optional compilation hooks, diagnostics and offline tests.
The [Slurm Bell example](../../examples/qrmi_slurm_bell/README.md) provides a minimal
flow and the three required execution blocks; the
[IQP Bell example](../../examples/qrmi_iqp_bell/README.md) needs only a runtime block.
