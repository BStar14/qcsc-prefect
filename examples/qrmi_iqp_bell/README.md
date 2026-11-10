# Bell sampling with QRMI directly on IQP

This example samples the same two-qubit Bell circuit as
[`qrmi_slurm_bell`](../qrmi_slurm_bell/README.md), but with
`QrmiRuntime(scheduler="iqp", ...)`. The flow runs QRMI in the Prefect worker
process: it fetches the Target, transpiles with the default compiler and submits
one IBM Quantum Compute Service job. No Slurm cluster, shared directory or
compute-node installation is used.

Both examples connect to IBM Quantum through QRMI. The difference is where the
code runs: inside a Slurm job (`slurm`) or in the workflow process (`iqp`).

## Prerequisites

- Install the Qiskit integration and QRMI with its IBM extra on the host that runs
  the flow (QRMI publishes Linux wheels):

  ```bash
  python -m pip install -e packages/qcsc-prefect-qiskit "qrmi[ibm]"
  ```

- The host must reach the IBM Quantum and IBM Cloud IAM endpoints.
- Use the intended Prefect workspace or server for the saved block below.

See the [library guide](../../docs/howto/howto_qrmi_slurm_sampler.md) for the
result contract, failure handling and options.

## Save the runtime block

`resource_config` uses QRMI's configuration keys without the `<resource>_`
prefix. It is stored as a secret. Do not set `QRMI_IBM_QCS_SESSION_ID` or
`QRMI_JOB_ACQUISITION_TOKEN`: `scheduler="iqp"` submits jobs without a session,
and the runtime rejects these keys.

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
).save("qrmi-iqp-bell", overwrite=True)
```

## Run

```bash
python examples/qrmi_iqp_bell/flow.py --shots 100 --runtime-block-name qrmi-iqp-bell
```

The flow prints the width, shots, counts, IQP job ID and resource as JSON.
Counts are mostly `00` and `11` on real hardware.
