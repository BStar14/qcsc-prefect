# Configure QFw Access to a Quantum Device

This guide describes how to configure QFw and QRMI to access a target quantum device for use with the `qcsc-prefect` SQD workflow.

For this workflow, QFw requires:

- a device-access configuration file describing the target quantum resource
- a credential file containing the provider credentials required for quantum job submission


## 1. Device-access configuration

QFw uses a device-access configuration file to describe the available quantum resources and how they should be accessed.

In the QFw-SLURM environment used by this tutorial, the configuration is created at:

```text
/workspace/qfw-container-base/device-access.yaml
```

The configuration used in this tutorial contains an entry similar to:

```yaml
ibm_fez:
  provider: ibm
  provider-device-id: ibm_fez
  resource-type: IBMQuantumComputeService
  url: "https://quantum.cloud.ibm.com/api/v1"
  service-crn: "REDACTED"
  credential-db: my-credentials.json
  libraries: [qrmi]
  preference: qrmi
  execution-owner: qrmi
  caps:
    get_device_info: [qrmi]
    get_coupling_graph: [qrmi]
    get_backend_info: [qrmi]
    get_dynamic_backend_info: [qrmi]
    run_circuit: [qrmi]
    get_task_timing: [qrmi]
    get_task_metadata: [qrmi]
    get_calibration_snapshot: [qrmi]
```

The important fields for this workflow are:

- `provider-device-id` — identifies the provider-side quantum device
- `url` and `service-crn` — identify the IBM Quantum Compute service used to access the device
- `credential-db` — identifies the credential database used for authentication
- `libraries` — lists the available QFw interfaces for the device
- `preference` — selects the preferred interface
- `execution-owner` — identifies the interface responsible for circuit execution
- `caps` — maps supported device operations to the interface that provides them

In this configuration, QRMI is used for device access and quantum circuit execution.

The exact configuration may differ for other providers or QFw environments.

## 2. Provider credentials

The device-access configuration references a credential database:

```yaml
credential-db: my-credentials.json
```

The credential file contains the provider authentication information used by QFw when submitting quantum jobs.

For example:

```json
{
  "users": {
    "root": {
      "enabled": true,
      "devices": {
        "ibm_fez": {
          "enabled": true,
          "api_key": "REDACTED"
        }
      }
    }
  }
}
```

The device name in the credential file should correspond to the device configured in `device-access.yaml`.

Replace `REDACTED` with a valid API key for the target quantum service.

## 3. Verify the configuration files

Inside the `slurmctld` container, verify that the device-access configuration is available:

```bash
ls -l /workspace/qfw-container-base/device-access.yaml
```

If the configuration references:

```yaml
credential-db: my-credentials.json
```

verify that the corresponding credential file is also available:

```bash
ls -l /workspace/qfw-container-base/my-credentials.json
```

The exact credential-file location depends on the QFw configuration used in your environment.
