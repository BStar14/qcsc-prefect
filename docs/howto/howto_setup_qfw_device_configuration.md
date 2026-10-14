# Configure QFw Access to a Quantum Device

This guide describes how to configure QFw and QRMI to access a target quantum device for use with the `qcsc-prefect` SQD workflow.

For this workflow, QFw requires:

- a device-access configuration file describing the target quantum resources
- a credential file containing the provider credentials required for quantum job submission

The exact configuration depends on the type of IBM Quantum resource being used. For example, QFw may access a device through an `IBMQuantumComputeService` resource for cloud-based access or an `IBMQuantumSystem` resource for direct system access.

## 1. Device-access configuration

QFw uses a device-access configuration file to describe the available quantum resources and how they should be accessed.

In the QFw-SLURM environment used by this tutorial, the configuration is created at:

```text
/workspace/qfw-container-base/device-access.yaml
```

A single device-access configuration can define multiple quantum resources including resources with different `resource-type` values. The examples below show entries for `IBMQuantumComputeService` and `IBMQuantumSystem`; both entries may coexist in the same `device-access.yaml` file.

### IBM Quantum Compute Service

A device accessed through IBM Quantum Compute Service can be configured with an entry similar to:

```yaml
qpus:
  ibm_fez:
    provider: ibm
    provider-device-id: ibm_fez
    resource-type: IBMQuantumComputeService
    url: "https://quantum.cloud.ibm.com/api/v1"
    service-crn: "<crn>"
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

For this resource type:

- `provider-device-id` identifies the provider-side quantum device
- `url` and `service-crn` identify the IBM Quantum Compute Service used to access the device
- `credential-db` identifies the credential database used for authentication
- `libraries` lists the available QFw interfaces for the device
- `preference` selects the preferred interface
- `execution-owner` identifies the interface responsible for circuit execution
- `caps` maps supported device operations to the interface that provides them


### IBM Quantum System

A device configured as an IBM Quantum System can be described with an entry similar to:

```yaml
qpus:
  test_device:
    provider: ibm
    resource-type: IBMQuantumSystem
    provider-device-id: test_device
    url: "<url>"
    iam-endpoint: "https://iam.test.cloud.ibm.com"
    service-crn: "<crn>"
    s3-endpoint: "<s3-endpoint>"
    s3-bucket: "<s3-bucket-name>"
    s3-region: "<s3-region>"
    job-timeout-seconds: "<timeout>"
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

For this resource type:

- `provider-device-id` identifies the configured IBM Quantum System
- `url` identifies the system endpoint used by QFw/QRMI
- `iam-endpoint` identifies the IAM endpoint used for authentication
- `service-crn` identifies the configured IBM service instance
- `s3-endpoint`, `s3-bucket`, and `s3-region` configure the object-storage location used by the environment
- `job-timeout-seconds` defines the timeout used for quantum job execution
- `credential-db` identifies the credential database used for authentication

In both configurations, QRMI is used by QFw for device access and quantum circuit execution.

The SQD workflow does not depend on a specific IBM resource type. The QFw resource block identifies the configured device and QFw resolves the corresponding resource from the device-access configuration.

The exact configuration may differ for other providers or QFw environments.

## 2. Provider credentials

The device-access configuration references a credential database:

```yaml
credential-db: my-credentials.json
```

A single credential database can contain authentication information for multiple quantum resources. Each device entry should correspond to a device configured in `device-access.yaml`.

For example, the same credential database can contain credentials for both an `IBMQuantumComputeService` resource and an `IBMQuantumSystem` resource:

```json
{
  "users": {
    "root": {
      "enabled": true,
      "devices": {
        "ibm_fez": {
          "enabled": true,
          "api_key": "<REDACTED>"
        },
        "test_device": {
          "enabled": true,
          "api_key": "<REDACTED>",
          "aws_access_key_id": "<REDACTED>",
          "aws_secret_access_key": "<REDACTED>"
        }
      }
    }
  }
}
```

In this example:

- `ibm_fez` contains the API key used to access the `IBMQuantumComputeService` resource.
- `test_device` contains the credentials required by the `IBMQuantumSystem` configuration including the API key and object-storage credentials.

The device names under `devices` should match the corresponding device names configured in `device-access.yaml`.

The exact credential fields depend on the resource type and the QFw environment. Replace the `<REDACTED>` values with valid credentials for the corresponding quantum resource.


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
