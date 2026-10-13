# How to Run and Access a Prefect Server on a Slurm Login Node

This guide describes how to run a Prefect server on a Slurm login node and access the Prefect UI from your local computer using SSH port forwarding.

It also includes notes for running Prefect inside the `QFw-SLURM` container environment.

This setup is intended for small-scale development, testing and tutorial environments. For shared or production environments, use the Prefect deployment appropriate for your infrastructure.

## Prerequisites

Before starting, make sure that:

- You have access to a Slurm cluster.
- You can SSH to the Slurm login node.
- Prefect is installed in the Python environment used for the workflow.

For the QFw-SLURM environment, also make sure that:

- The QFw-SLURM container environment is running.
- QFw and QRMI are installed and available in the workflow environment.
- The target quantum device and provider credentials are configured for QFw.

## Step 1. Log in to the Slurm login node

From your local computer, connect to the Slurm login node:

```bash
ssh <username>@<login-node>
```

### Generic Slurm environment

Activate the Python environment containing Prefect:

```bash
cd /path/to/qcsc-prefect
source .venv/bin/activate
```

### QFw-SLURM environment

In the QFw-SLURM setup, run the Prefect server on the `slurmctld` container.

Enter the container using the mechanism appropriate for your QFw-SLURM deployment, then activate the QFw environment:

```bash
source /opt/openqse/qfw/bin/qfw-activate --venv /opt/openqse/qfw-venv
```

The QFw environment contains the QFw and QRMI libraries needed by the quantum execution path.

## Step 2. Start the Prefect server

Start the Prefect server:

```bash
prefect server start --host 0.0.0.0 --background
```

The Prefect server listens on port `4200`.

Using `--host 0.0.0.0` allows the server to be reached through port forwarding from outside the login node or container.

On a fresh Prefect installation, the active profile may be `ephemeral`.
When starting the server for the first time, Prefect may prompt you to either
update the current profile or create a new profile.
You can either continue using the ephemeral profile or create a dedicated profile, such as `sqd` and switch to it.

For example, on a fresh installation Prefect may prompt you to create a new profile:

```text
(env) [root@login-node]# prefect server start --host 0.0.0.0 --background

Prefect collects anonymous usage data to improve the product.
To opt out: set PREFECT_SERVER_ANALYTICS_ENABLED=false on the server, or DO_NOT_TRACK=1 in the client.
Learn more: https://docs.prefect.io/concepts/telemetry

The `PREFECT_API_URL` setting for your current profile doesn't match the address of the server that's running. You need to set it to
communicate with the server.
? How would you like to proceed? [Use arrows to move; enter to select]
> Create a new profile with `PREFECT_API_URL` set and switch to it
  Set `PREFECT_API_URL` in the current profile: 'ephemeral'
? Enter a new profile name: sqd
Switched to new profile 'sqd'

 ___ ___ ___ ___ ___ ___ _____
| _ \ _ \ __| __| __/ __|_   _|
|  _/   / _|| _|| _| (__  | |
|_| |_|_\___|_| |___\___| |_|

Configure Prefect to communicate with the server with:

    prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api

View the API reference documentation at http://0.0.0.0:4200/docs

Check out the dashboard at http://0.0.0.0:4200



The Prefect server is running in the background. Run `prefect server stop` to stop it.

```
## Step 3. Configure the Prefect client

Configure the Prefect client in the same environment to connect to the local Prefect server:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

Verify the active Prefect configuration:

```bash
prefect profile inspect
```

The active profile should show:

```text
PREFECT_API_URL='http://127.0.0.1:4200/api'
```

Verify that the server is accessible:

```bash
prefect server status
```

You can also verify the API directly:

```bash
curl http://127.0.0.1:4200/api/health
```

A successful response indicates that the Prefect server is running and accessible from the login node or container.

## Step 4. Access the Prefect UI from your local computer

The Prefect UI is served on port `4200`.

### Generic Slurm environment

If the Prefect server is running directly on the Slurm login node, open a new terminal on your local computer and run:

```bash
ssh -L 4200:127.0.0.1:4200 <username>@<login-node>
```

Keep this SSH connection open while using the Prefect UI.

Then open:

```text
http://127.0.0.1:4200
```

in your local web browser.

### QFw-SLURM environment

If the QFw-SLURM environment is running locally, for example on your laptop and port `4200` is exposed from the `slurmctld` container, open the Prefect UI directly in your local web browser:

```text
http://127.0.0.1:4200
```

You can verify access first with:

```bash
curl http://127.0.0.1:4200/api/health
```

If the QFw-SLURM environment is running on a remote machine, create an SSH tunnel from your local computer to port `4200` on the remote host:

```bash
ssh -L 4200:127.0.0.1:4200 <username>@<remote-host>
```

Keep the SSH connection open while using the Prefect UI.

Then open:

```text
http://127.0.0.1:4200
```

in your local web browser.

## Step 5. Verify Prefect from the workflow environment

Before creating Prefect blocks or running the SQD workflow, verify that the active environment is connected to the expected Prefect server:

```bash
prefect profile inspect
```

and:

```bash
curl http://127.0.0.1:4200/api/health
```

For QFw-SLURM, these commands should be run inside the `slurmctld` container after activating the QFw environment.


## Step 6. Stop the Prefect server

When the server is no longer needed, stop it in the environment where it was started:

```bash
prefect server stop
```

You can start it again later with:

```bash
prefect server start --host 0.0.0.0 --background
```
