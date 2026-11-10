# How to Set Up Python Environment on Fugaku Pre/Post Node

This guide explains how to set up a Python virtual environment for workflow execution on Fugaku.

## Instructions

> [!IMPORTANT]
> - Replace `u12345` with your actual account name.

### Step 1: Log in to Fugaku and execute the interact session for Pre/Post Node

Execute the interact session for Pre/Post Node in the login node.

<img src="../images/icon-login-fugaku.png" alt="pc" width="70"/><br>
```bash
srun -p mem2 -n 1 --mem 4G --time=60 --pty bash -i
```

### Step 2: Load Spack Module and Install UV

Load Python3.12 and sqlite and Install `uv`.

<img src="../images/icon-prepost-fugaku.png" alt="pc" width="70"/><br>
```bash
. /vol0004/apps/oss/spack/share/spack/setup-env.sh
spack load sqlite@3.46.0
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 3: Create a Python Virtual Environment

Use `uv` to create a virtual environment with Python 3.12:

<img src="../images/icon-prepost-fugaku.png" alt="pc" width="70"/><br>
```bash
uv venv ~/venv/qcsc-host -p 3.12
```

This will install Python 3.12 and set up a new environment named `prefect`.

### Step 4: Activate the Environment

Activate the virtual environment and point OpenSSL to the CA bundle installed by `certifi`:

<img src="../images/icon-prepost-fugaku.png" alt="pc" width="70"/><br>
```bash
source ~/venv/qcsc-host/bin/activate
```
