# How to Set Up Python Environment on the MDX Workflow Client

This guide explains how to set up a Python virtual environment for workflow execution on the mdx platform.

## Prerequisites

Before you begin, ensure the following:

- You have an MDX workflow client account.
- SSH authentication to the workflow client is configured.

## Instructions

> [!IMPORTANT]
> - Replace `z12345` with your actual account name.
> - If you encounter an error indicating insufficient space in the `~` (home) directory on mdx, use the `/large/z12345` directory instead.


### Step 1: Log in to MDX Workflow Client

Connect to MDX workflow client:

<img src="../images/icon-pc.png" alt="pc" width="50"/><br>
```bash
ssh -A z12345@mdx-workflow.example.org
```

### Step 2: Install `uv` Package Manager

Run the following command to install `uv`:

<img src="../images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 3: Create a Python Virtual Environment

Use `uv` to create a virtual environment with Python 3.12:

<img src="../images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
uv venv ~/venv/qcsc-host -p 3.12
```

This will install Python 3.12 and set up a new environment named `prefect`.

### Step 4: Activate the Environment and Install Prefect.

Activate the virtual environment:

<img src="../images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
source ~/venv/qcsc-host/bin/activate
uv pip install prefect
```

Your Python environment is now ready for workflow development.

### Step 5: Add github.com to SSH known_hosts

If this is the first time to connect with `github.com` on this machine, make sure you added GitHub's SSH host key:

<img src="../images/icon-mdx.png" alt="mdx" width="50"/><br>
```bash
ssh -T git@github.com
```

You should see the output like:

```text
Hi ...! You've successfully authenticated, but GitHub does not provide shell access.
```

---
*END OF GUIDE*
