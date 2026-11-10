# Run SBD Closed-loop Workflow on Fugaku (qcsc-prefect)

> **Quantum source configuration:** `quantum_source` now names a saved
> `QuantumSourceBlock`; its default is `quantum-source`. The block generator saves
> this Block from the `[source]` table. Set the Block's `config.kind` to `random`,
> `qrmi` or `qfw` to choose the platform, or pass another saved source Block name.
> Older values `random`, `real-device` and `qfw` work only after source Blocks with
> those names are saved. See [Quantum Sources](../howto/howto_use_quantum_source.md)
> and the [SQD source configuration](https://github.com/qiskit-community/qcsc-prefect/tree/main/algorithms/closed_loop/sqd_de#quantum-sampling).


This tutorial walks us through reproducing a Sample-based Quantum Diagonalization (SQD) experiment using the `qcsc-prefect` architecture.
We will run a hybrid quantum-classical workflow using the [SBD](https://github.com/r-ccs-cms/sbd) solver to diagonalize a sparse chemistry Hamiltonian on Fugaku, orchestrated via Prefect.

<img src="../images/img-closed-loop-fugaku.png" alt="sbd" width="90%"/><br>

The goal is to compute the ground state energy of N2-MO state.

## Prerequisites

Before starting, make sure:

- You have completed [Step1 : How to Set Up Python Environment on Fugaku Pre/Post Node](../howto/howto_setup_python_env_fugaku.md).
- You have completed [Step2 : How to Set Up IBM Quantum Access Credentials for Prefect](../howto/howto_setup_prefect_qiskit_fugaku.md).

> [!IMPORTANT]
> Replace `ra00000`, `u12345` and `vol0000x` with your actual group, account name and mount volume.

## 0. What changes from BitCounts?

### What you did in BitCounts (quick recap)
- **(HPC side)** Compile C++ and produce an executable.
- **(Prefect side)** Create blocks and variables.
- **(Flow side)** A Flow loads Blocks and Variables and runs tasks (quantum → HPC → post-process).

### What SBD closed-loop adds
- HPC execution expands from a single binary to full **SBD solver (`diag`) + closed-loop logic**.
- Additional Blocks are required (solver job block, etc.).
- **Deployment (Deploy)** becomes important so that participants can run from the Prefect UI reliably.
- Block creation is automated via script instead of manual UI editing.

### What is `SBDSolverJob` and why it appears here

`SBDSolverJob` is a **workflow-facing facade block** for the SBD domain. It is used so users can select one solver preset from the UI (for example CPU/GPU variants) with:

```
sbd_solver_job/<block_name>
```

Important: this does **not** replace the 3-block architecture.

- `CommandBlock` = WHAT executable to run
- `ExecutionProfileBlock` = HOW to run (MPI/walltime/modules)
- `HPCProfileBlock` = WHERE to run (queue/group/target)
- `SBDSolverJob` = SBD-specific wrapper that stores:
  - references to the three blocks above
  - SBD-specific runtime arguments (`task_comm_size`, `block`, `iteration`, etc.)
  - job file conventions (`root_dir`, `script_filename`)

At runtime, `SBDSolverJob.run(...)` eventually calls `run_job_from_blocks(...)` and delegates actual submission to those three base blocks.

---
## 1. Big picture: Flow / Task / Block / Variable / Deployment

### 1.1 Minimal "where it runs" model
- **Workflow host**: where the Prefect process runs the Flow (runner/worker side)
- **Fugaku**: where `diag` runs via PJM/MPI
- **IBM Quantum**: where quantum sampling runs via Qiskit Runtime

### 1.2 Mapping to Prefect core concepts

#### Flow (end-to-end experiment procedure)
The entire closed-loop SQD experiment is implemented as a single Flow, including iterations, branching, and convergence checks.

#### Tasks (individual steps)
- Execute quantum sampling (IBM Quantum / Qiskit Runtime)
- Subsampling / configuration recovery (Python)
- Davidson diagonalization (PJM job on Fugaku)
- Collect results and store artifacts (Prefect)

#### Blocks (reusable "configuration + credentials")
- **Quantum**: IBM Quantum credentials / runtime configuration
- **HPC**: SBD solver job settings (rscgrp, nodes, executable path, modules, etc.)
- **Command**: command execution settings
- **Execution Profile**: MPI execution settings

#### Variables (runtime parameters)
- Quantum sampler options (shots, etc.) and other run-time knobs are stored as Prefect Variables.

#### Deployment (how the Flow becomes runnable from the UI)
Deployment is the launch entry that tells Prefect:
- which Flow to run,
- under which deployment name,
- and how it should be executed by a serving process.

---
## 2. Tutorial steps
<img src="../images/img-sbd-setup-flow-fugaku.png" alt="sbd setup" width="90%"/><br>

### Step 1. Enter your workflow environment

Connect to the environment where Prefect CLI is configured and Fugaku scheduler commands are available.

<img src="../images/icon-pc.png" alt="pc" width="50"/><br>
```bash
ssh -A <your_account>@<fugaku_login_host>
```

Execute the interact session for Pre/Post Node in the login node.

<img src="../images/icon-login-fugaku.png" alt="login" width="70"/><br>
```bash
srun -p mem2 -n 1 --mem 8G --time=60 --pty bash -i
```

## Step 2. Prepare Prefect and Quantum runtime (Pre/Post Node)

<img src="../images/icon-prepost-fugaku.png" alt="prepost" width="70"/><br>
```bash
cd /path/to/work

git clone git@github.com:qiskit-community/qcsc-prefect.git
cd qcsc-prefect

source ~/venv/prefect/bin/activate
uv pip install prefect-qiskit
export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')
uv pip install --no-deps \
  -e packages/qcsc-prefect-core \
  -e packages/qcsc-prefect-adapters \
  -e packages/qcsc-prefect-blocks \
  -e packages/qcsc-prefect-executor \
  -e packages/qcsc-prefect-dice \
  -e packages/qcsc-prefect-sbd

uv pip install -e algorithms/qcsc_workflow_utility
uv pip install -e algorithms/closed_loop/sqd_de
```


Check installation:

<img src="../images/icon-prepost-fugaku.png" alt="prepost" width="70"/><br>
```bash
uv pip list | grep -E "(qcsc-prefect|sbd|qcsc)"

qcsc-prefect-blocks                 0.1.0
qcsc-prefect-core                   0.1.0
qcsc-prefect-executor               0.1.0
prefect-sbd                        0.1.0
qcsc-workflow-utility              0.1.0
sbd                                0.1.0
```

---

### Step 3. Build the SBD solver on Fugaku Login Node (prepare the HPC executable)

Create a new terminal and login to Fugaku Login Node.
Navigate to native source and build. This procedure can also be executed on the Pre-Post Node.

<img src="../images/icon-login-fugaku.png" alt="login" width="70"/>  or  <img src="../images/icon-prepost-fugaku.png" alt="prepost" width="70"/><br>
```bash
cd /path/to/work/qcsc-prefect/packages/qcsc-prefect-sbd/native
bash ./build_sbd_fugaku.sh
```

Confirm executable:

<img src="../images/icon-login-fugaku.png" alt="login" width="70"/>  or  <img src="../images/icon-prepost-fugaku.png" alt="prepost" width="70"/><br>

```bash
ls -l | grep diag
realpath ./diag
```

Example output:

```text
/path/to/work/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag
```

We will use this path in the next step.

---

### Step 4. Generate Prefect blocks by script (Pre/Post Node)

#### 4.1 Create a job working directory and copy config template

<img src="../images/icon-prepost-fugaku.png" alt="prepost" width="70"/><br>
```bash
cd /path/to/work/qcsc-prefect
mkdir -p /path/to/work/sbd_jobs
```
<img src="../images/icon-prepost-fugaku.png" alt="prepost" width="70"/><br>
```bash
realpath /path/to/work/sbd_jobs
```
Example output:

```text
/volxxxxx/mdt6/data/raxxxxx/uxxxxx/sbd_jobs
```
We will use this path in the next step.

<img src="../images/icon-prepost-fugaku.png" alt="prepost" width="70"/><br>
```
cp algorithms/closed_loop/sqd_de/sbd_blocks.fugaku.example.toml algorithms/closed_loop/sqd_de/sbd_blocks.toml
vim algorithms/closed_loop/sqd_de/sbd_blocks.toml
```

#### 4.2 Edit the configuration file

The copied example already contains the Fugaku tutorial baseline:

- `hpc_target = "fugaku"`
- `launcher = "mpiexec"`
- `num_nodes = 2`
- `mpiprocs = 2`
- `mpi_options = ["-n", "2"]`
- `script_filename = "sbd_solver.pjm"`
- `metrics_artifact_key = "fugaku-sbd-metrics"`

Edit `algorithms/closed_loop/sqd_de/sbd_blocks.toml` and update at least:

- `group`
- `queue`
- `work_dir`
- `sbd_executable`
- `fugaku_gfscache`

| Parameter | Value / Example | Description |
|---|---|---|
| `hpc_target` | `fugaku` | Target scheduler backend |
| `group` | `raXXXXXX` | Fugaku Group ID |
| `queue` | `small` | Fugaku resource group (`rscgrp`) |
| `work_dir` | `/volxxxxx/mdt6/data/raxxxxx/uxxxxx/sbd_jobs` | Job working directory |
| `sbd_executable` | `/volxxxxx/mdt6/data/raxxxxx/uxxxxx/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag` | Absolute path to executable |
| `num_nodes` | `2` | Number of allocated nodes for this tutorial |
| `launcher` | `mpiexec` | MPI launcher |
| `mpiprocs` | `2` | Number of MPI processes |
| `mpi_options` | `["-n", "2"]` | MPI options for launcher |
| `shots` | `50000` | Prefect variable value for SQD sampling (`sqd_options.params.shots`) |
| `fugaku_gfscache` | `/vol0002` | Optional GFSCACHE setting |
| `fugaku_spack_modules` | site-specific list | Optional Spack modules |

#### 4.3 Run block creation script

<img src="../images/icon-prepost-fugaku.png" alt="prepost" width="70"/><br>
```bash
python algorithms/closed_loop/sqd_de/create_blocks.py \
  --config algorithms/closed_loop/sqd_de/sbd_blocks.toml \
  --hpc-target fugaku
```

This creates the following blocks (default names):

- **CommandBlock**: `cmd-sbd-diag`
- **ExecutionProfileBlock**: `exec-sbd-fugaku`
- **HPCProfileBlock**: `hpc-fugaku-sbd`
- **SBD Solver Job**: `davidson-solver`
- **Prefect Variable**: `sqd_options`

`davidson-solver` is the block you later pass as `Solver Block Ref` in the UI.
Internally, this block points to:
- `cmd-sbd-diag`
- `exec-sbd-fugaku`
- `hpc-fugaku-sbd`

---

### Step 5. Deploy SBD workflow (Pre/Post Node)

**Deploy = register a Flow as a runnable entry point (Deployment) so it can be started from the Prefect UI/CLI by name.**

Set Ray runtime parameter before deploy:

<img src="../images/icon-prepost-fugaku.png" alt="prepost" width="70"/><br>
```bash
export PREFECT_RAY_NUM_CPUS=2
```

> [!NOTE]
> `SBD_TASK_RUNNER` is `ray` by default.
> If Ray is unstable in your environment, use `export SBD_TASK_RUNNER=concurrent` for troubleshooting.

Deploy:

<img src="../images/icon-prepost-fugaku.png" alt="prepost" width="70"/><br>
```bash
sbd-deploy
```

### Step 6. Provide workflow parameters

In the Prefect console, click **Run** → **Custom run** and set at least:

| Field | Value / Example |
|---|---|
| FCIDump File | `/path/to/work/qcsc-prefect/algorithms/closed_loop/sqd_de/data/fcidump_N2_MO.txt` |
| SQD Subspace Dimension (Optional) | `1000000` |
| Differential Evolution Iterations (Optional) | `1` (start small for testing) |
| Quantum Source Block (Optional) | `quantum-source` |
| Random Seed (Optional) | `24` |
| Solver Block Ref | `sbd_solver_job/davidson-solver` |

> [!CAUTION]
> The sample `circ_params.sabre_layout_trials` values used in this tutorial are intentionally kept low on Fugaku so the workflow does not spend a very long time in SABRE layout search before submitting to IBM Quantum.
> Please treat them as safe starting points for this tutorial, not universal best values.
> Adjust `sabre_layout_trials` according to your own backend, qubit count, and circuit specification.

`Solver Block Ref` means: "which `SBDSolverJob` preset should this run use?"
- It is a stable entry point for users.
- HPC details are still resolved through the underlying 3 blocks.

<img src="../images/img-sbd-workflow-paramaters-small.png" alt="params" width="90%"/><br>

### Step 7. Execute the workflow

Click **Start Now** → **Submit**.

<img src="../images/img-sbd-flow-run.png" alt="flow run" width="90%"/><br>

After the run completes, check the `sqd-telemetry` artifact. It should contain intermediate energies.

<img src="../images/img-sqd-telemetry.png" alt="telemetry" width="90%"/><br>

---

## 3. What happens when you "Submit" from the Prefect UI?

### 3.1 What the UI actually creates
When you choose **Run → Custom run → Submit**, the Prefect server creates a **Flow Run request** for a specific Deployment, with the parameters you provided.

### 3.2 What actually executes the Flow
The process started by `sbd-deploy` is the **serving process**. It:
1. polls the Prefect server for new Flow Runs,
2. when it finds one, it executes the Flow on the host where `sbd-deploy` is running.

> If the serving process stops, the UI can still create Flow Runs, but there is no active runner to pick them up.

### 3.3 Confirm deployment information

1) List deployments

<img src="../images/icon-prepost-fugaku.png" alt="prepost" width="70"/><br>

```bash
prefect deployment ls
```

2) Inspect deployment

<img src="../images/icon-prepost-fugaku.png" alt="prepost" width="70"/><br>

```bash
prefect deployment inspect 'riken-sqd-de/riken_sqd_de'
```

3) Locate Flow definition

The flow is defined in `algorithms/closed_loop/sqd_de/sbd/main.py`.

### 3.4 What happens in this architecture

1. `walker_sqd` loads `SBDSolverJob` by name.
2. `SBDSolverJob.run(...)` prepares `fcidump.txt` and `AlphaDets.bin`.
3. Job is submitted by `run_job_from_blocks(...)` using:
   - `CommandBlock`
   - `ExecutionProfileBlock`
   - `HPCProfileBlock`
4. SBD output files (`davidson_energy.txt`, `occ_a.txt`, `occ_b.txt`, `carryover.bin`) are parsed.
5. Flow stores telemetry artifact (`sqd-telemetry`).

This keeps workflow code stable while HPC settings are controlled by block instances.

---

*END OF TUTORIAL*
