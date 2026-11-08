# Size the SQD diagonalization for your HPC resources

The open-loop SQD workflow (`algorithms/open_loop/sqd`) diagonalizes each subspace with one of
three solvers: [Fulqrum](https://github.com/Qiskit/fulqrum), [SBD](https://github.com/r-ccs-cms/sbd),
or [DICE](https://github.com/caleb-johnson/Dice). Each solver splits the work across processes,
threads, and GPUs in its own way. This tutorial links the hardware of a node to the memory that a
subspace needs, and then to the values in `sqd_blocks.toml` that request the resources.

In this tutorial, you learn how to:

- Count the nodes that one SQD iteration occupies.
- Estimate the size of a subspace and the memory that each solver needs for it.
- Read how the execution profile fields become Slurm, PBS, or PJM requests.
- Choose the MPI ranks, OpenMP threads, GPUs, and solver grid for Fugaku, ROQUO, and a
  generic Slurm cluster.
- Confirm the estimate with a small run.

> [!NOTE]
> Every memory figure in this tutorial is an estimate from the data structures of the solvers.
> Confirm it with a small run on your system before you request a large allocation, as described
> in [Check the run](#check-the-run). The Davidson settings (`block`, `iteration`, `tolerance`),
> carryover, LUCJ, transpilation, circuit parameters, and shots are covered in a later document.

## Contents

- [Prerequisites](#prerequisites)
- [Count the nodes that one SQD iteration uses](#count-the-nodes-that-one-sqd-iteration-uses)
- [Estimate the problem size and memory](#estimate-the-problem-size-and-memory)
- [Map block fields to scheduler requests](#map-block-fields-to-scheduler-requests)
- Size each solver:
  - [Fulqrum](#size-a-fulqrum-solve)
  - [SBD](#size-an-sbd-solve)
  - [DICE](#size-a-dice-solve)
- [Choose settings for each system](#choose-settings-for-each-system):
  - [Fugaku](#fugaku)
  - [ROQUO](#roquo)
  - [Generic Slurm cluster](#generic-slurm-cluster)
- [Check the run](#check-the-run)
- [References](#references)

## Prerequisites

- A working SQD setup on your system, as in
  [Set up qcsc-prefect on a Slurm cluster](https://quantum.cloud.ibm.com/docs/en/guides/set-up-qcsc-prefect-slurm)
  and the tutorial
  [Manage a quantum-centric supercomputing workflow with qcsc-prefect](https://quantum.cloud.ibm.com/docs/en/tutorials/manage-qcsc-workflow-with-qcsc-prefect).
- A solver executable built on the target system: `qcsc-fulqrum-solve`, SBD `diag` or `diag-gpu`,
  or DICE `Dice`.
- The variables `QCSC_PREFECT` (the qcsc-prefect checkout) and `QCSC_WORK_DIR` (the shared work
  directory) from the setup guide. The shell commands of this tutorial use them.
  `sqd_blocks.toml` does not expand environment variables, so its snippets write the checkout
  as `~/qcsc-prefect`. Replace it with the value of `QCSC_PREFECT` if your checkout is elsewhere.
- Your node specifications: cores, memory, and GPUs per node. On Slurm, run
  `sinfo -o "%P %n %c %m %G"` to list the CPUs, the memory in MB, and the GPUs of each node.

## Count the nodes that one SQD iteration uses

Each SQD iteration draws `num_batches` subspaces (default 4) and solves them at the same time.
Each batch is one scheduler job that requests the resources of the execution profile block. The
nodes that one iteration holds at the same time are therefore:

```text
nodes per iteration = num_batches x num_nodes
```

The iterations run one after another, so this product is the peak. When all sampled strings fit
in `subspace_dim`, the workflow draws one batch only. The node limit of a queue applies to each
batch job, not to the product, but the product counts against your share of the queue.

## Estimate the problem size and memory

### Subspace dimension

For a closed-shell run, each batch keeps at most `int(sqrt(subspace_dim))` strings per spin, and
the subspace is the product of the alpha and beta lists:

```text
n_a = n_b <= int(sqrt(subspace_dim))
D   = n_a x n_b <= subspace_dim
```

Two limits can make `D` smaller. The first is the number of distinct strings that the samples
contain. The second is the number of strings that exist, `C(norb, n_alpha)`. Open-shell batches
size each spin separately, and the SBD conversion keeps `n_a x n_b <= subspace_dim`.

One amplitude vector holds `D` float64 values, so it takes `8 B x D`. The following table shows
two examples: 36 orbitals with 18 alpha and 18 beta electrons, and 16 orbitals with 8 and 8.

| `subspace_dim` | Strings per spin (norb 36) | `D` (norb 36) | Strings per spin (norb 16) | `D` (norb 16) | One vector, `8 B x D` |
| --- | --- | --- | --- | --- | --- |
| 1e6 | 1,000 | 1.0e6 | 1,000 | 1.0e6 | 8 MB |
| 1e8 | 10,000 | 1.0e8 | 10,000 | 1.0e8 | 0.8 GB |
| 9e8 | 30,000 | 9.0e8 | FCI | FCI | 7.2 GB (norb 36) |

With 16 orbitals, only C(16, 8) = 12,870 strings exist per spin. At `subspace_dim = 9e8` the
subspace is therefore the full configuration interaction (FCI) space.

### Memory per determinant

Every solver keeps several vectors of the length of its share of `D`. The following estimates
count those vectors. They leave out smaller tables, such as SBD's excitation helpers or
Fulqrum's operator tables, so keep at least 20 to 30 percent of the memory free.

| Solver and mode | Memory per determinant that a process holds (estimate) | Example |
| --- | --- | --- |
| Fulqrum `cpu`, `matrix_free` | about `(max_basis_size + 4) x 8 B` | 20 basis vectors: about 190 B |
| Fulqrum `cpu`, `csr` | `matrix_free` plus `nnz x 16 B` (`nnz x 12 B` with fewer than 2^31 nonzeros) | Measure `nnz` |
| Fulqrum `gpu-mpi`, on the GPU | `(2 x max_basis_size + 5) x 8 B`, plus 1 GiB scratch | `max_basis_size = 20`: 360 B |
| SBD, CPU (`diag`) | `(2 x block + 6) x 8 B` | `block = 10`: 208 B |
| SBD, GPU (`diag-gpu`), on the GPU | `(2 x block + 6) x 8 B + 8 B x ceil(2 x norb / 20)` | `block = 10`, norb 36: 240 B |
| DICE | about 37 B per determinant for each rank, plus about 670 B per determinant for the job | See [Size a DICE solve](#size-a-dice-solve) |

Here `block` is the Davidson block of SBD and `max_basis_size` is the eigensolver search space of
Fulqrum. Both appear only as factors in this tutorial. When `max_basis_size` is unset, the
eigensolver picks a value (20 for `scipy`), so set it when you size a run with this table.

## Map block fields to scheduler requests

`sqd-create-blocks` merges the system profile (`sqd_common/profiles/<profile>.toml`), its
`[<solver>]` table, your top-level keys, and your `[<solver>]` table, in that order. The result
fills the execution profile block and the HPC profile block. The following table shows where
each field goes in the job script.

| Field | Slurm (`sbatch`) | Miyabi (PBS) | Fugaku (PJM) |
| --- | --- | --- | --- |
| `num_nodes` | `--nodes` | `select=<num_nodes>` | `-L "node=<num_nodes>"` |
| `mpiprocs` | `--ntasks-per-node` | `:mpiprocs=` | Not rendered |
| `ompthreads` | `--cpus-per-task` | `:ompthreads=` | Not rendered |
| `gpus_per_node` | `--gpus-per-node` | Not rendered, at most 1 | Not allowed |
| `slurm_memory` | `--mem` (per node) | Not used | Not used |
| `fugaku_mpi_options_for_pjm` | Not used | Not used | `--mpi "<option>"` |
| `queue`, `project` | `--partition`, `--account` | `-q`, `-W group_list` | `rscgrp=`, `-g` |
| `launcher`, `mpi_options` | `<launcher> <mpi_options> <exe>` | Same | Same |

The solver blocks of Fulqrum and SBD also export `OMP_NUM_THREADS` with the value of
`ompthreads`, unless `[environments]` sets it. The DICE blocks do not.

The rank count of a job comes from different places on each scheduler:

- **Slurm:** `srun` starts `num_nodes x mpiprocs` tasks from the allocation. Do not add `-n` to
  `mpi_options`.
- **Miyabi:** `mpiexec.hydra -np` or `mpirun -n` in `mpi_options` sets the rank count. Keep it
  equal to `num_nodes x mpiprocs`.
- **Fugaku:** `mpiexec -n` in `mpi_options` sets the rank count, and
  `fugaku_mpi_options_for_pjm = ["max-proc-per-node=<ranks per node>"]` sets the ranks per node.
  Keep `-n` equal to `num_nodes x ranks per node`. PJM does not set the threads, so the
  `OMP_NUM_THREADS` export of the solver block (or your `[environments]`) sets them.

> [!NOTE]
> `sqd-create-blocks` stops when the count of `-n` or `-np` in `mpi_options` differs from
> `num_nodes x mpiprocs`, or when `max-proc-per-node` differs from `mpiprocs` on Fugaku. For
> DICE, `mpi_options` defaults to `num_nodes x mpiprocs` ranks. The SBD tables of the bundled
> profiles set fixed counts, `-np 4` on Miyabi and `-n 2` on Fugaku, so set `mpi_options` again
> when you change `num_nodes` or `mpiprocs` for SBD.

On every system, keep `ranks per node x ompthreads` at or below the physical cores of a node.

## Size a Fulqrum solve

The `backend` key of the `[fulqrum]` table selects how Fulqrum uses the allocation.

| `backend` | Processes | Memory that scales with `D` | Scheduler request |
| --- | --- | --- | --- |
| `cpu` | 1 process, OpenMP threads | Host: full vectors, plus the CSR matrix with `hamiltonian_format = "csr"` | `launcher = "single"`, `num_nodes = 1`, `mpiprocs = 1`, `ompthreads` = cores |
| `gpu` | 1 process that uses every GPU it sees | Host: full vectors. GPUs: the operator | `launcher = "single"`, `num_nodes = 1`, `mpiprocs = 1`, `gpus_per_node` = GPUs of the node |
| `gpu-mpi` | 1 MPI rank per GPU, on any number of nodes | Each GPU: one tile of every vector | MPI launcher, `mpiprocs = gpus_per_node` |

Only `gpu-mpi` splits the vectors across MPI ranks. The `cpu` and `gpu` backends run one process
on one node, so the full vectors must fit in the memory of that node. A large subspace, such as
`D = 9e8`, needs the vectors split across several processes with MPI: on CPU nodes, use SBD.
MPI support for the Fulqrum `cpu` backend is planned in a coming Fulqrum release.

With `backend = "cpu"`, choose between two Hamiltonian formats:

- `hamiltonian_format = "csr"` (default) builds an explicit sparse matrix. It is faster but
  needs `nnz x 16 B` more memory (`nnz x 12 B` with 32-bit indices). During the conversion, the
  intermediate format and the CSR copy exist at the same time, so plan for about twice that
  amount. Fulqrum stops with an error when the copy does not fit in the available memory.
- `hamiltonian_format = "matrix_free"` applies the Hamiltonian without storing it, which needs
  only the vectors.

You cannot compute `nnz` in advance. Run a smaller `subspace_dim` with `csr`, read `nnz` in the
`metadata` of `result.json` in the job directory, and scale it at least linearly with `D`.

With `backend = "gpu-mpi"`, the ranks form a grid `process_grid = [p_alpha, p_beta]`. Each rank
owns a tile of `ceil(n_a / p_alpha) x ceil(n_b / p_beta)` determinants and uses GPU
`local rank % GPU count`. The product `p_alpha x p_beta` must equal the rank count. Block creation
checks it against `num_nodes x mpiprocs`, and the solver checks it again at run time. Without
`process_grid`, Fulqrum chooses the grid. `sqd-create-blocks` sets `mpiprocs` to `gpus_per_node`
for this backend. Before it solves, the distributed operator compares its predicted GPU memory
with 85 percent of the free memory and stops on every rank when it does not fit.

The per-GPU estimate for `gpu-mpi` follows from the tile size:

```text
GPU memory per rank ~ (2 x max_basis_size + 5) x 8 B x n_local + 1 GiB + operator tables
n_local = ceil(n_a / p_alpha) x ceil(n_b / p_beta)
```

For `D = 9e8` and `max_basis_size = 20`, the vectors need about 324 GB in total. Four GPUs with a
2 x 2 grid need about 81 GB each, and six GPUs with a 2 x 3 grid need about 54 GB each.

> [!NOTE]
> The GPU profiles use the MPI launcher for `gpu-mpi`. To use `backend = "gpu"` with a GPU
> profile, also set `launcher = "single"` and `mpiprocs = 1`, so that one process starts.

## Size an SBD solve

SBD divides the MPI ranks of a job into four communicators. You set three of their sizes. The
fourth, `h_comm`, takes the ranks that are left.

| Communicator | Size | What it splits | Vector tile on each rank |
| --- | --- | --- | --- |
| `b_comm` | `adet_comm_size x bdet_comm_size` | The wavefunction: alpha strings into `adet_comm_size` parts and beta strings into `bdet_comm_size` parts | Own tile, `(n_a / adet_comm_size) x (n_b / bdet_comm_size)` |
| `t_comm` | `task_comm_size` | The list of tile-pair tasks of the Hamiltonian product | Copy of the tile |
| `h_comm` | `ranks / (task x adet x bdet)` | The bra rows of each task, round robin | Copy of the tile |

The product `task_comm_size x adet_comm_size x bdet_comm_size` must divide the total rank count.
SBD stops at startup otherwise, and `sqd-create-blocks` checks it against
`num_nodes x mpiprocs` before it saves the blocks. The usual choice is a product equal to the
rank count, so that `h_comm` has one rank and every rank owns its own tile. Ranks in `t_comm` and
`h_comm` share work but hold copies of the same tile, so they multiply memory instead of
dividing it.

### Estimate the memory per rank

The CPU Davidson solver holds `2 x block` subspace vectors and about six more work vectors, all of
the tile size:

```text
memory per rank ~ (2 x block + 6) x 8 B x D / (adet_comm_size x bdet_comm_size) + helper tables
```

The helper tables list the excitations between strings. They grow with the strings per spin in a
tile, not with `D`, but you can only measure them. With `task_comm_size = 1` and the product
equal to the rank count, the memory of a node depends only on the node count:

```text
memory per node ~ (2 x block + 6) x 8 B x D / num_nodes
```

The ranks per node then decide the parallelism, not the memory.

### Choose the grid

1. Estimate the total vector memory, `(2 x block + 6) x 8 B x D`. With `block = 10` and
   `D = 9e8`, this is 187 GB.
2. Divide it by the memory that you allow per node, for example 70 percent of the node memory.
   Round up to get the smallest `num_nodes`. On a node with 128 GiB, 187 GB / 86 GB gives 3, and
   4 nodes leave headroom.
3. Choose the ranks per node so that `mpiprocs x ompthreads` fits the cores, for example 16 ranks
   with 4 threads on 64 cores.
4. Split the rank count into `adet_comm_size x bdet_comm_size` as close to a square as possible.
   4 nodes x 16 ranks = 64 ranks = 8 x 8, which gives a tile of about 1.4e7 determinants and
   about 2.9 GB of vectors per rank.
5. Keep `task_comm_size = 1` unless you have free ranks and free memory.

### Run SBD on GPUs

With `solver_mode = "gpu"` (set by the `slurm-gpu` and `miyabi-gpu` profiles), `sbd_executable`
points to `diag-gpu`, and the blocks run one rank per GPU. `sqd-create-blocks` sets `mpiprocs` to
`gpus_per_node`. The same grid rule applies, with the GPU count in place of the rank count.

- **Binding:** `diag-gpu` selects GPU `node-local rank % GPU count` from `SLURM_LOCALID` or
  `OMPI_COMM_WORLD_LOCAL_RANK`, before `MPI_Init`. Every rank must see all GPUs of its node, so do
  not add `--gpu-bind` to `mpi_options`.
- **Threads:** GPU builds abort at startup when `OMP_NUM_THREADS` is 1, because the helper tables
  are built on the CPU. Give each rank several cores, for example the cores of a node divided by
  its GPUs.
- **Memory:** besides the vectors, each GPU holds the determinants of its tile, at
  `8 B x ceil(2 x norb / 20)` per determinant. The GPU estimate is therefore 240 B per
  determinant for `block = 10` and 36 orbitals.
- **Several ranks:** `diag-gpu` passes GPU buffers to MPI, which needs a CUDA-aware MPI. Otherwise
  add SBD's host-staging define, `-DSBD_NON_CUDA_AWARE_MPI`, to `CCFLAGS` of `build_sbd_gpu.sh`.

## Size a DICE solve

DICE runs `mpiprocs` ranks on each node and supports Miyabi and Fugaku. The bundled profiles use
one OpenMP thread per rank. Each DICE rank holds data proportional to `D`, so more nodes do not
reduce the memory of a rank. Fewer ranks per node reduce the memory of a node.

A sweep on Miyabi-C (`algorithms/open_loop/sqd/perf/tuning_mpi_parameters_with_dice.ipynb`) fits
the total memory of a job, summed over its nodes, to a bilinear model:

```text
F [GB] ~ -0.63 + 0.054 u + 6.70 v + 0.374 u v
u = total MPI ranks (num_nodes x mpiprocs), v = D / 1e7
```

Read it as two parts:

- Each rank needs about `0.054 + 0.374 v` GB, which is about 37 B per determinant. At `D = 1e8`
  this is 3.8 GB per rank.
- The job as a whole needs about `6.70 v` GB more, about 670 B per determinant. At `D = 1e8` this is 67 GB.
  The node of rank 0 carries more of it, so keep a margin of about 20 percent.

The smallest node count that fits follows from the memory `M` that you allow per node:

```text
num_nodes >= (6.70 v - 0.63) / (M - mpiprocs x (0.054 + 0.374 v))
```

The same sweep found that, at `D = 1e8`, the wall time stops improving at about 10 nodes and that
10 to 20 ranks per node use the cores best. Measured memory came in about 10 GB under the model.
The model comes from one molecule on Miyabi-C with Intel MPI, so treat other systems and larger
`D` as extrapolations.

## Choose settings for each system

The following sections give a starting layout for `D = 1e8` and `D = 9e8`, with `block = 10` for
SBD and `max_basis_size = 20` for Fulqrum. The node counts are the smallest that fit the estimates
of the preceding sections. Add nodes for speed after a small run confirms the memory.

### Fugaku

Each A64FX node has 48 compute cores in 4 core memory groups (CMGs) of 12, and 32 GiB of HBM2.
Jobs get less than the full 32 GiB, so plan with about 20 GB per node. Fugaku has no GPUs.

| Solver | Ranks per node x threads | `D = 1e8` | `D = 9e8` |
| --- | --- | --- | --- |
| Fulqrum `cpu` | 1 x 48 | `matrix_free` needs about 19 GB on one node, which is tight | Does not fit |
| SBD CPU | 4 x 12 (one rank per CMG) or 1 x 48 | 2 nodes, grid 2 x 4 (10.4 GB per node) | 10 nodes minimum, 16 nodes with grid 8 x 8 (11.7 GB per node) |
| DICE | 1 x 1 or 2 x 1 | 5 nodes (1 rank each) or 6 nodes (2 ranks each), extrapolated | Does not fit: one rank needs about 34 GB |

Fulqrum and DICE do not fit at `D = 9e8` on Fugaku for the following reasons:

- **Fulqrum:** a subspace of this size needs MPI to split the vectors across nodes. The only
  Fulqrum backend with MPI is `gpu-mpi`, and Fugaku has no GPUs. The `cpu` backend runs one
  process on one node and holds every vector of the full length `D`. At `D = 9e8`,
  `matrix_free` needs about `(20 + 4) x 8 B x 9e8 = 173 GB`, more than five times the 32 GiB
  of a node. Adding nodes does not help until the `cpu` backend supports MPI, which is planned
  in a coming Fulqrum release.
- **DICE:** every DICE rank holds its own data of the full length `D`, about 37 B per
  determinant. At `D = 9e8`, one rank needs about `0.054 + 0.374 x 90 = 34 GB`, which is more
  than the 32 GiB of a node before the job memory is added. More nodes add ranks, and every
  rank needs the same 34 GB, so no node count fits.

SBD is the only one of the three solvers that divides the vectors across ranks on CPU nodes,
so it is the solver for large subspaces on Fugaku.

```toml
solver = "sbd"
profile = "fugaku"
queue = "small"                      # Your resource group
project = "hpXXXXXX"
work_dir = "/vol0004/hpXXXXXX/sqd"

[sbd]
sbd_executable = "~/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag"
num_nodes = 16
mpiprocs = 4                         # Ranks per node; PJM does not render it
ompthreads = 12                      # Exported as OMP_NUM_THREADS
mpi_options = ["-n", "64"]           # num_nodes x ranks per node
fugaku_mpi_options_for_pjm = ["max-proc-per-node=4"]
adet_comm_size = 8
bdet_comm_size = 8
```

For DICE on Fugaku, set the threads in `[environments]`, because the DICE blocks do not export
`OMP_NUM_THREADS`:

```toml
[environments]
OMP_NUM_THREADS = "1"
```

### ROQUO

ROQUO at RIKEN R-CCS has 135 NVIDIA GB200 NVL4 nodes. Each node has two Grace CPUs with 144 Arm
cores in total and four Blackwell GPUs. The scheduler is Slurm and the MPI is HPC-X (Open MPI with
PMIx), so `srun --mpi=pmix` launches the ranks. A full node is 1600000M of memory in Slurm.
Each GPU has about 186 GB of HBM3E. Check the value on your nodes with `nvidia-smi`. The login
nodes are x86_64 and the compute nodes are aarch64, so build the solvers on a compute node.

ROQUO has no bundled profile. Use `slurm-gpu`, which already sets `srun`, `--mpi=pmix`, 4 GPUs,
and one rank per GPU, and override the threads, the memory, and the partition. Plan with about
150 GB per GPU.

| Solver | Ranks per node x threads | `D = 1e8` | `D = 9e8` |
| --- | --- | --- | --- |
| Fulqrum `gpu-mpi` | 4 x 36 | 1 node, `process_grid = [2, 2]` (about 9 GB per GPU) | 1 node, `process_grid = [2, 2]` (about 81 GB per GPU) |
| SBD GPU | 4 x 36 | 1 node, grid 2 x 2 (about 6 GB per GPU) | 1 node, grid 2 x 2 (about 54 GB per GPU) |

```toml
solver = "sbd"
profile = "slurm-gpu"
queue = "roquo"                      # Partition
project = "YOUR_ACCOUNT"
work_dir = "/home/YOUR_USER/work/sqd"  # QCSC_WORK_DIR, shared with the compute nodes
ompthreads = 36                      # 144 cores / 4 ranks
slurm_memory = "1600000M"            # #SBATCH --mem, a full node
modules = []                         # The modules the solver was built with

[sbd]
sbd_executable = "~/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag-gpu"
num_nodes = 1
adet_comm_size = 2
bdet_comm_size = 2
```

For Fulqrum on ROQUO, build the GPU environment on a compute node:

```bash
# Run in a terminal on a ROQUO compute node.
cd "$QCSC_PREFECT"
FULQRUM_WITH_GPU=1 FQ_CUDA_ARCH=sm_100 bash packages/qcsc-prefect-fulqrum/native/build_fulqrum.sh \
  packages/qcsc-prefect-fulqrum/native/fulqrum-gpu
```

Then replace the `[sbd]` table with a `[fulqrum]` table:

```toml
[fulqrum]
fulqrum_executable = "~/qcsc-prefect/packages/qcsc-prefect-fulqrum/native/fulqrum-gpu/bin/qcsc-fulqrum-solve"
num_nodes = 1
process_grid = [2, 2]                # Product = 4 GPUs
max_basis_size = 20
```

### Generic Slurm cluster

Read the cores, the memory, and the GPUs of each node, as in [Prerequisites](#prerequisites). Use the
`slurm` profile for CPU nodes and `slurm-gpu` for GPU nodes. Set `slurm_memory` when the default
memory per CPU of your site is too small for the job.

| Node type | Solver | Settings |
| --- | --- | --- |
| CPU | Fulqrum `cpu` | `num_nodes = 1`, `ompthreads` = cores of the node |
| CPU | SBD | `mpiprocs x ompthreads` = cores, `num_nodes` from the [grid steps](#choose-the-grid), `adet_comm_size x bdet_comm_size = num_nodes x mpiprocs` |
| GPU | Fulqrum `gpu-mpi` | `gpus_per_node` = GPUs of the node, `ompthreads` = cores / GPUs, `process_grid` product = total GPUs |
| GPU | SBD GPU | Same as Fulqrum `gpu-mpi`, with the SBD grid in place of `process_grid` |

For example, CPU nodes with 64 cores and 256 GB hold `D = 9e8` for SBD at about 187 GB in total
on 2 nodes (94 GB each), with 16 ranks x 4 threads per node and a grid of 4 x 8:

```toml
solver = "sbd"
profile = "slurm"
queue = "normal"
work_dir = "/shared/work/sqd"     # QCSC_WORK_DIR
slurm_memory = "240G"

[sbd]
sbd_executable = "~/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag"
num_nodes = 2
mpiprocs = 16                        # --ntasks-per-node
ompthreads = 4                       # --cpus-per-task
adet_comm_size = 4
bdet_comm_size = 8
```

For a GPU partition, keep `gpus_per_node` equal to the GPUs of a node and set `ompthreads` to the
cores of a node divided by its GPUs. If your Slurm version does not pass `--cpus-per-task` from
the batch script to `srun`, add `SRUN_CPUS_PER_TASK` with the same value to `[environments]`.

## Check the run

Before a large run, create the blocks with `--dry-run` to print the resolved settings, and run
one SQD iteration at a smaller `subspace_dim`. Then check the following items.

1. **Resolved settings:** `sqd-create-blocks --config "$QCSC_PREFECT/algorithms/open_loop/sqd/sqd_blocks.toml" --dry-run` shows
   `num_nodes`, `mpiprocs`, `ompthreads`, `mpi_options`, and the solver grid. It fails when the
   SBD grid does not divide `num_nodes x mpiprocs`, when the Fulqrum `process_grid` does not
   match it, when the rank count of `mpi_options` or `max-proc-per-node` does not match it, or
   when `gpus_per_node` does not fit the target.
2. **Cores:** `mpiprocs x ompthreads` is at most the cores of a node.
3. **Measured memory:** read the peak memory of the solver job and scale it with `D`.
   - Slurm: `sacct -j <job_id> --format=JobID,MaxRSS,NNodes` (MaxRSS is per task).
   - PBS: `qstat -fx <job_id>` and its `resources_used.mem` (whole job).
   - Fugaku: the statistics file that the job script requests with `#PJM -S`, in the job
     directory as `*.stats`.
   - GPUs: `nvidia-smi` on a compute node during the solve.
4. **Fulqrum `csr`:** read `nnz` in the `metadata` of `result.json` in the job directory under
   `$QCSC_WORK_DIR`.
5. **Concurrent jobs:** `num_batches x num_nodes` nodes fit your queue share.

## References

- Fulqrum: <https://github.com/Qiskit/fulqrum>
- SBD, selected basis diagonalization: <https://github.com/r-ccs-cms/sbd>
- DICE, as built by `packages/qcsc-prefect-dice/native/build_dice.sh`:
  <https://github.com/caleb-johnson/Dice>
- DICE MPI tuning notebook: `algorithms/open_loop/sqd/perf/tuning_mpi_parameters_with_dice.ipynb`
- Fugaku system specifications, RIKEN R-CCS: <https://www.r-ccs.riken.jp/en/fugaku/about/>
- ROQUO system configuration, RIKEN R-CCS:
  <https://www.r-ccs.riken.jp/en/outreach/topics/20251118-1>
