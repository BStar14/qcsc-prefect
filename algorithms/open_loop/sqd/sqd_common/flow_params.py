"""Flow parameters shared by the SQD workflows.

One ``FlowParameters`` model is the single argument of each SQD flow, so a
deployment's run form in the Prefect UI shows every setting of a run in one
place, grouped by the Qiskit Patterns stage it controls.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_RUNTIME_BLOCK_NAME = "ibm-runner"
# Repository root of a source checkout (algorithms/open_loop/sqd/sqd_common/ is four levels down).
REPO_ROOT = Path(__file__).resolve().parents[4]


class MoleculeGeometry(BaseModel):
    """Molecule definition by geometry and basis."""

    atom: str = Field(
        description="Definition for molecule structure.",
        title="Atom",
    )
    basis: str = Field(
        default="6-31g",
        description="Name of basis set.",
        title="Basis Set",
    )
    symmetry: str | bool = Field(
        default=False,
        description="Whether to use symmetry, otherwise string of point group name.",
        title="Symmetry",
    )
    spin_sq: float = Field(
        default=0.0,
        description=(
            "Target value for the total spin squared for the ground state. "
            "The DICE solver uses it as a spin penalty; fulqrum ignores it."
        ),
        title="Total Spin Squared",
    )
    ccsd_maxiter: int = Field(
        default=4,
        description=(
            "Number of CCSD kernel runs. Each run after the first restarts from the "
            "previous amplitudes, which removes the memory effect of the DIIS history."
        ),
        title="CCSD Restarts",
        ge=1,
    )


class FCIDumpFile(BaseModel):
    """Molecule definition by FCIDump file."""

    fcidump_file: str = Field(
        description=(
            "Location of FCIDump file storing 1-electron and 2-electron integrals. "
            "We assume these are precomputed in the MO basis."
        ),
        title="FCIDump File",
    )

    @field_validator("fcidump_file")
    @classmethod
    def _resolve_fcidump_file(cls, value: str) -> str:
        """Resolve a relative path from the working directory, else from the repository root.

        A deployment serves from the package directory, so example paths such as
        ``algorithms/open_loop/sqd/data/n2_631g_fc.fcidump`` resolve against the checkout.
        """
        path = Path(value).expanduser()
        if path.is_absolute() or path.exists() or not (REPO_ROOT / path).exists():
            return str(path)
        return str(REPO_ROOT / path)

    spin_sq: float = Field(
        default=0.0,
        description=(
            "Target value for the total spin squared for the ground state. "
            "The DICE solver uses it as a spin penalty; fulqrum ignores it."
        ),
        title="Total Spin Squared",
    )
    ccsd_maxiter: int = Field(
        default=4,
        description=(
            "Number of CCSD kernel runs. Each run after the first restarts from the "
            "previous amplitudes, which removes the memory effect of the DIIS history."
        ),
        title="CCSD Restarts",
        ge=1,
    )


class CircuitParameters(BaseModel):
    """LUCJ ansatz and transpiler settings."""

    n_lucj_layers: int = Field(
        default=1,
        description="Number of repetition of the unit layer of LUCJ ansatz.",
        title="LUCJ Layers",
        ge=1,
    )
    compress: bool = Field(
        default=False,
        description=(
            "Optimize the LUCJ parameters to minimize the error of the truncated "
            "double factorization of the CCSD t2 amplitudes."
        ),
        title="Compress LUCJ Parameters",
    )
    compress_maxiter: int = Field(
        default=100,
        description="Maximum number of optimizer iterations when Compress is on.",
        title="Compression Max Iteration",
        ge=1,
    )
    max_ab_pairs: int | None = Field(
        default=None,
        description=(
            "Keep at most this many alpha-beta interaction pairs (every 4th orbital on a "
            "heavy-hex lattice). Leave empty to keep all of them."
        ),
        title="Max Alpha-Beta Pairs",
        ge=1,
    )
    use_reset_mitigation: bool = Field(
        default=False,
        description=(
            "Use reset mitigation scheme that post-selects outcomes with non-ground initial state. "
            "This post-selection reduces the net shot number and its retention rate depends on "
            "the quality of hardware reset instruction."
        ),
        title="Reset Mitigation",
    )
    optimization_level: int = Field(
        default=3,
        description="Transpile: Optimization level of transpiler",
        title="Optimization Level",
        ge=0,
        le=3,
    )
    sabre_max_iterations: int = Field(
        default=8,
        description=(
            "Transpile: The number of forward-backward routing iterations to refine "
            "the layout and reduce routing costs."
        ),
        title="SABRE Max Iteration",
        ge=1,
    )
    sabre_swap_trials: int = Field(
        default=10,
        description=(
            "Transpile: The number of routing trials for each layout, refining gate "
            "placement for better routing."
        ),
        title="SABRE SWAP Trials",
        ge=1,
    )
    sabre_layout_trials: int = Field(
        default=10000,
        description=(
            "Transpile: The number of random initial layouts tested, selecting the "
            "one that minimizes SWAP gates."
        ),
        title="SABRE Layout Trials",
        ge=1,
    )


class SamplingParameters(BaseModel):
    """Where the bitstrings come from: a QPU, an earlier job, or a random generator."""

    quantum_source: Literal["real-device", "random"] = Field(
        default="real-device",
        description=(
            "Sample on a quantum computer, or draw deterministic uniform random "
            "bitstrings without a QPU."
        ),
        title="Quantum Source",
    )
    shots: int = Field(
        default=100_000,
        description="Number of shots.",
        title="Shots",
        ge=1,
    )
    random_seed: int = Field(
        default=24,
        description="RNG seed of the random bitstrings when Quantum Source is 'random'.",
        title="Random Seed",
        ge=0,
    )
    runner_name: str = Field(
        default=DEFAULT_RUNTIME_BLOCK_NAME,
        description=(
            "QuantumRuntime block for IBM Quantum Runtime sampling. Not used with "
            "QRMI Runtime Config."
        ),
        title="Quantum Runtime Block",
    )
    sampler_options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra IBM Quantum Runtime sampler options, merged with Shots "
            "(for example {'options': {'dynamical_decoupling': {'enable': true}}}). "
            "Not supported with QRMI."
        ),
        title="Sampler Options",
    )
    qrmi_runtime_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "QrmiRuntime settings to sample through QRMI on Slurm (root_dir, "
            "command_block_name, execution_profile_block_name, hpc_profile_block_name, "
            "resource). Leave empty to use the Quantum Runtime Block."
        ),
        title="QRMI Runtime Config",
    )
    qiskit_runtime_job_id: str | None = Field(
        default=None,
        description=(
            "Load the samples of a finished IBM Quantum Runtime job instead of sampling. "
            "Uses the credentials of the Quantum Runtime Block."
        ),
        title="Load from Qiskit Runtime Job ID",
    )
    qrmi_result_dir: str | None = Field(
        default=None,
        description=(
            "Load the samples of a finished QRMI Slurm job instead of sampling. "
            "Give its job directory (the work_dir in the sqd-sampling-source artifact)."
        ),
        title="Load from QRMI Job Directory",
    )

    @model_validator(mode="after")
    def _check_sources(self) -> "SamplingParameters":
        reloads = [
            name
            for name in ("qiskit_runtime_job_id", "qrmi_result_dir")
            if getattr(self, name) is not None
        ]
        if len(reloads) > 1:
            raise ValueError("Set at most one of qiskit_runtime_job_id and qrmi_result_dir.")
        if self.quantum_source == "random":
            conflicts = reloads + (
                ["qrmi_runtime_config"] if self.qrmi_runtime_config is not None else []
            )
            if conflicts:
                raise ValueError(f"Quantum Source 'random' cannot be combined with {conflicts[0]}.")
        if self.qrmi_runtime_config is not None and self.qiskit_runtime_job_id is not None:
            raise ValueError(
                "qiskit_runtime_job_id loads an IBM Quantum Runtime job; "
                "use qrmi_result_dir with qrmi_runtime_config."
            )
        if self.qrmi_runtime_config is not None and self.sampler_options:
            raise ValueError("sampler_options are not supported with qrmi_runtime_config.")
        if "shots" in self.sampler_options.get("params", {}):
            raise ValueError("Set shots with the Shots parameter, not in sampler_options.")
        return self


class SQDParameters(BaseModel):
    """Configuration recovery and diagonalization settings."""

    subspace_dim: int = Field(
        description="Dimension d of subsampled bitstrings for diagonalization.",
        title="Subspace Dimension",
        ge=1,
    )
    num_batches: int = Field(
        description=(
            "Number of batches of configurations used by the different calls to the "
            "eigenstate solver. Each batch is one solver job; they run concurrently."
        ),
        title="Batch Number",
        ge=1,
    )
    max_iterations: int = Field(
        description="Number of self-consistent configuration recovery iterations.",
        title="Max Iteration",
        ge=1,
    )
    checkpoint: bool = Field(
        default=True,
        description=(
            "Save the recovery state after every iteration, so a Retry of this run continues "
            "from the last completed iteration. Disk use: each run writes one ckpt_t00_w0.npz "
            "and source.json under <solver block root_dir>/sqd_checkpoints/<flow run id>/, "
            "about 16 x (number of orbitals) bytes plus a few kB, rewritten in place. That is "
            "under 1 MB even for 1000 orbitals, but one folder is added per run and never "
            "removed. The solver job directories next to it are much larger."
        ),
        title="Checkpoint Recovery",
    )
    resume_checkpoint: str | None = Field(
        default=None,
        description=(
            "Continue the recovery of another run from its ckpt_t00_w0.npz (or the folder "
            "that holds it, with its source.json). The samples, the Hamiltonian, Subspace "
            "Dimension and Batch Number must be the same as in that run; only Max Iteration "
            "may change. A checkpoint with 32 completed iterations and Max Iteration 200 "
            "continues at iteration 33/200. The file is only read."
        ),
        title="Resume from Checkpoint",
    )


class FlowParameters(BaseModel):
    """Parameters of an SQD workflow run, grouped by Qiskit Patterns stage.

    Map: ``molecule``. Optimize and Execute: ``circuit`` and ``sampling``.
    Post-process: ``sqd`` and ``solver_block_name``.
    """

    molecule: MoleculeGeometry | FCIDumpFile = Field(
        description="PySCF definition of the molecule to solve for.",
        title="Molecule",
    )
    circuit: CircuitParameters = Field(
        default_factory=CircuitParameters,
        description="LUCJ ansatz and transpiler settings.",
        title="Circuit",
    )
    sampling: SamplingParameters = Field(
        default_factory=SamplingParameters,
        description="Quantum sampling source and settings.",
        title="Sampling",
    )
    sqd: SQDParameters = Field(
        description="Configuration recovery and diagonalization settings.",
        title="SQD",
    )
    solver_block_name: str = Field(
        description="Name of the solver block that diagonalizes each subspace.",
        title="Solver Block",
    )
