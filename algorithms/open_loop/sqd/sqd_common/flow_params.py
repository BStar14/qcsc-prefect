"""Flow parameters shared by the SQD workflows.

One ``FlowParameters`` model is the single argument of each SQD flow, so a
deployment's run form in the Prefect UI shows every setting of a run in one
place, grouped by the Qiskit Patterns stage it controls.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# QrmiRuntimeBlock that samples, and the IBM QuantumRuntime block that reloads a job.
DEFAULT_QRMI_BLOCK_NAME = "qrmi-runtime"
DEFAULT_RUNTIME_BLOCK_NAME = "ibm-runner"
DEFAULT_SOLVER_BLOCK_REF = "fulqrum_solver_job/sqd-fulqrum-solver"
# Repository root of a source checkout (algorithms/open_loop/sqd/sqd_common/ is four levels down).
REPO_ROOT = Path(__file__).resolve().parents[4]


def _at(position: int) -> dict[str, int]:
    """Field order in the Prefect run form, which otherwise sorts fields alphabetically."""
    return {"position": position}


def _drop_empty(data: dict[str, Any], *names: str) -> dict[str, Any]:
    """Drop removed keys left empty, e.g. by a run form filled before they were removed.

    The Prefect UI keeps such keys hidden in the parameters it submits.
    """
    return {
        key: value for key, value in data.items() if key not in names or value not in (None, {})
    }


def _shown_as_required(*names: str):
    """Mark defaulted key fields as required in the run form, so that it highlights them."""

    def extra(schema: dict[str, Any]) -> None:
        required = schema.setdefault("required", [])
        required.extend(name for name in names if name not in required)

    return extra


class MoleculeGeometry(BaseModel):
    """Molecule definition by geometry and basis."""

    atom: str = Field(
        description="Definition for molecule structure.",
        title="Atom",
        json_schema_extra=_at(0),
    )
    basis: str = Field(
        default="6-31g",
        description="Name of basis set.",
        title="Basis Set",
        json_schema_extra=_at(1),
    )
    symmetry: str | bool = Field(
        default=False,
        description="Whether to use symmetry, otherwise string of point group name.",
        title="Symmetry",
        json_schema_extra=_at(2),
    )
    spin_sq: float | None = Field(
        default=0.0,
        description=(
            "Target value for the total spin squared for the ground state, or null for no "
            "target. DICE uses it as a spin penalty, fulqrum ignores it, and SBD requires null."
        ),
        title="Total Spin Squared",
        json_schema_extra=_at(3),
    )
    ccsd_maxiter: int = Field(
        default=4,
        description=(
            "Number of CCSD kernel runs. Each run after the first restarts from the "
            "previous amplitudes, which removes the memory effect of the DIIS history."
        ),
        title="CCSD Restarts",
        ge=1,
        json_schema_extra=_at(4),
    )


class FCIDumpFile(BaseModel):
    """Molecule definition by FCIDump file."""

    fcidump_file: str = Field(
        description=(
            "Location of FCIDump file storing 1-electron and 2-electron integrals. "
            "We assume these are precomputed in the MO basis."
        ),
        title="FCIDump File",
        json_schema_extra=_at(0),
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

    spin_sq: float | None = Field(
        default=0.0,
        description=(
            "Target value for the total spin squared for the ground state, or null for no "
            "target. DICE uses it as a spin penalty, fulqrum ignores it, and SBD requires null."
        ),
        title="Total Spin Squared",
        json_schema_extra=_at(1),
    )
    ccsd_maxiter: int = Field(
        default=4,
        description=(
            "Number of CCSD kernel runs. Each run after the first restarts from the "
            "previous amplitudes, which removes the memory effect of the DIIS history."
        ),
        title="CCSD Restarts",
        ge=1,
        json_schema_extra=_at(2),
    )


class TranspilerParameters(BaseModel):
    """Preset pass manager with a large SABRE layout search."""

    optimization_level: int = Field(
        default=3,
        description="Optimization level of the preset pass manager.",
        title="Optimization Level",
        ge=0,
        le=3,
        json_schema_extra=_at(0),
    )
    sabre_layout_trials: int = Field(
        default=10000,
        description=(
            "The number of random initial layouts tested, selecting the one that minimizes "
            "SWAP gates."
        ),
        title="SABRE Layout Trials",
        ge=1,
        json_schema_extra=_at(1),
    )
    sabre_swap_trials: int = Field(
        default=10,
        description=(
            "The number of routing trials for each layout, refining gate placement for "
            "better routing."
        ),
        title="SABRE SWAP Trials",
        ge=1,
        json_schema_extra=_at(2),
    )
    sabre_max_iterations: int = Field(
        default=8,
        description=(
            "The number of forward-backward routing iterations to refine the layout and "
            "reduce routing costs."
        ),
        title="SABRE Max Iteration",
        ge=1,
        json_schema_extra=_at(3),
    )


class CircuitParameters(BaseModel):
    """LUCJ ansatz and transpiler settings."""

    n_lucj_layers: int = Field(
        default=1,
        description="Number of repetition of the unit layer of LUCJ ansatz.",
        title="LUCJ Layers",
        ge=1,
        json_schema_extra=_at(0),
    )
    compress: bool = Field(
        default=True,
        description=(
            "Optimize the LUCJ parameters to minimize the error of the truncated "
            "double factorization of the CCSD t2 amplitudes."
        ),
        title="Compress LUCJ Parameters",
        json_schema_extra=_at(1),
    )
    compress_maxiter: int = Field(
        default=20,
        description="Maximum number of optimizer iterations when Compress is on.",
        title="Compression Max Iteration",
        ge=1,
        json_schema_extra=_at(2),
    )
    use_reset_mitigation: bool = Field(
        default=False,
        description=(
            "Use reset mitigation scheme that post-selects outcomes with non-ground initial state. "
            "This post-selection reduces the net shot number and its retention rate depends on "
            "the quality of hardware reset instruction."
        ),
        title="Reset Mitigation",
        json_schema_extra=_at(3),
    )
    configure_transpiler: bool = Field(
        default=False,
        description=(
            "Use the Transpiler settings below. Off: optimization level 3 with the default "
            "SABRE layout search (10000 layout trials, 10 SWAP trials, 8 iterations)."
        ),
        title="Configure Transpiler",
        json_schema_extra=_at(4),
    )
    transpiler: TranspilerParameters = Field(
        default_factory=TranspilerParameters,
        description="Used only when Configure Transpiler is on.",
        title="Transpiler",
        json_schema_extra=_at(5),
    )

    @property
    def transpile_options(self) -> TranspilerParameters:
        """The transpiler settings in effect: the configured ones, or the defaults."""
        return self.transpiler if self.configure_transpiler else TranspilerParameters()

    @model_validator(mode="before")
    @classmethod
    def _reject_moved_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = _drop_empty(data, "max_ab_pairs")
            if "max_ab_pairs" in data:
                raise ValueError(
                    "circuit.max_ab_pairs was removed: the heavy-hex pattern keeps at most 5 "
                    "alpha-beta pairs."
                )
            # A key left at its old default, or at the value it moved to, changes nothing.
            transpiler = data.get("transpiler")
            configured = transpiler if isinstance(transpiler, dict) else {}
            defaults = TranspilerParameters()
            data = {
                key: value
                for key, value in data.items()
                if key not in TranspilerParameters.model_fields
                or value not in (getattr(defaults, key), configured.get(key))
            }
            moved = sorted(set(data) & set(TranspilerParameters.model_fields))
            if moved:
                raise ValueError(
                    f"circuit.{moved[0]} moved to circuit.transpiler.{moved[0]}: set "
                    "configure_transpiler = true and the transpiler settings there."
                )
        return data


class SamplingParameters(BaseModel):
    """Where the bitstrings come from: a QPU, an earlier job, or a random generator."""

    quantum_source: Literal["real-device", "random"] = Field(
        default="real-device",
        description=(
            "Sample on a quantum computer, or draw deterministic uniform random "
            "bitstrings without a QPU."
        ),
        title="Quantum Source",
        json_schema_extra=_at(0),
    )
    shots: int = Field(
        default=10_000,
        description="Number of shots.",
        title="Shots",
        ge=1,
        json_schema_extra=_at(1),
    )
    random_seed: int = Field(
        default=24,
        description="RNG seed of the random bitstrings when Quantum Source is 'random'.",
        title="Random Seed",
        ge=0,
        json_schema_extra=_at(3),
    )
    runtime_block_name: str = Field(
        default=DEFAULT_QRMI_BLOCK_NAME,
        description=(
            "QrmiRuntimeBlock that samples the circuit, through IQP from the workflow "
            "process or in a Slurm job (its scheduler). The flow sets the transpiler "
            "hook and its configuration from the Circuit settings."
        ),
        title="QRMI Runtime Block",
        json_schema_extra=_at(2),
    )
    runner_name: str = Field(
        default=DEFAULT_RUNTIME_BLOCK_NAME,
        description=(
            "IBM QuantumRuntime block whose credentials read the job of Load from Qiskit "
            "Runtime Job ID. Not used for sampling."
        ),
        title="Quantum Runtime Block (reload)",
        json_schema_extra=_at(6),
    )
    qiskit_runtime_job_id: str | None = Field(
        default=None,
        description=(
            "Load the samples of a finished IBM Quantum job (for example the job_id of an "
            "IQP run in the sqd-sampling-source artifact) instead of sampling. Uses the "
            "credentials of the Quantum Runtime Block (reload)."
        ),
        title="Load from Qiskit Runtime Job ID",
        json_schema_extra=_at(5),
    )
    qrmi_result_dir: str | None = Field(
        default=None,
        description=(
            "Load the samples of a finished QRMI Slurm job instead of sampling. "
            "Give its job directory (the qrmi_result_dir in the sqd-sampling-source "
            "artifact)."
        ),
        title="Load from QRMI Job Directory",
        json_schema_extra=_at(4),
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
        if self.quantum_source == "random" and reloads:
            raise ValueError(f"Quantum Source 'random' cannot be combined with {reloads[0]}.")
        return self

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = _drop_empty(data, "qrmi_runtime_config", "sampler_options")
            removed = {
                "qrmi_runtime_config": "save a QrmiRuntimeBlock and set runtime_block_name",
                "sampler_options": "QRMI takes only the Shots parameter",
            }
            for name, hint in removed.items():
                if name in data:
                    raise ValueError(f"sampling.{name} was removed in sqd 2.0: {hint}.")
        return data


class SQDParameters(BaseModel):
    """Configuration recovery and diagonalization settings."""

    model_config = ConfigDict(
        json_schema_extra=_shown_as_required("subspace_dim", "num_batches", "max_iterations")
    )

    subspace_dim: int = Field(
        default=90_000,
        description="Dimension d of subsampled bitstrings for diagonalization.",
        title="Subspace Dimension",
        ge=1,
        json_schema_extra=_at(0),
    )
    num_batches: int = Field(
        default=4,
        description=(
            "Number of batches of configurations used by the different calls to the "
            "eigenstate solver. Each batch is one solver job; they run concurrently."
        ),
        title="Batch Number",
        ge=1,
        json_schema_extra=_at(1),
    )
    max_iterations: int = Field(
        default=5,
        description="Number of self-consistent configuration recovery iterations.",
        title="Max Iteration",
        ge=1,
        json_schema_extra=_at(2),
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
        json_schema_extra=_at(3),
    )
    resume_checkpoint: str | None = Field(
        default=None,
        description=(
            "Continue the recovery of another run from its ckpt_t00_w0.npz (or the folder "
            "that holds it, with its source.json). The samples, the Hamiltonian, Subspace "
            "Dimension, Batch Number and the solver type (the slug of Solver Block) must be "
            "the same as in that run; only Max Iteration and the solver block name "
            "may change. A checkpoint with 32 completed iterations and Max Iteration 200 "
            "continues at iteration 33/200. The file is only read."
        ),
        title="Resume from Checkpoint",
        json_schema_extra=_at(4),
    )


class FlowParameters(BaseModel):
    """Parameters of an SQD workflow run, grouped by Qiskit Patterns stage.

    Map: ``molecule``. Optimize and Execute: ``circuit`` and ``sampling``.
    Post-process: ``sqd`` and ``solver_block_ref``.
    """

    molecule: MoleculeGeometry | FCIDumpFile = Field(
        description="PySCF definition of the molecule to solve for.",
        title="Molecule",
        json_schema_extra=_at(0),
    )
    circuit: CircuitParameters = Field(
        default_factory=CircuitParameters,
        description="LUCJ ansatz and transpiler settings.",
        title="Circuit",
        json_schema_extra=_at(1),
    )
    sampling: SamplingParameters = Field(
        default_factory=SamplingParameters,
        description="Quantum sampling source and settings.",
        title="Sampling",
        json_schema_extra=_at(2),
    )
    sqd: SQDParameters = Field(
        default_factory=SQDParameters,
        description="Configuration recovery and diagonalization settings.",
        title="SQD",
        json_schema_extra=_at(3),
    )
    solver_block_ref: str = Field(
        default=DEFAULT_SOLVER_BLOCK_REF,
        description=(
            "Solver block that diagonalizes each subspace, as '<block type slug>/<block "
            "name>': fulqrum_solver_job/<name> (default), dice_shci_solver_job/<name> "
            "(needs sqd[dice]) or sbd_solver_job/<name> (needs sqd[sbd]; closed shell, "
            "Total Spin Squared null)."
        ),
        title="Solver Block",
        json_schema_extra=_at(4),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_solver_block_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and "solver_block_name" in data:
            raise ValueError(
                "solver_block_name was removed in sqd 2.0: set solver_block_ref to "
                f"'fulqrum_solver_job/{data['solver_block_name']}' (or "
                "dice_shci_solver_job/..., sbd_solver_job/...)."
            )
        return data
