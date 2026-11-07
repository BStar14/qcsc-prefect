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
DEFAULT_ADVANCED_BLOCK_NAME = "sqd-advanced-settings"
# Run-form fields moved to the Advanced SQD Settings block, with their last run-form defaults.
MOVED_TO_ADVANCED = {
    "ccsd_maxiter": 4,
    "use_reset_mitigation": False,
    "configure_transpiler": False,
    "transpiler": None,
    "optimization_level": 3,
    "sabre_layout_trials": 10000,
    "sabre_swap_trials": 10,
    "sabre_max_iterations": 8,
}
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


def _drop_moved_to_advanced(data: dict[str, Any], where: str) -> dict[str, Any]:
    """Drop a stale field that moved to the Advanced SQD Settings block; reject a set one.

    A run form filled before the move keeps the fields hidden at their old values. A
    value equal to its old default, or a transpiler that was not switched on, is stale.
    """
    configured = bool(data.get("configure_transpiler"))
    kept = {}
    for key, value in data.items():
        if key not in MOVED_TO_ADVANCED:
            kept[key] = value
            continue
        if key == "transpiler":
            stale = value in (None, {}) or not configured or value == _OLD_TRANSPILER_DEFAULTS
        else:
            stale = value is None or value == MOVED_TO_ADVANCED[key]
        if not stale:
            raise ValueError(
                f"{where}.{key} moved to the Advanced SQD Settings block "
                f"({DEFAULT_ADVANCED_BLOCK_NAME}): set it there, or in [sqd_advanced] of the "
                "sqd-create-blocks config."
            )
    return kept


# The circuit.transpiler group of a run form before the move, at its defaults.
_OLD_TRANSPILER_DEFAULTS = {
    key: MOVED_TO_ADVANCED[key]
    for key in (
        "optimization_level",
        "sabre_layout_trials",
        "sabre_swap_trials",
        "sabre_max_iterations",
    )
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

    @model_validator(mode="before")
    @classmethod
    def _drop_moved(cls, data: Any) -> Any:
        return _drop_moved_to_advanced(data, "molecule") if isinstance(data, dict) else data


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

    @model_validator(mode="before")
    @classmethod
    def _drop_moved(cls, data: Any) -> Any:
        return _drop_moved_to_advanced(data, "molecule") if isinstance(data, dict) else data


class CircuitParameters(BaseModel):
    """LUCJ ansatz settings."""

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
    qpu_topology: Literal["heavy-hex", "square"] = Field(
        default="heavy-hex",
        description=(
            "Qubit connectivity of the QPU, which sets the LUCJ interaction pairs. heavy-hex "
            "(IBM Heron): alpha-beta pairs on every 4th orbital, at most 5. square (IBM "
            "Nighthawk): alpha-beta pairs on every orbital."
        ),
        title="QPU Topology",
        json_schema_extra=_at(3),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_moved_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = _drop_empty(data, "max_ab_pairs")
            if "max_ab_pairs" in data:
                raise ValueError(
                    "circuit.max_ab_pairs was removed: the alpha-beta pairs follow "
                    "circuit.qpu_topology (QPU Topology); heavy-hex keeps at most 5."
                )
            data = _drop_moved_to_advanced(data, "circuit")
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
        json_schema_extra=_at(4),
    )
    runtime_block_name: str = Field(
        default=DEFAULT_QRMI_BLOCK_NAME,
        description=(
            "QrmiRuntimeBlock that samples the circuit, through IQP from the workflow "
            "process or in a Slurm job (its scheduler). The flow sets the transpiler "
            "hook and its configuration from the Advanced SQD Settings."
        ),
        title="QRMI Runtime Block",
        json_schema_extra=_at(2),
    )
    qpu_resource: str | None = Field(
        default=None,
        description=(
            "Backend for this run, for example ibm_kingston, instead of the QRMI Runtime "
            "Block's resource. For Slurm it is also the --qpu resource and must be in the "
            "cluster's qrmi_config.json. Empty uses the block's."
        ),
        title="QPU Backend",
        json_schema_extra=_at(3),
    )
    runner_name: str = Field(
        default=DEFAULT_RUNTIME_BLOCK_NAME,
        description=(
            "IBM QuantumRuntime block whose credentials read the job of Load from Qiskit "
            "Runtime Job ID. Not used for sampling."
        ),
        title="Quantum Runtime Block (reload)",
        json_schema_extra=_at(7),
    )
    qiskit_runtime_job_id: str | None = Field(
        default=None,
        description=(
            "Load the samples of a finished IBM Quantum job (for example the job_id of an "
            "IQP run in the sqd-sampling-source artifact) instead of sampling. Uses the "
            "credentials of the Quantum Runtime Block (reload)."
        ),
        title="Load from Qiskit Runtime Job ID",
        json_schema_extra=_at(6),
    )
    qrmi_result_dir: str | None = Field(
        default=None,
        description=(
            "Load the samples of a finished QRMI Slurm job instead of sampling. "
            "Give its job directory (the qrmi_result_dir in the sqd-sampling-source "
            "artifact)."
        ),
        title="Load from QRMI Job Directory",
        json_schema_extra=_at(5),
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
        if self.qpu_resource is not None:
            self.qpu_resource = self.qpu_resource.strip() or None
        if self.qpu_resource and any(char.isspace() for char in self.qpu_resource):
            raise ValueError("QPU Backend must be one backend name without spaces.")
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
    carryover_fraction: float = Field(
        default=0.5,
        description=(
            "Share of each spin's CI strings, heaviest in the best ground state of an "
            "iteration, kept in every subspace of the next iteration. They count toward "
            "Subspace Dimension. 0 turns carryover off."
        ),
        title="Carryover Fraction",
        ge=0.0,
        le=1.0,
        json_schema_extra=_at(3),
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
        json_schema_extra=_at(4),
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
        json_schema_extra=_at(5),
    )


class FlowParameters(BaseModel):
    """Parameters of an SQD workflow run, grouped by Qiskit Patterns stage.

    Map: ``molecule``. Optimize and Execute: ``circuit`` and ``sampling``.
    Post-process: ``sqd``. The solver block and the rarely changed settings are in the
    Advanced SQD Settings block named by ``advanced_block_name``.
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
    advanced_block_name: str = Field(
        default=DEFAULT_ADVANCED_BLOCK_NAME,
        description=(
            "Advanced SQD Settings block with the solver block, CCSD restarts, reset "
            "mitigation and the transpiler. sqd-create-blocks saves it; without it the "
            "defaults apply."
        ),
        title="Advanced Settings Block",
        json_schema_extra=_at(4),
    )
    solver_block_ref: str | None = Field(
        default=None,
        description=(
            "Leave empty to use the Solver Block of the Advanced Settings Block. Set it, as "
            "'<block type slug>/<block name>', only to run with another solver block."
        ),
        title="Solver Block Override",
        json_schema_extra=_at(5),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_solver_block_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and "solver_block_name" in data:
            raise ValueError(
                "solver_block_name was removed in sqd 2.0: set Solver Block in the Advanced "
                "SQD Settings block (sqd-create-blocks sets it for the solver it creates)."
            )
        if isinstance(data, dict) and data.get("solver_block_ref") in (
            "",
            DEFAULT_SOLVER_BLOCK_REF,
        ):
            # The default of a run form filled before 2.2 is stale: the block holds it now.
            data = {key: value for key, value in data.items() if key != "solver_block_ref"}
        return data
