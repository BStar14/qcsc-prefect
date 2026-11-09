from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qcsc_prefect_adapters.base.jinja_env import make_env
from qcsc_prefect_core.models.execution_profile import ExecutionProfile

_ENV = make_env("qcsc_prefect_adapters.fugaku")
_TEMPLATE = "batch.pjm.j2"
_BULK_MANIFEST_TEMPLATE = "batch_bulk_manifest.pjm.j2"


@dataclass(frozen=True)
class FugakuJobRequest:
    """Target-specific request fields required to build a Fugaku PJM job.

    Attributes:
        queue_name: Fugaku resource group name.
        project: Fugaku group name passed to PJM.
        executable: Absolute or scheduler-visible command path to execute.
        job_name: PJM job name used for scheduler display and output filenames.
        gfscache: Optional ``PJM_LLIO_GFSCACHE`` path.
        mpi_options_for_pjm: Optional MPI options emitted as PJM directives.
        spack_modules: Optional Spack modules loaded by the generated script.
        pjm_resources: Additional raw ``#PJM -L`` resource directives.
    """

    queue_name: str
    project: str
    executable: str
    job_name: str = "prefect_job"
    gfscache: str | None = None
    mpi_options_for_pjm: list[str] | None = None
    spack_modules: list[str] | None = None
    pjm_resources: list[str] | None = None


def to_fugaku_template_kwargs(
    *,
    work_dir: Path,
    exec_profile: ExecutionProfile,
    req: FugakuJobRequest,
    script_basename: str = "batch.pjm",
) -> dict:
    """Build template variables for the Fugaku PJM script.

    Args:
        work_dir: Base working directory used to derive output paths.
        exec_profile: Scheduler-independent execution profile.
        req: Fugaku-specific scheduler request fields.
        script_basename: Script basename used for output/stat filenames.

    Returns:
        A dictionary that can be passed to the Fugaku Jinja template.

    Raises:
        ValueError: If ``exec_profile.gpus_per_node`` is set. Fugaku nodes
            have no GPUs.
    """

    if exec_profile.gpus_per_node is not None:
        raise ValueError(
            "Fugaku nodes have no GPUs; gpus_per_node must be unset for hpc_target 'fugaku'."
        )

    stdout_path = work_dir / f"{script_basename}.{req.job_name}.out"
    stderr_path = work_dir / f"{script_basename}.{req.job_name}.err"
    stat_path = work_dir / f"{script_basename}.{req.job_name}.stats"

    kw: dict = {
        "resource_group": req.queue_name,
        "group_name": req.project,
        "job_name": req.job_name,
        "num_nodes": exec_profile.num_nodes,
        "elapse_time": exec_profile.walltime,
        "launcher": exec_profile.launcher,
        "executable": req.executable,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stat_path": str(stat_path),
        "pjm_resources": list(req.pjm_resources or []),
        "mpi_options_for_pjm": list(req.mpi_options_for_pjm or []),
        "gfscache": req.gfscache,
        "modules": list(exec_profile.modules or []),
        "spack_modules": list(req.spack_modules or []),
        "environments": dict(exec_profile.environments or {}),
        "mpi_options": list(exec_profile.mpi_options or []),
        "arguments": list(exec_profile.arguments or []),
    }
    return kw


def render_script(
    *,
    work_dir: Path,
    exec_profile: ExecutionProfile,
    req: FugakuJobRequest,
    script_basename: str = "batch.pjm",
) -> str:
    """Render a Fugaku job script text from the configured Jinja template.

    .. note::
        The template file is configured by module constant ``_TEMPLATE`` and
        is expected to be a ``.j2`` file.

    Args:
        work_dir: Base working directory used in template variables.
        exec_profile: Scheduler-independent execution profile.
        req: Fugaku-specific scheduler request fields.
        script_basename: Script basename used for output/stat filenames.

    Returns:
        Rendered PJM script text.
    """

    template = _ENV.get_template(_TEMPLATE)
    kwargs = to_fugaku_template_kwargs(
        work_dir=work_dir,
        exec_profile=exec_profile,
        req=req,
        script_basename=script_basename,
    )
    return template.render(**kwargs)


def to_fugaku_manifest_bulk_template_kwargs(
    *,
    work_dir: Path,
    bulk_manifest_dir: Path,
    exec_profile: ExecutionProfile,
    req: FugakuJobRequest,
    script_basename: str = "batch_bulk.pjm",
) -> dict:
    """Build template variables for a manifest-driven Fugaku native bulk script."""

    kwargs = to_fugaku_template_kwargs(
        work_dir=work_dir,
        exec_profile=exec_profile,
        req=req,
        script_basename=script_basename,
    )
    kwargs["bulk_manifest_dir"] = str(Path(bulk_manifest_dir).expanduser())
    return kwargs


def render_manifest_bulk_script(
    *,
    work_dir: Path,
    bulk_manifest_dir: Path,
    exec_profile: ExecutionProfile,
    req: FugakuJobRequest,
    script_basename: str = "batch_bulk.pjm",
) -> str:
    """Render a Fugaku native bulk script that selects a per-subjob manifest."""

    template = _ENV.get_template(_BULK_MANIFEST_TEMPLATE)
    kwargs = to_fugaku_manifest_bulk_template_kwargs(
        work_dir=work_dir,
        bulk_manifest_dir=bulk_manifest_dir,
        exec_profile=exec_profile,
        req=req,
        script_basename=script_basename,
    )
    return template.render(**kwargs)


def write_script_file(*, work_dir: Path, filename: str, text: str) -> Path:
    """Write a rendered job script into the work directory.

    .. note::
        This function is expected to be called inside
        `qcsc_prefect_executor.fugaku.run.run_fugaku_job`.
        Workflow authors normally do not need to call it directly.

    .. note::
        The ``text`` argument is expected to come from `render_script`,
        which renders the ``.j2`` template specified by ``_TEMPLATE``.

    Args:
        work_dir: Base working directory where the script file is created.
        filename: Script file name (for example ``batch.pjm``).
        text: Rendered script text.

    Returns:
        Absolute path to the created job script file.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / filename
    path.write_text(text)
    return path
