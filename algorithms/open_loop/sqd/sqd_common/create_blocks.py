"""``sqd-create-blocks``: create the Prefect blocks of the SQD workflow for one solver.

Usage::

    sqd-create-blocks --config sqd_blocks.toml       # solver from the file (default fulqrum)
    sqd-create-blocks --solver dice --config sqd_blocks.toml
    sqd-create-blocks --solver sbd --config sqd_blocks.toml   # records the SBD block

The remaining arguments go to the solver's own script (``sqd_fulqrum.create_blocks``
or ``sqd_dice.create_blocks``). Each also saves the Advanced SQD Settings block,
whose Solver Block names the solver block it created, so the runs use that solver.
The SBD solver block is created with ``algorithms/closed_loop/sqd_de/create_blocks.py``;
with ``solver = "sbd"`` this script saves only the Advanced SQD Settings block, whose
Solver Block names that block: ``[sbd] solver_block_name``, default ``davidson-solver``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqd_common.block_config import read_config_file, solver_of

SOLVERS = ("fulqrum", "dice", "sbd")
# The SBD solver block that algorithms/closed_loop/sqd_de/create_blocks.py saves by default.
SBD_SOLVER_BLOCK_NAME = "davidson-solver"


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    solver = _config_solver(argv)
    if "--solver" in argv:
        index = argv.index("--solver")
        if index + 1 >= len(argv):
            raise SystemExit(f"--solver needs one of {', '.join(SOLVERS)}")
        solver = argv[index + 1]
        del argv[index : index + 2]
    if solver == "sbd":
        _record_sbd(argv)
        return
    if solver not in SOLVERS:
        raise SystemExit(f"--solver must be one of {', '.join(SOLVERS)}; got {solver!r}")

    if solver == "fulqrum":
        from sqd_fulqrum.create_blocks import main as create
    else:
        try:
            from sqd_dice.create_blocks import main as create
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f'The DICE solver needs qcsc-prefect-dice: pip install "sqd[dice]" ({exc})'
            )
        sys.argv = [sys.argv[0], *argv]
        create()
        return
    create(argv)


def _record_sbd(argv: list[str]) -> None:
    """Save the Advanced SQD Settings block for an SBD solver block made elsewhere.

    ``algorithms/closed_loop/sqd_de/create_blocks.py`` saves the SBD solver block
    and its job blocks; ``[sqd_advanced]`` of this config applies as for the other
    solvers. ``[qrmi]`` or ``[runtime]`` saves the shared QRMI blocks as well.
    """
    from sqd_common.advanced_blocks import (
        advanced_block_name,
        advanced_settings,
        save_advanced_block,
    )
    from sqd_common.block_config import load_block_config
    from sqd_common.block_names import solver_ref
    from sqd_common.qrmi_blocks import (
        create_qrmi_blocks,
        resolve_qrmi_settings,
        resolve_runtime_settings,
    )
    from sqd_common.runtime_block import build_runtime_block, describe, runtime_block_name

    config_path = _config_path(argv)
    config = load_block_config(config_path, "sbd") if config_path is not None else {}
    name = str(config.get("solver_block_name") or SBD_SOLVER_BLOCK_NAME).strip()
    ref = solver_ref("sbd", name)
    settings = advanced_settings(config, ref)
    block_name = advanced_block_name(config)
    qrmi = resolve_qrmi_settings(argparse.Namespace(), config, _sbd_job_settings(config))
    runtime = resolve_runtime_settings(config, qrmi)
    runtime_name = runtime_block_name(None, config)
    if "--dry-run" in argv:
        resolved = {"advanced_block": {"name": block_name, **settings.model_dump()}}
        if qrmi is not None:
            resolved["qrmi_blocks"] = qrmi
        if runtime is not None:
            resolved["runtime_block"] = {"name": runtime_name, **describe(runtime)}
        print(json.dumps(resolved, indent=2, default=str))
        return
    if qrmi is not None:
        create_qrmi_blocks(qrmi)
    if runtime is not None:
        build_runtime_block(runtime).save(runtime_name, overwrite=True)
        print(f"QRMI runtime block: {runtime_name} (scheduler={runtime['scheduler']})")
    save_advanced_block(settings, block_name)
    print(f"Advanced SQD Settings block: {block_name} (Solver Block {ref})")
    print(
        "Create the SBD solver block itself with algorithms/closed_loop/sqd_de/create_blocks.py "
        f"(name {name!r}); this script does not."
    )


def _sbd_job_settings(config: dict) -> dict:
    """The cluster settings that the QRMI job inherits, as the solver scripts pass them."""
    hpc_target = str(config.get("hpc_target", "slurm")).strip().lower()
    if "qrmi" in config and not str(config.get("queue") or "").strip():
        raise RuntimeError("Missing required setting: queue (the QRMI job's Slurm partition)")
    if "qrmi" in config and not str(config.get("work_dir") or "").strip():
        raise RuntimeError("Missing required setting: work_dir")
    return {
        "hpc_target": hpc_target,
        "queue": str(config.get("queue") or "").strip(),
        "project": str(config.get("project") or "").strip(),
        "root_dir": str(config.get("work_dir") or ""),
        "ompthreads": config.get("ompthreads"),
        "modules": config.get("modules"),
        "pre_commands": config.get("pre_commands"),
        "environments": config.get("environments"),
        "slurm_memory": config.get("slurm_memory"),
    }


def _config_path(argv: list[str]) -> Path | None:
    if "--config" not in argv:
        return None
    index = argv.index("--config")
    return Path(argv[index + 1]) if index + 1 < len(argv) else None


def _config_solver(argv: list[str]) -> str:
    """The ``solver`` of the ``--config`` file, else fulqrum."""
    if "--config" not in argv:
        return "fulqrum"
    index = argv.index("--config")
    if index + 1 >= len(argv):
        return "fulqrum"
    path = Path(argv[index + 1])
    if not path.exists():
        return "fulqrum"
    return solver_of(read_config_file(path))


if __name__ == "__main__":
    main()
