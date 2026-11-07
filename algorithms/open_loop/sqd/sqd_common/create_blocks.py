"""``sqd-create-blocks``: create the Prefect blocks of the SQD workflow for one solver.

Usage::

    sqd-create-blocks --config sqd_blocks.toml       # solver from the file (default fulqrum)
    sqd-create-blocks --solver dice --config sqd_blocks.toml

The remaining arguments go to the solver's own script (``sqd_fulqrum.create_blocks``
or ``sqd_dice.create_blocks``). Each also saves the Advanced SQD Settings block,
whose Solver Block names the solver block it created, so the runs use that solver.
An SBD solver block is created with ``algorithms/closed_loop/sqd_de/create_blocks.py``;
set Solver Block of the Advanced SQD Settings block to ``sbd_solver_job/<name>``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqd_common.block_config import read_config_file, solver_of

SOLVERS = ("fulqrum", "dice")


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
        raise SystemExit(
            "Create the SBD solver block with algorithms/closed_loop/sqd_de/create_blocks.py, "
            "then set Solver Block of the Advanced SQD Settings block to "
            "'sbd_solver_job/<name>'."
        )
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
