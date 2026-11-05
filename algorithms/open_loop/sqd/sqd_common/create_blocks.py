"""``sqd-create-blocks``: create the Prefect blocks of the SQD workflow for one solver.

Usage::

    sqd-create-blocks --config sqd_fulqrum_blocks.slurm.toml            # fulqrum (default)
    sqd-create-blocks --solver dice --config sqd_blocks.toml --hpc-target miyabi

The remaining arguments go to the solver's own script (``sqd_fulqrum.create_blocks``
or ``sqd_dice.create_blocks``). An SBD solver block is created with
``algorithms/closed_loop/sqd_de/create_blocks.py``; select it in a run with
``solver_block_ref = "sbd_solver_job/<name>"``.
"""

from __future__ import annotations

import sys

SOLVERS = ("fulqrum", "dice")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    solver = "fulqrum"
    if "--solver" in argv:
        index = argv.index("--solver")
        if index + 1 >= len(argv):
            raise SystemExit(f"--solver needs one of {', '.join(SOLVERS)}")
        solver = argv[index + 1]
        del argv[index : index + 2]
    if solver == "sbd":
        raise SystemExit(
            "Create the SBD solver block with algorithms/closed_loop/sqd_de/create_blocks.py, "
            "then set solver_block_ref = 'sbd_solver_job/<name>' in the flow parameters."
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


if __name__ == "__main__":
    main()
