"""Serve an SQD deployment from a source checkout.

Usage (from the repository root)::

    python algorithms/open_loop/sqd/deploy.py fulqrum   # same as sqd-fulqrum-deploy
    python algorithms/open_loop/sqd/deploy.py dice      # same as sqd-deploy
"""

import sys


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"fulqrum", "dice"}:
        raise SystemExit("Usage: python algorithms/open_loop/sqd/deploy.py {fulqrum|dice}")

    if sys.argv[1] == "fulqrum":
        from sqd_fulqrum.main import deploy
    else:
        from sqd_dice.main import deploy
    deploy()


if __name__ == "__main__":
    main()
