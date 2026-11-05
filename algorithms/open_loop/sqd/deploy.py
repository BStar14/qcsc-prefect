"""Serve the ``sqd`` deployment from a source checkout (same as ``sqd-deploy``).

Usage (from the repository root)::

    python algorithms/open_loop/sqd/deploy.py

Each run selects its solver with ``solver_block_ref`` in the flow parameters.
"""

from sqd_common.main import deploy

if __name__ == "__main__":
    deploy()
