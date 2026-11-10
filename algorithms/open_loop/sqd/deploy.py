"""Serve the ``sqd`` deployment from a source checkout (same as ``sqd-deploy``).

Usage (from the repository root)::

    python algorithms/open_loop/sqd/deploy.py

Each run uses the solver block of the Advanced SQD Settings block, which
``sqd-create-blocks`` saves; ``solver_block_ref`` overrides it for one run.
"""

from sqd_common.main import deploy

if __name__ == "__main__":
    deploy()
