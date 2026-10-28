"""Make sqd_fulqrum and its local dependencies importable without installation."""

from __future__ import annotations

import sys
from pathlib import Path

_SQD = Path(__file__).resolve().parents[1]
_REPO = _SQD.parents[1]

for _path in (
    _REPO / "algorithms" / "qcsc_workflow_utility" / "src",
    _REPO / "packages" / "qcsc-prefect-core" / "src",
    _REPO / "packages" / "qcsc-prefect-adapters" / "src",
    _REPO / "packages" / "qcsc-prefect-blocks" / "src",
    _REPO / "packages" / "qcsc-prefect-executor" / "src",
    _REPO / "packages" / "qcsc-prefect-fulqrum" / "src",
    _REPO / "packages" / "qcsc-prefect-qiskit" / "src",
    _SQD,
):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
