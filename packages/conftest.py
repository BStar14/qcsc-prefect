from __future__ import annotations

import sys
from pathlib import Path


def _add_src_path(path: Path) -> None:
    if path.exists():
        sys.path.insert(0, str(path))


_ROOT = Path(__file__).resolve().parent
_add_src_path(_ROOT / "qcsc-prefect-core" / "src")
_add_src_path(_ROOT / "qcsc-prefect-adapters" / "src")
_add_src_path(_ROOT / "qcsc-prefect-executor" / "src")
_add_src_path(_ROOT / "qcsc-prefect-qiskit" / "src")
_add_src_path(_ROOT / "qcsc-prefect-fulqrum" / "src")
