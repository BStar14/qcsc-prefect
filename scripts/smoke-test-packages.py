#!/usr/bin/env python3
"""Import smoke tests for installed qcsc-prefect distributions."""

from __future__ import annotations

import importlib
import importlib.util

REQUIRED_MODULES = [
    "qcsc_prefect_core",
    "qcsc_prefect_blocks",
    "qcsc_prefect_adapters",
    "qcsc_prefect_executor",
]

OPTIONAL_MODULES = [
    ("qcsc-prefect-qiskit", "qcsc_prefect.integrations.qiskit"),
    ("qcsc-prefect-dice", "qcsc_prefect_dice"),
    ("qcsc-prefect-fulqrum", "qcsc_prefect_fulqrum"),
]


def import_required(module_name: str) -> None:
    importlib.import_module(module_name)
    print(f"Imported required module: {module_name}")


def import_optional(distribution_name: str, module_name: str) -> None:
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        spec = None

    if spec is None:
        print(f"Optional distribution not installed, skipping: {distribution_name}")
        return
    importlib.import_module(module_name)
    print(f"Imported optional module: {module_name}")


def main() -> int:
    for module_name in REQUIRED_MODULES:
        import_required(module_name)

    for distribution_name, module_name in OPTIONAL_MODULES:
        import_optional(distribution_name, module_name)

    print("qcsc-prefect package smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
