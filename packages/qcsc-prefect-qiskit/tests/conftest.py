"""Shared fixtures of the qcsc-prefect-qiskit tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_user_qrmi_config(monkeypatch, tmp_path):
    """Keep the user's ~/.qiskit/qrmi_config.json, and its API key, out of the tests."""
    monkeypatch.setenv("QRMI_CONFIG", str(tmp_path / "no-qrmi_config.json"))
