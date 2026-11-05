"""CCSD restarts in qcsc_workflow_utility.chem."""

from __future__ import annotations

import logging

import numpy as np
import pytest

for _mod in ("prefect", "pydantic_numpy", "pyscf"):
    pytest.importorskip(_mod)

import qcsc_workflow_utility.chem as chem  # noqa: E402

LIH = "Li 0 0 0; H 0 0 1.6"


@pytest.fixture(autouse=True)
def _offline_logger(monkeypatch):
    monkeypatch.setattr(chem, "get_run_logger", lambda: logging.getLogger("chem.test"))


@pytest.fixture
def kernel_calls(monkeypatch):
    calls = []
    original = chem.cc.CCSD

    def counting_ccsd(mf):
        mycc = original(mf)
        kernel = mycc.kernel

        def counted(*args, **kwargs):
            calls.append(args)
            return kernel(*args, **kwargs)

        mycc.kernel = counted
        return mycc

    monkeypatch.setattr(chem.cc, "CCSD", counting_ccsd)
    return calls


@pytest.mark.parametrize("ccsd_maxiter", [1, 4])
def test_ccsd_runs_once_then_restarts_from_amplitudes(kernel_calls, ccsd_maxiter):
    chem.compute_molecular_integrals_from_geometry.fn(
        atom=LIH, basis="sto-3g", ccsd_maxiter=ccsd_maxiter
    )
    assert len(kernel_calls) == ccsd_maxiter
    assert kernel_calls[0] == ()
    # Restarts pass the previous (t1, t2).
    assert all(len(args) == 2 for args in kernel_calls[1:])


def test_default_is_a_single_ccsd_run(kernel_calls):
    chem.compute_molecular_integrals_from_geometry.fn(atom=LIH, basis="sto-3g")
    assert len(kernel_calls) == 1


def test_restarts_converge_to_the_same_amplitudes():
    once = chem.compute_molecular_integrals_from_geometry.fn(atom=LIH, basis="sto-3g")
    four = chem.compute_molecular_integrals_from_geometry.fn(
        atom=LIH, basis="sto-3g", ccsd_maxiter=4
    )
    np.testing.assert_allclose(four.t2, once.t2, atol=1e-6)


def test_ccsd_maxiter_must_be_positive():
    with pytest.raises(ValueError, match="ccsd_maxiter"):
        chem.compute_molecular_integrals_from_geometry.fn(atom=LIH, basis="sto-3g", ccsd_maxiter=0)
