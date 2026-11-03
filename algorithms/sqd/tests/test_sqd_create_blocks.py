"""The [runtime] config table builds a valid QrmiRuntimeBlock without secrets in the file."""

from __future__ import annotations

import pytest
from sqd_dice.create_blocks import _runtime_block

QRMI = {
    "QRMI_IBM_QCS_ENDPOINT": "https://quantum.example/api/v1",
    "QRMI_IBM_QCS_IAM_ENDPOINT": "https://iam.example",
    "QRMI_IBM_QCS_SERVICE_CRN": "crn:v1:example",
}


def test_iqp_block_reads_the_api_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("QRMI_IBM_QCS_IAM_APIKEY", "secret-api-key")
    block = _runtime_block({"resource": "ibm_test", "qrmi": dict(QRMI)})

    runtime = block.get_runtime()
    assert runtime.scheduler == "iqp"
    assert runtime.resource == "ibm_test"
    assert runtime.resource_config == {**QRMI, "QRMI_IBM_QCS_IAM_APIKEY": "secret-api-key"}
    assert "secret-api-key" not in repr(block)


def test_iqp_block_requires_the_api_key_in_the_environment(monkeypatch):
    monkeypatch.delenv("QRMI_IBM_QCS_IAM_APIKEY", raising=False)
    with pytest.raises(RuntimeError, match="QRMI_IBM_QCS_IAM_APIKEY"):
        _runtime_block({"resource": "ibm_test", "qrmi": dict(QRMI)})


def test_api_key_in_the_config_file_is_rejected(monkeypatch):
    monkeypatch.setenv("QRMI_IBM_QCS_IAM_APIKEY", "secret-api-key")
    qrmi = {**QRMI, "QRMI_IBM_QCS_IAM_APIKEY": "written-in-file"}
    with pytest.raises(RuntimeError, match="Do not write QRMI_IBM_QCS_IAM_APIKEY"):
        _runtime_block({"resource": "ibm_test", "qrmi": qrmi})


@pytest.mark.parametrize("key", ["transpiler", "transpile_config", "resource_config"])
def test_settings_owned_by_the_flow_are_rejected(monkeypatch, key):
    monkeypatch.setenv("QRMI_IBM_QCS_IAM_APIKEY", "secret-api-key")
    with pytest.raises(RuntimeError, match=key):
        _runtime_block({"resource": "ibm_test", "qrmi": dict(QRMI), key: {}})


def test_slurm_block_uses_block_names_and_no_qrmi_settings(tmp_path):
    block = _runtime_block(
        {
            "scheduler": "slurm",
            "resource": "test_heron",
            "root_dir": str(tmp_path),
            "command_block_name": "c",
            "execution_profile_block_name": "e",
            "hpc_profile_block_name": "h",
        }
    )
    runtime = block.get_runtime()
    assert runtime.scheduler == "slurm"
    assert runtime.root_dir == tmp_path
    assert runtime.resource_config is None

    with pytest.raises(RuntimeError, match=r"\[runtime.qrmi\] is only used"):
        _runtime_block({"scheduler": "slurm", "resource": "test_heron", "qrmi": dict(QRMI)})


def test_invalid_settings_fail_before_saving():
    with pytest.raises(ValueError, match="requires root_dir"):
        _runtime_block({"scheduler": "slurm", "resource": "test_heron"})
