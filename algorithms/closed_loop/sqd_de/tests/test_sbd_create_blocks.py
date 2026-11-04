"""The [runtime] config table builds a valid QrmiRuntimeBlock without secrets in the file."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "create_blocks.py"
QRMI = {
    "QRMI_IBM_QCS_ENDPOINT": "https://quantum.example/api/v1",
    "QRMI_IBM_QCS_IAM_ENDPOINT": "https://iam.example",
    "QRMI_IBM_QCS_SERVICE_CRN": "crn:v1:example",
}


@pytest.fixture(scope="module")
def create_blocks():
    spec = importlib.util.spec_from_file_location("sbd_create_blocks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_iqp_block_reads_the_api_key_from_the_environment(create_blocks, monkeypatch):
    monkeypatch.setenv("QRMI_IBM_QCS_IAM_APIKEY", "secret-api-key")
    block = create_blocks._runtime_block({"resource": "ibm_test", "qrmi": dict(QRMI)})

    runtime = block.get_runtime()
    assert runtime.scheduler == "iqp"
    assert runtime.resource == "ibm_test"
    assert runtime.resource_config == {**QRMI, "QRMI_IBM_QCS_IAM_APIKEY": "secret-api-key"}
    assert "secret-api-key" not in repr(block)


def test_iqp_block_requires_the_api_key_in_the_environment(create_blocks, monkeypatch):
    monkeypatch.delenv("QRMI_IBM_QCS_IAM_APIKEY", raising=False)
    with pytest.raises(RuntimeError, match="QRMI_IBM_QCS_IAM_APIKEY"):
        create_blocks._runtime_block({"resource": "ibm_test", "qrmi": dict(QRMI)})


def test_api_key_in_the_config_file_is_rejected(create_blocks, monkeypatch):
    monkeypatch.setenv("QRMI_IBM_QCS_IAM_APIKEY", "secret-api-key")
    qrmi = {**QRMI, "QRMI_IBM_QCS_IAM_APIKEY": "written-in-file"}
    with pytest.raises(RuntimeError, match="Do not write QRMI_IBM_QCS_IAM_APIKEY"):
        create_blocks._runtime_block({"resource": "ibm_test", "qrmi": qrmi})


@pytest.mark.parametrize("key", ["transpiler", "transpile_config", "resource_config"])
def test_settings_owned_by_the_flow_are_rejected(create_blocks, monkeypatch, key):
    monkeypatch.setenv("QRMI_IBM_QCS_IAM_APIKEY", "secret-api-key")
    with pytest.raises(RuntimeError, match=key):
        create_blocks._runtime_block({"resource": "ibm_test", "qrmi": dict(QRMI), key: {}})


def test_slurm_block_uses_block_names_and_no_qrmi_settings(create_blocks, tmp_path):
    block = create_blocks._runtime_block(
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
    assert runtime.resource_config is None

    with pytest.raises(RuntimeError, match=r"\[runtime.qrmi\] is only used"):
        create_blocks._runtime_block(
            {"scheduler": "slurm", "resource": "test_heron", "qrmi": dict(QRMI)}
        )


def test_invalid_settings_fail_before_saving(create_blocks):
    with pytest.raises(ValueError, match="requires root_dir"):
        create_blocks._runtime_block({"scheduler": "slurm", "resource": "test_heron"})


def test_example_configs_parse(create_blocks, monkeypatch):
    # The Miyabi example creates an IQP block; the Fugaku and Slurm examples (also used by
    # the QFw-SLURM tutorial) have the [runtime] tables commented out.
    monkeypatch.setenv("QRMI_IBM_QCS_IAM_APIKEY", "secret-api-key")
    examples = SCRIPT.parent
    miyabi = create_blocks._load_config_file(examples / "sbd_blocks.example.toml")
    assert create_blocks._runtime_block(miyabi["runtime"]).scheduler == "iqp"
    for name in ("sbd_blocks.fugaku.example.toml", "sbd_blocks.slurm.example.toml"):
        assert "runtime" not in create_blocks._load_config_file(examples / name)
