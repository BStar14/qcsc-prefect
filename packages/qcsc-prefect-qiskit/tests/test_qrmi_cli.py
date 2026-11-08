"""qcsc-qrmi login/refresh/list/check and the qrmi_config.json it writes."""

from __future__ import annotations

import json
import stat

import pytest
from qcsc_prefect.integrations.qiskit.qrmi import cli, config_file
from qcsc_prefect.integrations.qiskit.qrmi.block import QrmiRuntimeBlock

KEY = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-QWERT"
FOUND = [
    config_file.Instance("main-instance", "crn:v1:main", "premium", ["ibm_kingston", "ibm_fez"]),
    config_file.Instance("open-instance", "crn:v1:open", "open", ["ibm_kingston"]),
]


class Prompt(cli._Prompt):
    """Answers in order; records everything printed."""

    def __init__(self, answers, secret=KEY):
        self.answers = list(answers)
        self.printed = []
        super().__init__(
            ask=lambda question: self.answers.pop(0) if self.answers else "",
            secret=lambda question: secret,
            out=self.printed.append,
        )

    @property
    def text(self):
        return "\n".join(self.printed)


@pytest.fixture
def blocks(monkeypatch):
    saved = {}

    def save(block_name, settings, labels, backends):
        # The real block class, so that the test sees the dropdowns; only saving is fake.
        from qcsc_prefect.integrations.qiskit.qrmi.block import qrmi_runtime_block_class

        saved[block_name] = qrmi_runtime_block_class(labels, backends)(**settings)

    monkeypatch.setattr(cli, "save_block", save)
    monkeypatch.setattr(cli, "load_block", lambda name: saved.get(name))
    return saved


@pytest.fixture
def discover(monkeypatch):
    calls = []

    def fake(api_key):
        calls.append(api_key)
        return [config_file.Instance(i.name, i.crn, i.plan, list(i.backends)) for i in FOUND]

    monkeypatch.setattr(cli, "discover_instances", fake)
    return calls


def _login(tmp_path, answers, *extra):
    path = tmp_path / "qrmi_config.json"
    args = cli._parser().parse_args(["login", "--config", str(path), *extra])
    prompt = Prompt(answers)
    assert cli.cmd_login(args, prompt) == 0
    return path, prompt


def test_login_writes_one_key_one_instance_list_one_backend_list(
    tmp_path, blocks, discover, capsys
):
    # Keep all instances, default instance 1, default backend 2 (ibm_kingston).
    path, prompt = _login(tmp_path, ["", "1", "2"])
    assert discover == [KEY]
    data = json.loads(path.read_text())
    assert data["environment"] == {"QRMI_IBM_QCS_IAM_APIKEY": KEY}
    assert data["default_instance"] == "main-instance"
    assert [(i["name"], i["plan"], i["crn"]) for i in data["instances"]] == [
        ("main-instance", "premium", "crn:v1:main"),
        ("open-instance", "open", "crn:v1:open"),
    ]
    # Each backend once, whichever instances have it; each entry holds only the key.
    assert [entry["name"] for entry in data["resources"]] == ["ibm_fez", "ibm_kingston"]
    for entry in data["resources"]:
        assert entry["type"] == "ibm-quantum-compute-service"
        assert entry["environment"] == {"QRMI_IBM_QCS_IAM_APIKEY": KEY}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # The key is never printed.
    assert KEY not in prompt.text and KEY not in capsys.readouterr().out

    block = blocks["qrmi-runtime"]
    assert (block.instance, block.resource) == ("main-instance (premium)", "ibm_kingston")
    assert block.instances == {
        "main-instance (premium)": "crn:v1:main",
        "open-instance (open)": "crn:v1:open",
    }
    assert block.instance_backends["open-instance (open)"] == ["ibm_kingston"]
    assert KEY not in json.dumps(block.model_dump())
    schema = type(block).model_json_schema()["properties"]
    assert schema["instance"]["enum"] == list(block.instances)
    assert schema["resource"]["enum"] == ["ibm_fez", "ibm_kingston"]
    # The TOML lines to keep the choice, and the Slurm install note.
    assert 'resource = "ibm_kingston"' in prompt.text
    assert 'instance = "main-instance"' in prompt.text
    assert "plugstack.conf" in prompt.text
    assert "Run sqd-create-blocks next" in prompt.text


def test_login_keeps_other_resources_and_rewrites_in_place(tmp_path, blocks, discover):
    path = tmp_path / "qrmi_config.json"
    pasqal = {"name": "FRESNEL", "type": "pasqal-cloud", "environment": {"TOKEN": "t"}}
    path.write_text(json.dumps({"resources": [pasqal], "site_note": "kept"}))
    inode = path.stat().st_ino
    _login(tmp_path, ["2", "", ""])
    data = json.loads(path.read_text())
    # Only open-instance kept; Pasqal and the unknown key stay as they were.
    assert [i["name"] for i in data["instances"]] == ["open-instance"]
    assert data["resources"][-1] == pasqal and data["site_note"] == "kept"
    assert path.stat().st_ino == inode  # a single-file container mount still sees it


def test_login_keeps_the_block_slurm_settings(tmp_path, blocks, discover):
    blocks["qrmi-runtime"] = QrmiRuntimeBlock(
        resource="test_heron",
        root_dir="/data/sqd/qrmi",
        command_block_name="cmd-x",
        execution_profile_block_name="exec-x",
        hpc_profile_block_name="hpc-x",
        timeout_seconds=99,
    )
    _, prompt = _login(tmp_path, ["", "2", ""])
    block = blocks["qrmi-runtime"]
    assert (block.command_block_name, block.timeout_seconds) == ("cmd-x", 99)
    assert (block.instance, block.resource) == ("open-instance (open)", "ibm_kingston")
    assert "Run sqd-create-blocks next" not in prompt.text


def test_login_without_discovery_asks_for_each_instance(tmp_path, blocks, discover):
    answers = [
        "lab", "premium", "crn:v1:lab", "ibm_a, ibm_b",
        "class", "open", "crn:v1:class", "ibm_b",
        "",  # no more instances
        "", "2", "1",  # keep all, default class (open), backend ibm_b
    ]  # fmt: skip
    path, _ = _login(tmp_path, answers, "--no-discover")
    assert discover == []
    data = json.loads(path.read_text())
    assert [i["name"] for i in data["instances"]] == ["lab", "class"]
    assert [entry["name"] for entry in data["resources"]] == ["ibm_a", "ibm_b"]
    assert blocks["qrmi-runtime"].instance == "class (open)"


def test_login_reads_the_key_from_a_named_variable(tmp_path, blocks, discover, monkeypatch):
    monkeypatch.setenv("MY_IBM_KEY", "from-env")
    _login(tmp_path, ["", "", ""], "--api-key-env", "MY_IBM_KEY")
    assert discover == ["from-env"]
    monkeypatch.delenv("MY_IBM_KEY")
    with pytest.raises(SystemExit, match="MY_IBM_KEY is empty"):
        _login(tmp_path, [], "--api-key-env", "MY_IBM_KEY")


def test_refresh_updates_the_lists_and_keeps_the_choice(tmp_path, blocks, discover, monkeypatch):
    path, _ = _login(tmp_path, ["", "2", ""])  # open-instance, ibm_kingston
    FOUND_LATER = [
        config_file.Instance("main-instance", "crn:v1:main", "premium", ["ibm_fez", "ibm_new"]),
        config_file.Instance("open-instance", "crn:v1:open", "open", ["ibm_kingston", "ibm_new"]),
        config_file.Instance("other", "crn:v1:other", "open", ["ibm_x"]),
    ]
    monkeypatch.setattr(cli, "discover_instances", lambda key: FOUND_LATER)
    args = cli._parser().parse_args(["refresh", "--config", str(path)])
    prompt = Prompt([])
    assert cli.cmd_refresh(args, prompt) == 0
    data = json.loads(path.read_text())
    # A new instance the login did not keep stays out; new backends come in.
    assert [i["name"] for i in data["instances"]] == ["main-instance", "open-instance"]
    assert "ibm_new" in [entry["name"] for entry in data["resources"]]
    block = blocks["qrmi-runtime"]
    assert (block.instance, block.resource) == ("open-instance (open)", "ibm_kingston")
    assert block.instance_backends["open-instance (open)"] == ["ibm_kingston", "ibm_new"]


def test_list_never_prints_the_key(tmp_path, blocks, discover):
    path, _ = _login(tmp_path, ["", "", ""])
    prompt = Prompt([])
    cli.cmd_list(cli._parser().parse_args(["list", "--config", str(path)]), prompt)
    assert "API key: set" in prompt.text and KEY not in prompt.text
    assert "main-instance (premium) (default): ibm_fez, ibm_kingston" in prompt.text
    prompt = Prompt([])
    cli.cmd_list(cli._parser().parse_args(["list", "--config", str(path), "--json"]), prompt)
    assert json.loads(prompt.text)["api_key"] == "set" and KEY not in prompt.text


def test_check_reports_a_backend_the_instance_cannot_use(tmp_path, blocks, discover):
    from qcsc_prefect.integrations.qiskit.qrmi.block import BackendAccessError

    _login(tmp_path, ["", "2", ""])
    blocks["qrmi-runtime"] = blocks["qrmi-runtime"].model_copy(update={"resource": "ibm_fez"})
    args = cli._parser().parse_args(["check", "--offline"])
    with pytest.raises(BackendAccessError, match="available to: main-instance"):
        cli.cmd_check(args, Prompt([]))


def test_a_hand_written_config_still_gives_its_key(tmp_path):
    path = tmp_path / "qrmi_config.json"
    entry = {"QRMI_IBM_QCS_IAM_APIKEY": "hand", "QRMI_IBM_QCS_SERVICE_CRN": "crn"}
    path.write_text(
        json.dumps(
            {
                "resources": [
                    {"name": "ibm_a", "type": "ibm-quantum-compute-service", "environment": entry}
                ]
            }
        )
    )
    config = config_file.read_config(path)
    assert config.api_key == "hand" and config.instances == [] and config.other_resources == []
    assert "hand" not in repr(config)
