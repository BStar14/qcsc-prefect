"""qcsc-qrmi login/refresh/list/check and the qrmi_config.json it writes."""

from __future__ import annotations

import json
import logging
import re
import stat

import pytest
from qcsc_prefect.integrations.qiskit.qrmi import cli, config_file
from qcsc_prefect.integrations.qiskit.qrmi.block import QrmiRuntimeBlock

KEY = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-QWERT"
US = "crn:v1:bluemix:public:quantum-computing:us-east:a/acct:1111aaaa::"
EU = "crn:v1:bluemix:public:quantum-computing:eu-de:a/acct:2222bbbb::"
OPEN = "crn:v1:bluemix:public:quantum-computing:us-east:a/acct:3333cccc::"
# Two instances with the same name and plan, in two regions, and an open one.
FOUND = [
    config_file.Instance("Demo fleet", US, "premium", ["ibm_kingston", "ibm_fez"]),
    config_file.Instance("Demo fleet", EU, "premium", ["ibm_aachen", "ibm_berlin"]),
    config_file.Instance("open-instance", OPEN, "open", ["ibm_kingston"]),
]
US_LABEL = "Demo fleet (premium, us-east)"
EU_LABEL = "Demo fleet (premium, eu-de)"
OPEN_LABEL = "open-instance (open, us-east)"


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

    def fake(api_key, out=print):
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
    assert data["default_instance"] == US
    assert [i["crn"] for i in data["instances"]] == [US, EU, OPEN]
    # Each backend once, whichever instances have it; each entry holds only the key.
    names = [entry["name"] for entry in data["resources"]]
    assert names == ["ibm_aachen", "ibm_berlin", "ibm_fez", "ibm_kingston"]
    for entry in data["resources"]:
        assert entry["type"] == "ibm-quantum-compute-service"
        assert entry["environment"] == {"QRMI_IBM_QCS_IAM_APIKEY": KEY}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # The key is never printed.
    assert KEY not in prompt.text and KEY not in capsys.readouterr().out

    block = blocks["qrmi-runtime"]
    assert (block.instance, block.resource) == (US_LABEL, "ibm_kingston")
    assert block.instances == {US_LABEL: US, EU_LABEL: EU, OPEN_LABEL: OPEN}
    assert block.instance_backends[OPEN_LABEL] == ["ibm_kingston"]
    assert KEY not in json.dumps(block.settings())
    schema = type(block).model_json_schema()["properties"]
    assert schema["instance"]["enum"] == [US_LABEL, EU_LABEL, OPEN_LABEL]
    assert schema["resource"]["enum"] == names
    # The TOML lines to keep the choice, and the Slurm install note.
    assert 'resource = "ibm_kingston"' in prompt.text
    assert f'instance = "{US_LABEL}"' in prompt.text
    assert "plugstack.conf" in prompt.text
    assert "Run sqd-create-blocks next" in prompt.text


def test_same_name_instances_in_two_regions_stay_apart(tmp_path, blocks, discover):
    # Default instance 2 is the eu-de one; its backends are listed, not the us-east ones.
    path, prompt = _login(tmp_path, ["", "2", "ibm_berlin"])
    assert f"1. {US_LABEL}" in prompt.text and f"2. {EU_LABEL}" in prompt.text
    assert "Backends of Demo fleet (premium, eu-de):" in prompt.text
    data = json.loads(path.read_text())
    assert data["default_instance"] == EU
    block = blocks["qrmi-runtime"]
    assert (block.instance, block.resource) == (EU_LABEL, "ibm_berlin")
    # The eu-de CRN and its regional endpoint reach the job.
    settings = block.qrmi_settings()
    assert settings["QRMI_IBM_QCS_SERVICE_CRN"] == EU
    assert settings["QRMI_IBM_QCS_ENDPOINT"] == "https://eu-de.quantum.cloud.ibm.com/api/v1"


def test_a_backend_of_another_instance_is_explained(tmp_path, blocks, discover):
    # ibm_kingston belongs to the us-east instances, not to the eu-de default.
    _, prompt = _login(tmp_path, ["", "2", "ibm_kingston", "1"])
    assert f"ibm_kingston is not available to {EU_LABEL}." in prompt.text
    assert f"It is available to: {US_LABEL}, {OPEN_LABEL}." in prompt.text
    assert blocks["qrmi-runtime"].resource == "ibm_aachen"


def test_instances_are_chosen_by_number_only(tmp_path, blocks, discover):
    # A name is ambiguous ("Demo fleet" is two instances), so it is asked again.
    path, prompt = _login(tmp_path, ["Demo fleet", "2,3", "", ""])
    assert "Enter numbers from 1 to 3" in prompt.text
    data = json.loads(path.read_text())
    assert [i["crn"] for i in data["instances"]] == [EU, OPEN]


def test_identical_labels_get_a_crn_suffix(tmp_path, blocks, monkeypatch):
    twin = "crn:v1:bluemix:public:quantum-computing:us-east:a/acct:9999dddd::"
    found = [
        config_file.Instance("Demo fleet", US, "premium", ["ibm_fez"]),
        config_file.Instance("Demo fleet", twin, "premium", ["ibm_fez"]),
    ]
    monkeypatch.setattr(cli, "discover_instances", lambda key, out=print: found)
    _, prompt = _login(tmp_path, ["", "2", ""])
    labels = list(blocks["qrmi-runtime"].instances)
    assert labels == [
        "Demo fleet [11aaaa] (premium, us-east)",
        "Demo fleet [99dddd] (premium, us-east)",
    ]
    assert blocks["qrmi-runtime"].instances[blocks["qrmi-runtime"].instance] == twin


def test_login_keeps_other_resources_and_rewrites_in_place(tmp_path, blocks, discover):
    path = tmp_path / "qrmi_config.json"
    pasqal = {"name": "FRESNEL", "type": "pasqal-cloud", "environment": {"TOKEN": "t"}}
    path.write_text(json.dumps({"resources": [pasqal], "site_note": "kept"}))
    inode = path.stat().st_ino
    _login(tmp_path, ["3", "", ""])
    data = json.loads(path.read_text())
    # Only open-instance kept; Pasqal and the unknown key stay as they were.
    assert [i["crn"] for i in data["instances"]] == [OPEN]
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
    _, prompt = _login(tmp_path, ["", "3", ""])
    block = blocks["qrmi-runtime"]
    assert (block.command_block_name, block.timeout_seconds) == ("cmd-x", 99)
    assert (block.instance, block.resource) == (OPEN_LABEL, "ibm_kingston")
    assert "Run sqd-create-blocks next" not in prompt.text


def test_login_without_discovery_asks_for_each_instance(tmp_path, blocks, discover):
    answers = [
        "lab", "premium", "crn:v1:x:x:x:us-east:a/1:lab::", "ibm_a, ibm_b",
        "class", "open", "crn:v1:x:x:x:eu-de:a/1:class::", "ibm_b",
        "",  # no more instances
        "", "2", "1",  # keep all, default class, backend ibm_b
    ]  # fmt: skip
    path, _ = _login(tmp_path, answers, "--no-discover")
    assert discover == []
    data = json.loads(path.read_text())
    assert [i["name"] for i in data["instances"]] == ["lab", "class"]
    assert [entry["name"] for entry in data["resources"]] == ["ibm_a", "ibm_b"]
    assert blocks["qrmi-runtime"].instance == "class (open, eu-de)"


def test_login_reads_the_key_from_a_named_variable(tmp_path, blocks, discover, monkeypatch):
    monkeypatch.setenv("MY_IBM_KEY", "from-env")
    _login(tmp_path, ["", "", ""], "--api-key-env", "MY_IBM_KEY")
    assert discover == ["from-env"]
    monkeypatch.delenv("MY_IBM_KEY")
    with pytest.raises(SystemExit, match="MY_IBM_KEY is empty"):
        _login(tmp_path, [], "--api-key-env", "MY_IBM_KEY")


def test_refresh_updates_the_lists_and_keeps_the_choice(tmp_path, blocks, discover, monkeypatch):
    path, _ = _login(tmp_path, ["", "2", "2"])  # eu-de instance, ibm_berlin
    later = [
        config_file.Instance("Demo fleet", US, "premium", ["ibm_fez", "ibm_new"]),
        config_file.Instance("Demo fleet", EU, "premium", ["ibm_berlin", "ibm_new"]),
        config_file.Instance("other", OPEN.replace("3333", "4444"), "open", ["ibm_x"]),
    ]
    monkeypatch.setattr(cli, "discover_instances", lambda key, out=print: later)
    args = cli._parser().parse_args(["refresh", "--config", str(path)])
    assert cli.cmd_refresh(args, Prompt([])) == 0
    data = json.loads(path.read_text())
    # An instance the login did not keep stays out; new backends come in.
    assert [i["crn"] for i in data["instances"]] == [US, EU]
    assert "ibm_new" in [entry["name"] for entry in data["resources"]]
    block = blocks["qrmi-runtime"]
    # The same-name us-east instance is not mistaken for the chosen eu-de one.
    assert (block.instance, block.resource) == (EU_LABEL, "ibm_berlin")
    assert block.instance_backends[EU_LABEL] == ["ibm_berlin", "ibm_new"]


def test_list_never_prints_the_key(tmp_path, blocks, discover):
    path, _ = _login(tmp_path, ["", "", ""])
    prompt = Prompt([])
    cli.cmd_list(cli._parser().parse_args(["list", "--config", str(path)]), prompt)
    assert "API key: set" in prompt.text and KEY not in prompt.text
    assert f"{US_LABEL} (default): ibm_fez, ibm_kingston" in prompt.text
    prompt = Prompt([])
    cli.cmd_list(cli._parser().parse_args(["list", "--config", str(path), "--json"]), prompt)
    assert json.loads(prompt.text)["api_key"] == "set" and KEY not in prompt.text


def test_check_reports_a_backend_the_instance_cannot_use(tmp_path, blocks, discover):
    from qcsc_prefect.integrations.qiskit.qrmi.block import BackendAccessError

    _login(tmp_path, ["", "3", ""])
    blocks["qrmi-runtime"] = blocks["qrmi-runtime"].model_copy(update={"resource": "ibm_fez"})
    args = cli._parser().parse_args(["check", "--offline"])
    with pytest.raises(BackendAccessError, match=re.escape(f"available to: {US_LABEL}")):
        cli.cmd_check(args, Prompt([]))


def test_ibm_runtime_warnings_are_hidden_and_errors_are_one_line(monkeypatch, caplog):
    class Service:
        def __init__(self, **kwargs):
            logging.getLogger("qiskit_ibm_runtime.qiskit_runtime_service").warning(
                "Loading account with the given token. A saved account will not be used."
            )

        def instances(self):
            return [{"name": "Demo fleet", "crn": US, "plan": "premium"}]

        def backends(self, instance):
            raise ConnectionError("HTTPSConnectionPool(host='x'): Max retries exceeded")

    import qiskit_ibm_runtime

    monkeypatch.delenv("QISKIT_IBM_RUNTIME_LOG_LEVEL", raising=False)
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", Service)
    printed = []
    with caplog.at_level(logging.WARNING):
        assert cli.discover_instances(KEY, printed.append) == []
    assert "Loading account" not in caplog.text
    assert printed == [
        f"  Skipped {US_LABEL}: IBM Quantum could not be reached. Check the network "
        "connection of this host."
    ]

    class Rejected(Service):
        def instances(self):
            raise RuntimeError(
                "Unable to retrieve instances. Please check that you are using a valid API token."
            )

    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", Rejected)
    with pytest.raises(cli.DiscoveryError, match="^IBM Quantum rejected the API key"):
        cli.discover_instances(KEY)
    with pytest.raises(SystemExit, match="NOPE is empty or unset"):
        cli.main(["login", "--config", "/nonexistent/x.json", "--api-key-env", "NOPE"])


def test_a_hand_written_config_still_gives_its_key(tmp_path):
    path = tmp_path / "qrmi_config.json"
    entry = {"QRMI_IBM_QCS_IAM_APIKEY": "hand", "QRMI_IBM_QCS_SERVICE_CRN": "crn"}
    resource = {"name": "ibm_a", "type": "ibm-quantum-compute-service", "environment": entry}
    path.write_text(json.dumps({"resources": [resource]}))
    config = config_file.read_config(path)
    assert config.api_key == "hand" and config.instances == [] and config.other_resources == []
    assert "hand" not in repr(config)
