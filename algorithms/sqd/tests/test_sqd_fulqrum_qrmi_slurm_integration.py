"""SQD QRMI sampling through a real Slurm job, with QRMI itself replaced by fakes.

The workflow's ``sample_bitstrings`` task body submits the upstream QRMI worker
through the blocks built by ``sqd-fulqrum-create-blocks``. The job compiles the
LUCJ circuit on the compute node with ``sqd_fulqrum.transpile`` and decodes the
result on the host. Only the QPU side is faked, so no QPU or QRMI endpoint is
contacted:

- the HPC profile requests no ``--qpu``; the job environment instead carries
  the Batch mode and acquisition token that SPANK would set;
- a fake ``qrmi`` package returns a fixed 8-qubit Target;
- a fake ``task_runner`` samples the compiled ``isa.qpy`` with Qiskit's
  noiseless ``StatevectorSampler``.

No Prefect server is used; blocks are kept in memory. Run it from the repository
root on a Slurm submit host. The checkout and the job root must be visible at the
same paths on the compute nodes:

    SLURM_INTEGRATION=1 python -m pytest \
        algorithms/sqd/tests/test_sqd_fulqrum_qrmi_slurm_integration.py

Environment:
    SLURM_INTEGRATION=1        enable the test
    SQD_QRMI_SLURM_ROOT        shared job root (default /data/sqd-fulqrum-qrmi-integration)
    SQD_QRMI_SLURM_PYTHON      compute-node Python with Prefect, Qiskit and
                               qiskit-ibm-runtime (default: this interpreter)
    FULQRUM_SLURM_PARTITION    default from the example TOML
    FULQRUM_SLURM_TIMEOUT      seconds, default 600
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.slurm_integration

if os.getenv("SLURM_INTEGRATION", "").strip().lower() not in {"1", "true", "yes", "on"}:
    pytest.skip("Set SLURM_INTEGRATION=1 to run the Slurm round trip.", allow_module_level=True)
if shutil.which("sbatch") is None:
    pytest.skip("sbatch not found. Run on a Slurm login node.", allow_module_level=True)
for _mod in ("ffsim", "prefect_qiskit", "pydantic_numpy", "pyscf"):
    pytest.importorskip(_mod)
pytest.importorskip("qcsc_prefect.integrations.qiskit.qrmi")

import qcsc_workflow_utility.chem as chem  # noqa: E402
from qcsc_prefect.integrations.qiskit.qrmi import runtime as runtime_mod  # noqa: E402
from qcsc_prefect_executor import from_blocks  # noqa: E402
from qcsc_prefect_executor.slurm import run as slurm_run  # noqa: E402
from qcsc_prefect_fulqrum import block_utils  # noqa: E402
from sqd_fulqrum import create_blocks  # noqa: E402
from sqd_fulqrum import main as workflow  # noqa: E402
from sqd_fulqrum.qrmi import TRANSPILER  # noqa: E402

SQD = Path(__file__).resolve().parents[1]
REPO = SQD.parents[1]
EXAMPLE = SQD / "sqd_fulqrum_blocks.slurm.example.toml"
ROOT = Path(os.getenv("SQD_QRMI_SLURM_ROOT", "/data/sqd-fulqrum-qrmi-integration"))
NODE_PYTHON = os.getenv("SQD_QRMI_SLURM_PYTHON", sys.executable)
TIMEOUT = float(os.getenv("FULQRUM_SLURM_TIMEOUT", "600"))
RESOURCE = "fake_qpu"
SHOTS = 200
HEAVY = ("ffsim", "prefect_qiskit", "qcsc_prefect_fulqrum", "qcsc_workflow_utility", "pyscf")
LOGGER = logging.getLogger("sqd_fulqrum.qrmi_slurm_test")

FAKE_QRMI = """\
import json, sys
from pathlib import Path
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap


class _Resource:
    def resource_id(self):
        return "fake_qpu"


class QRMIService:
    def resources(self):
        return [_Resource()]


def get_target(resource):
    # The worker imports the compilation hook before it asks for the Target.
    loaded = sorted({name.split(".")[0] for name in sys.modules})
    Path("node_modules.json").write_text(json.dumps(loaded))
    return GenericBackendV2(
        num_qubits=8,
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=CouplingMap.from_line(8, bidirectional=True),
        seed=42,
    ).target
"""

FAKE_TASK_RUNNER = """\
import json, sys
from pathlib import Path
from qiskit import qpy
from qiskit.primitives import StatevectorSampler
from qiskit_ibm_runtime import RuntimeEncoder

resource, source, destination = sys.argv[1:]
assert resource == "fake_qpu"
payload = json.loads(Path(source).read_text())
assert payload["program_id"] == "sampler"
shots = payload["parameters"]["shots"]
with open("isa.qpy", "rb") as stream:
    (isa,) = qpy.load(stream)
result = StatevectorSampler(seed=7).run([(isa,)], shots=shots).result()
Path(destination).write_text(json.dumps(result, cls=RuntimeEncoder))
"""


class _LoggerStub:
    def info(self, msg, *args):
        LOGGER.info(msg, *args)

    def warning(self, msg, *args):
        LOGGER.warning(msg, *args)

    error = debug = info


def _write_fakes(root: Path) -> tuple[Path, Path]:
    site = root / "fake_site"
    (site / "qrmi" / "primitives").mkdir(parents=True)
    (site / "qrmi" / "__init__.py").write_text(FAKE_QRMI)
    (site / "qrmi" / "primitives" / "__init__.py").write_text("")
    (site / "qrmi" / "primitives" / "ibm.py").write_text("from qrmi import get_target\n")
    runner = root / "fake_task_runner"
    runner.write_text(f"#!{NODE_PYTHON}\n{FAKE_TASK_RUNNER}")
    runner.chmod(0o755)
    return site, runner


@pytest.fixture
def qrmi_config(monkeypatch):
    blocks: dict[str, Any] = {}

    def fake_save(self, name, overwrite=False):
        blocks[name] = self
        return self

    async def fake_artifact(*, table, key):
        blocks.setdefault("_artifacts", []).append((key, table))

    def loader(name):
        async def _load(block_name):
            return blocks[block_name]

        return type(name, (), {"load": staticmethod(_load)})

    async def sampler_options(name, default=None):
        return {"params": {"shots": SHOTS}}

    for cls in (
        block_utils.CommandBlock,
        block_utils.ExecutionProfileBlock,
        block_utils.HPCProfileBlock,
    ):
        monkeypatch.setattr(cls, "save", fake_save, raising=False)
    for name in ("CommandBlock", "ExecutionProfileBlock", "HPCProfileBlock"):
        monkeypatch.setattr(from_blocks, name, loader(name))
    monkeypatch.setattr(slurm_run, "get_run_logger", lambda: _LoggerStub())
    monkeypatch.setattr(slurm_run, "create_table_artifact", fake_artifact)
    monkeypatch.setattr(runtime_mod, "get_run_logger", lambda: _LoggerStub())
    monkeypatch.setattr(runtime_mod, "_sample_on_slurm", runtime_mod._sample_on_slurm.fn)
    monkeypatch.setattr(workflow, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(chem, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(workflow, "create_ansatz_circuits", workflow.create_ansatz_circuits.fn)
    monkeypatch.setattr(workflow.Variable, "get", sampler_options)

    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    site, runner = _write_fakes(ROOT)

    with EXAMPLE.open("rb") as f:
        config = tomllib.load(f)
    config["work_dir"] = str(ROOT)
    if os.getenv("FULQRUM_SLURM_PARTITION"):
        config["queue"] = os.environ["FULQRUM_SLURM_PARTITION"]
    config["qrmi"] = {
        "resource": RESOURCE,
        "python_executable": NODE_PYTHON,
        "walltime": "00:10:00",
        "ompthreads": 2,
        "timeout_seconds": TIMEOUT,
        "watch_poll_interval": 1.0,
        "environments": {
            # Only the worker, the hook and the fake QRMI: no ffsim, fulqrum or workflow.
            "PYTHONPATH": os.pathsep.join(
                [str(site), str(REPO / "packages" / "qcsc-prefect-qiskit" / "src"), str(SQD)]
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            # What SPANK sets after it acquires the Batch for --qpu=fake_qpu.
            f"{RESOURCE}_QRMI_IBM_QCS_SESSION_MODE": "batch",
            f"{RESOURCE}_QRMI_JOB_ACQUISITION_TOKEN": "fake-token",
        },
    }
    args = create_blocks._parse_args([])
    block_kwargs, _, _ = create_blocks.resolve_settings(args, config)
    qrmi = create_blocks.resolve_qrmi_settings(args, config, block_kwargs)
    create_blocks.create_qrmi_blocks(qrmi)

    # The fakes stand in for the QPU: request no QPU and run the fake task_runner.
    assert blocks[qrmi["hpc_profile_block_name"]].slurm_qpu == RESOURCE
    blocks[qrmi["hpc_profile_block_name"]].slurm_qpu = None
    blocks[qrmi["command_block_name"]].default_args += ["--task-runner", str(runner)]
    return create_blocks.qrmi_runtime_config(qrmi)


def test_qrmi_sampling_compiles_on_the_node_through_slurm(qrmi_config, caplog):
    elec_props = chem.compute_molecular_integrals_from_geometry.fn(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"
    )
    params = workflow.CircuitParameters(
        sabre_layout_trials=20, sabre_swap_trials=4, sabre_max_iterations=2
    )
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        bit_array = asyncio.run(
            workflow.sample_bitstrings.fn(
                circuit_params=params,
                elec_props=elec_props,
                runner_name="unused",
                option_name="sampler_options",
                quantum_source="real-device",
                random_seed=24,
                qrmi_runtime_config=qrmi_config,
            )
        )

    norb = elec_props.num_orbitals
    nalpha, nbeta = elec_props.num_electrons
    assert bit_array.num_shots == SHOTS and bit_array.num_bits == 2 * norb
    # LUCJ conserves particle number per spin. A noiseless sample of a correctly
    # transported, compiled and read-out circuit has only these Hamming weights.
    for bitstring in bit_array.get_counts():
        assert bitstring[norb:].count("1") == nalpha  # alpha: low bits, right half
        assert bitstring[:norb].count("1") == nbeta

    (job_dir,) = [p for p in (ROOT / "qrmi").iterdir() if p.is_dir()]
    record = json.loads((job_dir / "slurm_job.json").read_text())
    assert (record["state"], record["exit_status"]) == ("COMPLETED", 0)
    report = json.loads((job_dir / "transpile_report.json").read_text())
    assert report["transpiler"] == TRANSPILER
    assert report["target_num_qubits"] == 8
    assert (
        report["seed_transpiler"]
        == workflow.circuit_transpile_options(params, np.random.default_rng(24))["seed_transpiler"]
    )
    assert "#SBATCH --qpu" not in (job_dir / "sampler.slurm").read_text()
    loaded = set(json.loads((job_dir / "node_modules.json").read_text()))
    assert "sqd_fulqrum" in loaded
    assert loaded.isdisjoint(HEAVY), sorted(loaded & set(HEAVY))
    assert "Compiled on the compute node:" in caplog.text
