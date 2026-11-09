"""SQD QRMI sampling through a real Slurm job, with QRMI itself replaced by fakes.

The workflow's ``quantum_sampling`` task body submits the upstream QRMI worker
through the blocks built by ``sqd-create-blocks``. The job compiles the
LUCJ circuit on the compute node with ``sqd_common.transpile`` and decodes the
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
        algorithms/open_loop/sqd/tests/test_sqd_fulqrum_qrmi_slurm_integration.py

Environment:
    SLURM_INTEGRATION=1        enable the test
    SQD_QRMI_SLURM_ROOT        shared job root (default /data/sqd-fulqrum-qrmi-integration)
    SQD_QRMI_SLURM_PYTHON      compute-node Python, e.g. ~/venv/sqd-node/bin/python
                               (default: this interpreter)
    FULQRUM_SLURM_PARTITION    default from the example TOML
    FULQRUM_SLURM_TIMEOUT      seconds, default 600
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import shutil
import sys
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
from qiskit.primitives.containers import BitArray  # noqa: E402
from sqd_common import (  # noqa: E402
    lucj,
    qrmi_blocks,  # noqa: E402
    sampling,
)
from sqd_common.advanced import AdvancedSettings  # noqa: E402
from sqd_common.block_config import load_block_config  # noqa: E402
from sqd_common.flow_params import (  # noqa: E402
    CircuitParameters,
    SamplingParameters,
)
from sqd_common.qrmi import TRANSPILER  # noqa: E402
from sqd_fulqrum import create_blocks  # noqa: E402

SQD = Path(__file__).resolve().parents[1]
REPO = SQD.parents[2]
EXAMPLE = SQD / "sqd_blocks.example.toml"
ROOT = Path(os.getenv("SQD_QRMI_SLURM_ROOT", "/data/sqd-fulqrum-qrmi-integration"))
NODE_PYTHON = os.getenv("SQD_QRMI_SLURM_PYTHON", sys.executable)
TIMEOUT = float(os.getenv("FULQRUM_SLURM_TIMEOUT", "600"))
RESOURCE = "fake_qpu"
SHOTS = 200
HEAVY = ("ffsim", "prefect_qiskit", "qcsc_prefect_fulqrum", "qcsc_workflow_utility", "pyscf")
LOGGER = logging.getLogger("sqd_common.qrmi_slurm_test")

FAKE_QRMI = """\
import atexit, json, sys
from pathlib import Path
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap


class _Resource:
    def resource_id(self):
        return "fake_qpu"


class QRMIService:
    def resources(self):
        return [_Resource()]


def _record_modules():
    import os

    loaded = sorted({name.split(".")[0] for name in sys.modules})
    Path("node_modules.json").write_text(json.dumps(loaded))
    # The QRMI settings the job sees: the block's plain ones arrive through sbatch.
    qrmi_env = {k: v for k, v in os.environ.items() if "_QRMI_" in k}
    Path("node_qrmi_env.json").write_text(json.dumps(qrmi_env))


def get_backend(resource):
    # Record the worker's modules at exit, after it has imported the compilation hook.
    atexit.register(_record_modules)
    return GenericBackendV2(
        num_qubits=8,
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=CouplingMap.from_line(8, bidirectional=True),
        seed=42,
    )
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
print("2026-01-01 00:00:00,000 INFO Task ID: fake-task-1", flush=True)
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
    (site / "qrmi" / "primitives" / "ibm.py").write_text("from qrmi import get_backend\n")
    runner = root / "fake_task_runner"
    runner.write_text(f"#!{NODE_PYTHON}\n{FAKE_TASK_RUNNER}")
    runner.chmod(0o755)
    return site, runner


@pytest.fixture
def qrmi_runtime(monkeypatch):
    blocks: dict[str, Any] = {}

    def fake_save(self, name, overwrite=False):
        blocks[name] = self
        return self

    async def fake_artifact(*, table, key, **kwargs):
        blocks.setdefault("_artifacts", []).append((key, table))

    def loader(name):
        async def _load(block_name):
            return blocks[block_name]

        return type(name, (), {"load": staticmethod(_load)})

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
    for module in (chem, lucj, sampling):
        monkeypatch.setattr(module, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(sampling, "build_lucj_circuit", lucj.build_lucj_circuit.fn)
    monkeypatch.setattr(sampling, "create_table_artifact", fake_artifact)

    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    site, runner = _write_fakes(ROOT)

    config = load_block_config(EXAMPLE)
    config["work_dir"] = str(ROOT)
    if os.getenv("FULQRUM_SLURM_PARTITION"):
        config["queue"] = os.environ["FULQRUM_SLURM_PARTITION"]
    config["python_executable"] = NODE_PYTHON
    config["qrmi"] = {
        "resource": RESOURCE,
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
    block_kwargs = create_blocks.resolve_settings(args, config)
    qrmi = qrmi_blocks.resolve_qrmi_settings(args, config, block_kwargs)
    qrmi_blocks.create_qrmi_blocks(qrmi)

    # The QRMI job requests the block's backend; its HPC profile names no QPU.
    assert blocks[qrmi["hpc_profile_block_name"]].slurm_qpu is None
    blocks[qrmi["command_block_name"]].default_args += ["--task-runner", str(runner)]
    # The QrmiRuntimeBlock that create_blocks and qcsc-qrmi login save, loaded as the flow
    # does: the instance's CRN must reach the job, and no API key.
    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

    settings = {
        **qrmi_blocks.qrmi_runtime_settings(qrmi),
        "resource": RESOURCE,
        "instance": "testbed (open)",
        "instances": {"testbed (open)": "crn:v1:testbed"},
        "instance_backends": {"testbed (open)": [RESOURCE]},
    }
    block = QrmiRuntimeBlock(**settings)
    block.check_backend_access(live=False)
    # The fakes stand in for the QPU: request no QPU and run the fake task_runner.
    block_runtime = dataclasses.replace(block.get_runtime(), request_qpu=False)

    class Loaded:
        resource = block.resource

        def get_runtime(self, prepare=None):
            return block_runtime

    async def load(name):
        assert name == "qrmi-runtime"
        return Loaded()

    monkeypatch.setattr(sampling, "load_qrmi_block", load)
    return block_runtime


def test_qrmi_sampling_compiles_on_the_node_through_slurm(qrmi_runtime, caplog):
    elec_props = chem.compute_molecular_integrals_from_geometry.fn(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"
    )
    params = AdvancedSettings(sabre_layout_trials=20, sabre_swap_trials=4, sabre_max_iterations=2)
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        bool_array = asyncio.run(
            sampling.quantum_sampling.fn(
                lucj_params=lucj.get_lucj_params.fn(CircuitParameters(), elec_props, "heavy-hex"),
                advanced=params,
                sampling_params=SamplingParameters(shots=SHOTS),
            )
        )
    bit_array = BitArray.from_bool_array(bool_array)

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
    (pub,) = report["pubs"]
    assert (
        pub["seed_transpiler"]
        == lucj.circuit_transpile_options(params, np.random.default_rng(24))["seed_transpiler"]
    )
    script = (job_dir / "sampler.slurm").read_text()
    assert "#SBATCH --qpu" not in script
    # The block's CRN reaches the job through sbatch's environment, not the script.
    qrmi_env = json.loads((job_dir / "node_qrmi_env.json").read_text())
    assert qrmi_env[f"{RESOURCE}_QRMI_IBM_QCS_SERVICE_CRN"] == "crn:v1:testbed"
    assert "crn:v1:testbed" not in script
    assert not any("APIKEY" in key for key in qrmi_env)
    loaded = set(json.loads((job_dir / "node_modules.json").read_text()))
    assert "sqd_common" in loaded
    assert loaded.isdisjoint(HEAVY), sorted(loaded & set(HEAVY))
    assert "Compiled on the compute node:" in caplog.text
    # task_runner's log reaches the job's output through the worker.
    assert "Task ID: fake-task-1" in (job_dir / "output.out").read_text()
    assert not (job_dir / "task_ended.txt").exists()
