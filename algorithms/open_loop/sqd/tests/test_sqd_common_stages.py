"""The SQD stage tasks of sqd_common, called through ``.fn`` with Prefect mocked."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

for _mod in ("ffsim", "prefect_qiskit", "pydantic_numpy", "pyscf", "qiskit_addon_sqd"):
    pytest.importorskip(_mod)

import qcsc_workflow_utility.chem as chem  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from qiskit.primitives import PrimitiveResult, SamplerPubResult  # noqa: E402
from qiskit.primitives.containers import BitArray, DataBin  # noqa: E402
from qiskit_addon_sqd.fermion import SCIResult, solve_fermion  # noqa: E402
from qiskit_ibm_runtime import RuntimeEncoder  # noqa: E402
from sqd_common import advanced as advanced_mod  # noqa: E402
from sqd_common import flow_params, lucj, postprocess, recovery, sampling, topology  # noqa: E402

LOGGER = logging.getLogger("sqd_common.test")
LIH = "Li 0 0 0; H 0 0 1.6"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for module in (chem, lucj, sampling, postprocess, topology):
        monkeypatch.setattr(module, "get_run_logger", lambda: LOGGER)
    artifacts = []

    async def fake_artifact(**kwargs):
        artifacts.append(kwargs)

    monkeypatch.setattr(sampling, "create_table_artifact", fake_artifact)
    monkeypatch.setattr(postprocess, "create_table_artifact", fake_artifact)
    return artifacts


@pytest.fixture(scope="module")
def lih():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(chem, "get_run_logger", lambda: LOGGER)
        return chem.compute_molecular_integrals_from_geometry.fn(atom=LIH, basis="sto-3g")


def sample_result(reset: bool) -> PrimitiveResult:
    data = {"meas": BitArray.from_samples(["0101", "0110", "1001", "1010"], num_bits=4)}
    if reset:
        data["test"] = BitArray.from_samples(["0000", "0001", "0000", "0010"], num_bits=4)
    return PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})], metadata={})


# ---------------------------------------------------------------- parameters


def _flow_parameters(**sampling_kwargs):
    return flow_params.FlowParameters(
        molecule={"atom": LIH, "basis": "sto-3g"},
        sampling=sampling_kwargs,
        sqd={"subspace_dim": 64, "num_batches": 2, "max_iterations": 2},
    )


def test_flow_parameters_defaults():
    params = _flow_parameters()
    assert params.molecule.spin_sq == 0.0
    assert params.circuit.compress is True and params.circuit.compress_maxiter == 20
    assert params.sampling.quantum_source == "real-device"
    assert params.sampling.shots == 10_000
    assert params.sampling.runtime_block_name == "qrmi-runtime"
    assert params.sampling.runner_name == "ibm-runner"
    assert params.advanced_block_name == "sqd-advanced-settings"
    assert params.solver_block_ref is None
    defaults = advanced_mod.AdvancedSettings()
    assert defaults.solver_block_ref == "fulqrum_solver_job/sqd-fulqrum-solver"
    assert (defaults.ccsd_maxiter, defaults.use_reset_mitigation) == (10, False)


def test_flow_parameters_parse_fcidump_molecule():
    params = flow_params.FlowParameters.model_validate(
        {
            "molecule": {"fcidump_file": "/data/n2.fcidump", "spin_sq": None},
            "sqd": {"subspace_dim": 100, "num_batches": 1, "max_iterations": 1},
            "solver_block_ref": "sbd_solver_job/s",
        }
    )
    assert isinstance(params.molecule, flow_params.FCIDumpFile)
    assert params.molecule.spin_sq is None


def test_removed_solver_block_name_points_to_the_advanced_block():
    with pytest.raises(ValidationError, match="Solver Block in the Advanced SQD Settings"):
        flow_params.FlowParameters.model_validate(
            {
                "molecule": {"fcidump_file": "/data/n2.fcidump"},
                "sqd": {"subspace_dim": 1, "num_batches": 1, "max_iterations": 1},
                "solver_block_name": "my-solver",
            }
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"qiskit_runtime_job_id": "j", "qrmi_result_dir": "/d"}, "at most one"),
        ({"quantum_source": "random", "qiskit_runtime_job_id": "j"}, "random"),
        ({"quantum_source": "random", "qrmi_result_dir": "/d"}, "random"),
        # An empty value is a hidden key of an old run form and is ignored; a set one fails.
        ({"qrmi_runtime_config": {"resource": "ibm_kingston"}}, "runtime_block_name"),
        ({"sampler_options": {"params": {"shots": 5}}}, "only the Shots parameter"),
    ],
)
def test_sampling_parameters_reject_conflicts(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        flow_params.SamplingParameters(**kwargs)


def test_only_the_molecule_is_needed():
    params = flow_params.FlowParameters(molecule={"atom": LIH, "basis": "sto-3g"})
    sqd = params.sqd
    assert (sqd.subspace_dim, sqd.num_batches, sqd.max_iterations) == (90_000, 4, 5)


def test_transpiler_options_come_from_the_advanced_settings():
    options = lucj.circuit_transpile_options(advanced_mod.AdvancedSettings(), 5)
    assert options == {
        "optimization_level": 3,
        "seed_transpiler": 5,
        "sabre_max_iterations": 8,
        "sabre_layout_trials": 10000,
        "sabre_swap_trials": 10,
    }
    custom = advanced_mod.AdvancedSettings(sabre_layout_trials=7, optimization_level=1)
    assert lucj.circuit_transpile_options(custom, 5)["sabre_layout_trials"] == 7


@pytest.mark.parametrize(
    "model, data, match",
    [
        (flow_params.CircuitParameters, {"max_ab_pairs": 2}, "max_ab_pairs was removed"),
        (flow_params.CircuitParameters, {"use_reset_mitigation": True}, "Advanced SQD Settings"),
        (
            flow_params.CircuitParameters,
            {"configure_transpiler": True, "transpiler": {"sabre_layout_trials": 7}},
            "circuit.configure_transpiler moved to the Advanced SQD Settings",
        ),
        (flow_params.CircuitParameters, {"sabre_layout_trials": 10}, "Advanced SQD Settings"),
        (flow_params.FCIDumpFile, {"fcidump_file": "/x", "ccsd_maxiter": 2}, "molecule.ccsd"),
    ],
)
def test_moved_fields_point_to_the_advanced_block(model, data, match):
    with pytest.raises(ValidationError, match=match):
        model(**data)


def test_stale_run_form_keys_left_at_their_defaults_are_ignored():
    """The Prefect UI keeps removed fields hidden in a form filled before an upgrade."""
    circuit = flow_params.CircuitParameters.model_validate(
        {
            "compress": True,
            "transpiler": {"sabre_layout_trials": 10000, "optimization_level": 3},
            "max_ab_pairs": None,
            "optimization_level": 3,
            "sabre_layout_trials": 10000,
            "sabre_swap_trials": 10,
            "sabre_max_iterations": 8,
            "configure_transpiler": False,
            "use_reset_mitigation": False,
        }
    )
    assert circuit.compress is True
    molecule = flow_params.MoleculeGeometry(atom=LIH, ccsd_maxiter=4)
    assert "ccsd_maxiter" not in molecule.model_dump()
    params = flow_params.FlowParameters.model_validate(
        {
            "molecule": {"fcidump_file": "/x"},
            "solver_block_ref": "fulqrum_solver_job/sqd-fulqrum-solver",
        }
    )
    # The old default of the form never overrides the block's Solver Block.
    assert params.solver_block_ref is None
    sampling = flow_params.SamplingParameters(qrmi_runtime_config=None, sampler_options={})
    assert sampling.quantum_source == "real-device"


def _form_order(schema, model):
    properties = schema["definitions"][model]["properties"]
    return sorted(properties, key=lambda name: properties[name]["position"])


def test_run_form_follows_the_qiskit_patterns_order():
    """Every field has a position; the Prefect form would otherwise sort alphabetically."""
    from prefect.utilities.callables import parameter_schema
    from sqd_common.main import sqd

    schema = parameter_schema(sqd).model_dump_for_openapi()
    for model in (
        "FlowParameters",
        "MoleculeGeometry",
        "FCIDumpFile",
        "CircuitParameters",
        "SamplingParameters",
        "SQDParameters",
    ):
        properties = schema["definitions"][model]["properties"]
        positions = [field["position"] for field in properties.values()]
        assert sorted(positions) == list(range(len(positions))), model
    assert _form_order(schema, "FlowParameters") == [
        "molecule",
        "circuit",
        "sampling",
        "sqd",
        "advanced_block_name",
        "solver_block_ref",
    ]
    assert _form_order(schema, "CircuitParameters")[-1] == "qpu_topology"
    assert _form_order(schema, "SamplingParameters")[:2] == ["quantum_source", "shots"]
    sqd_schema = schema["definitions"]["SQDParameters"]
    # Defaulted, but marked required so the form highlights them.
    assert {"subspace_dim", "num_batches", "max_iterations"} <= set(sqd_schema["required"])
    assert sqd_schema["properties"]["subspace_dim"]["default"] == 90_000


# ---------------------------------------------------------------- Optimize


def test_interaction_pairs_heavy_hex():
    aa, ab = lucj.interaction_pairs(10)
    assert aa == [(p, p + 1) for p in range(9)]
    assert ab == [(0, 0), (4, 4), (8, 8)]
    # At most five alpha-beta pairs, from orbital 0.
    assert lucj.interaction_pairs(40)[1] == [(p, p) for p in range(0, 20, 4)]
    assert lucj.MAX_AB_PAIRS == 5


@pytest.mark.parametrize("qpu_topology", ["heavy-hex", "square"])
def test_get_lucj_params_matches_ffsim(lih, qpu_topology):
    import ffsim

    # Without compression, the parameters are ffsim's own truncated factorization.
    circuit_params = flow_params.CircuitParameters(n_lucj_layers=2, compress=False)
    params = lucj.get_lucj_params.fn(circuit_params, lih, qpu_topology)
    ref = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        lih.t2, n_reps=2, interaction_pairs=lucj.interaction_pairs(lih.num_orbitals, qpu_topology)
    )
    assert params.norb == lih.num_orbitals and params.nelec == lih.num_electrons
    assert params.qpu_topology == qpu_topology
    np.testing.assert_allclose(params.diag_coulomb_mats, ref.diag_coulomb_mats)
    np.testing.assert_allclose(params.orbital_rotations, ref.orbital_rotations)
    assert params.diag_coulomb_mats.shape == (2, 2, lih.num_orbitals, lih.num_orbitals)


def test_get_lucj_params_compression_logs_iterations(lih, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        params = lucj.get_lucj_params.fn(
            flow_params.CircuitParameters(compress=True, compress_maxiter=3), lih, "heavy-hex"
        )
    assert "Compression iteration 1" in caplog.text
    assert params.orbital_rotations.dtype == np.complex128


def test_lucj_parameters_round_trip_through_pickle(lih):
    import pickle

    params = lucj.get_lucj_params.fn(flow_params.CircuitParameters(), lih, "square")
    restored = pickle.loads(pickle.dumps(params))
    np.testing.assert_array_equal(restored.diag_coulomb_mats, params.diag_coulomb_mats)
    assert restored.qpu_topology == "square"


def test_get_lucj_params_cache_key_depends_on_the_topology(lih):
    """The persisted Optimize result of one topology is never reused for another."""
    from prefect.cache_policies import DEFAULT
    from prefect.utilities.callables import get_call_parameters

    task = lucj.get_lucj_params
    assert task.cache_policy == DEFAULT
    # The default policy reads only the task and the flow run ID from the run context.
    context = SimpleNamespace(task=task, task_run=SimpleNamespace(flow_run_id="run"))

    def key(qpu_topology):
        args = (flow_params.CircuitParameters(), lih, qpu_topology)
        inputs = get_call_parameters(task.fn, args, {})
        return task.cache_policy.compute_key(context, inputs, flow_parameters={})

    assert key("heavy-hex") == key("heavy-hex")
    assert key("heavy-hex") != key("square")


@pytest.mark.parametrize("reset", [False, True])
def test_build_lucj_circuit_registers(lih, reset):
    params = lucj.get_lucj_params.fn(flow_params.CircuitParameters(), lih, "heavy-hex")
    circuit = lucj.build_lucj_circuit.fn(params, use_reset_mitigation=reset)
    names = [reg.name for reg in circuit.cregs]
    assert names == (["meas", "test"] if reset else ["meas"])
    assert circuit.num_qubits == 2 * lih.num_orbitals


def test_stage_tasks_carry_pattern_tags():
    from qcsc_workflow_utility import (
        compute_molecular_integrals_from_fcidump,
        compute_molecular_integrals_from_geometry,
    )
    from sqd_common import flow

    assert lucj.get_lucj_params.tags == {"optimize"}
    for task in (
        sampling.quantum_sampling,
        sampling.load_runtime_job,
        sampling.load_qrmi_result,
        lucj.build_lucj_circuit,
        lucj.transpile_lucj_circuit,
    ):
        assert task.tags == {"execute"}, task.name
    assert postprocess.sqd_loop.tags == {"post-process"}
    assert (
        flow.compute_molecular_integrals_from_geometry is compute_molecular_integrals_from_geometry
    )
    for task in (
        compute_molecular_integrals_from_geometry,
        compute_molecular_integrals_from_fcidump,
    ):
        assert task.with_options(tags=["map"]).tags == {"map"}


def test_resumable_stages_persist_and_sqd_loop_never_caches():
    from prefect.cache_policies import NO_CACHE

    for task in (
        lucj.get_lucj_params,
        sampling.quantum_sampling,
        sampling.load_runtime_job,
        sampling.load_qrmi_result,
    ):
        assert task.persist_result is True, task.name
        assert task.result_serializer == "compressed/pickle", task.name
    assert postprocess.sqd_loop.cache_policy is NO_CACHE


# ---------------------------------------------------------------- Execute


@pytest.mark.parametrize("reset", [False, True])
def test_extract_bool_array(reset):
    bool_array, shots = sampling.extract_bool_array(sample_result(reset), reset)
    assert shots == 4
    expected = ["0101", "1001"] if reset else ["0101", "0110", "1001", "1010"]
    assert BitArray.from_bool_array(bool_array).get_bitstrings() == expected


def test_extract_bool_array_requires_test_register_for_reset():
    with pytest.raises(ValueError, match="no 'test' register"):
        sampling.extract_bool_array(sample_result(False), True)


def test_random_bool_array_is_deterministic():
    params = flow_params.SamplingParameters(quantum_source="random", shots=50, random_seed=7)
    a = sampling.random_bool_array(params, 3)
    b = sampling.random_bool_array(params, 3)
    assert a.shape == (50, 6) and a.dtype == bool
    np.testing.assert_array_equal(a, b)


class _FakeQrmiRuntime:
    """Records the hooks the flow sets and returns a result with scheduler metadata."""

    def __init__(self, scheduler, calls):
        self.scheduler, self.resource, self.calls = scheduler, "test_heron", calls

    def replace(self, **changes):
        self.calls["hooks"] = changes
        return self

    async def sampler(self, *, sampler_pubs, shots):
        self.calls["pubs"], self.calls["shots"] = sampler_pubs, shots
        result = sample_result(False)
        if self.scheduler == "iqp":
            result.metadata["qcsc_qrmi"] = {"job_id": "d7", "pubs": [{"isa_ops": {"cz": 3}}]}
        else:
            result.metadata["qcsc_slurm"] = {"job_id": "41", "work_dir": "/jobs/abc"}
        return result


@pytest.mark.parametrize("scheduler", ["iqp", "slurm"])
def test_sampling_uses_the_qrmi_block_with_the_sqd_hooks(monkeypatch, lih, offline, scheduler):
    from sqd_common import qrmi

    calls = {}
    runtime = _FakeQrmiRuntime(scheduler, calls)

    async def load(name):
        calls["block"] = name
        return runtime

    monkeypatch.setattr(sampling, "_load_qrmi_runtime", load)
    monkeypatch.setattr(qrmi.dataclasses, "replace", lambda obj, **changes: obj.replace(**changes))
    monkeypatch.setattr(sampling, "build_lucj_circuit", lucj.build_lucj_circuit.fn)
    advanced = advanced_mod.AdvancedSettings(sabre_layout_trials=7)
    params = lucj.get_lucj_params.fn(flow_params.CircuitParameters(), lih, "square")
    sampling_params = flow_params.SamplingParameters(shots=4, runtime_block_name="my-qrmi")
    bool_array = asyncio.run(sampling.quantum_sampling.fn(params, advanced, sampling_params))

    assert calls["block"] == "my-qrmi" and calls["shots"] == 4
    hooks = calls["hooks"]
    assert hooks["transpiler"] == qrmi.TRANSPILER and hooks["transpile_config"] == {}
    expected = lucj.circuit_transpile_options(advanced, np.random.default_rng(24))
    assert hooks["prepare"].keywords == {"transpile_config": expected}
    assert expected["sabre_layout_trials"] == 7
    assert bool_array.shape == (4, 4)
    row = offline[0]["table"][0]
    assert row["source"] == f"qrmi-{scheduler}" and row["resource"] == "test_heron"
    assert row["qpu_topology"] == "square"
    if scheduler == "iqp":
        assert row["qiskit_runtime_job_id"] == "d7" and row["qrmi_result_dir"] is None
    else:
        assert row["qrmi_result_dir"] == "/jobs/abc" and row["slurm_job_id"] == "41"


def test_missing_runtime_block_explains_random_fallback(monkeypatch, lih):
    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

    async def missing(name, prepare=None):
        raise ValueError(f"Unable to find block document named {name}")

    monkeypatch.setattr(QrmiRuntimeBlock, "aload_runtime", staticmethod(missing))
    params = lucj.get_lucj_params.fn(flow_params.CircuitParameters(), lih, "heavy-hex")
    with pytest.raises(RuntimeError, match="quantum_source='random'"):
        asyncio.run(
            sampling.quantum_sampling.fn(
                params, advanced_mod.AdvancedSettings(), flow_params.SamplingParameters()
            )
        )


@pytest.mark.parametrize("reset", [False, True])
def test_load_runtime_job_reads_the_job_result(monkeypatch, offline, reset):
    seen = []

    class Client:
        async def get_primitive_result(self, job_id):
            seen.append(job_id)
            return sample_result(reset)

    runtime = SimpleNamespace(
        resource_name="ibm_test", credentials=SimpleNamespace(get_client=lambda: Client())
    )

    async def load(name):
        assert name == "ibm-runner"
        return runtime

    monkeypatch.setattr(sampling.QuantumRuntime, "load", staticmethod(load))
    bool_array = asyncio.run(
        sampling.load_runtime_job.fn("d123", "ibm-runner", use_reset_mitigation=reset)
    )
    assert seen == ["d123"]
    assert len(bool_array) == (2 if reset else 4)
    row = offline[0]["table"][0]
    assert row["qiskit_runtime_job_id"] == "d123" and row["shots"] == 4
    assert row["shots_kept"] == len(bool_array)
    # A reload builds no circuit, so it records no topology.
    assert row["qpu_topology"] is None


def _qrmi_job_dir(tmp_path, reset, schema=2):
    registers = {"meas": 4, **({"test": 4} if reset else {})}
    if schema == 2:
        registers = [registers]
    request = {
        "schema_version": schema,
        "resource": "test_heron",
        "shots": 4,
        "registers": registers,
    }
    (tmp_path / "sampler_request.json").write_text(json.dumps(request))
    (tmp_path / "result.json").write_text(json.dumps(sample_result(reset), cls=RuntimeEncoder))
    return tmp_path


@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize("reset", [False, True])
def test_load_qrmi_result_decodes_the_job_directory(tmp_path, offline, reset, schema):
    job_dir = _qrmi_job_dir(tmp_path, reset, schema)
    bool_array = asyncio.run(sampling.load_qrmi_result.fn(str(job_dir), use_reset_mitigation=reset))
    assert BitArray.from_bool_array(bool_array).get_bitstrings() == (
        ["0101", "1001"] if reset else ["0101", "0110", "1001", "1010"]
    )
    row = offline[0]["table"][0]
    assert row["qrmi_result_dir"] == str(job_dir) and row["resource"] == "test_heron"


def test_load_qrmi_result_rejects_reset_on_a_job_without_test_register(tmp_path):
    job_dir = _qrmi_job_dir(tmp_path, reset=False)
    with pytest.raises(ValueError, match="no 'test' register"):
        asyncio.run(sampling.load_qrmi_result.fn(str(job_dir), use_reset_mitigation=True))


def test_load_qrmi_result_requires_result_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="sampler_request.json"):
        asyncio.run(sampling.load_qrmi_result.fn(str(tmp_path), use_reset_mitigation=False))


# ---------------------------------------------------------------- Post-process


class _SolveFermion:
    # A fulqrum block in the resume key; subclasses that stand in for it inherit it.
    _block_type_slug = "fulqrum_solver_job"

    def __init__(self):
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        energy, state, occ, _ = solve_fermion(
            kwargs["ci_strings"],
            kwargs["one_body_tensor"],
            kwargs["two_body_tensor"],
            open_shell=True,
        )
        return SCIResult(energy=energy, sci_state=state, orbital_occupancies=occ)


def test_sqd_loop_diagonalizes_with_the_solver_block(lih, offline, tmp_path):
    solver = _SolveFermion()
    solver.root_dir = str(tmp_path)
    bool_array = sampling.random_bool_array(
        flow_params.SamplingParameters(quantum_source="random", shots=3000), lih.num_orbitals
    )
    energy = asyncio.run(
        postprocess.sqd_loop.fn(
            bool_array,
            lih,
            flow_params.SQDParameters(subspace_dim=64, num_batches=2, max_iterations=2),
            solver,
            "my-artifact",
        )
    )
    assert len(solver.calls) == 4
    for call in solver.calls:
        assert call["one_body_tensor"] is lih.one_body_tensor
        assert call["spin_sq"] == lih.spin_sq
        assert call["norb"] == 6 and call["nelec"] == (2, 2)
    artifact, checkpoint = offline
    assert artifact["key"] == "my-artifact"
    assert checkpoint["key"] == postprocess.CHECKPOINT_ARTIFACT_KEY
    assert checkpoint["table"][0]["checkpoint"].startswith(str(tmp_path))
    assert [row["round"] for row in artifact["table"]] == [1, 1, 2, 2]
    assert energy == pytest.approx(min(row["energy"] for row in artifact["table"]))
    # LiH/STO-3G: SQD is variational, a few mHa above FCI (-7.882 Ha) at most here.
    assert -7.8825 < energy < -7.80


# ---------------------------------------------------------------- Post-process checkpoint


class _Crash(RuntimeError):
    pass


@pytest.fixture
def ckpt_run(monkeypatch, tmp_path, lih, offline):
    """Call sqd_loop as the flow run ``run_id`` with a solver block rooted in tmp_path."""
    bool_array = sampling.random_bool_array(
        flow_params.SamplingParameters(quantum_source="random", shots=3000), lih.num_orbitals
    )

    def run(run_id, solver=None, molecule=lih, **sqd):
        monkeypatch.setattr(postprocess, "flow_run", SimpleNamespace(get_id=lambda: run_id))
        solver = solver or _SolveFermion()
        solver.root_dir = str(tmp_path)
        offline.clear()
        params = flow_params.SQDParameters(**{"subspace_dim": 64, "num_batches": 2, **sqd})
        energy = asyncio.run(
            postprocess.sqd_loop.fn(bool_array, molecule, params, solver, "recovery")
        )
        return energy, solver, {a["key"]: a["table"] for a in offline}

    run.root = tmp_path / postprocess.CHECKPOINT_SUBDIR
    return run


class _CrashingSolve(_SolveFermion):
    def __init__(self, crash_at):
        super().__init__()
        self.crash_at = crash_at

    async def run(self, **kwargs):
        if len(self.calls) == self.crash_at:
            raise _Crash("walltime")
        return await super().run(**kwargs)


def _ckpt(path):
    with np.load(path / "ckpt_t00_w0.npz") as raw:
        return int(raw["next_step"]), float(raw["best_energy"])


def test_retry_resumes_after_the_last_completed_round(ckpt_run):
    # A crashed solver job fails its whole round, naming the crash as the cause.
    with pytest.raises(recovery.RoundFailedError, match="iteration 3 failed") as info:
        ckpt_run("run-a", _CrashingSolve(crash_at=4), max_iterations=4)
    assert isinstance(info.value.__cause__, _Crash)
    assert _ckpt(ckpt_run.root / "run-a")[0] == 2

    energy, solver, artifacts = ckpt_run("run-a", max_iterations=4)
    assert len(solver.calls) == 4  # rounds 3 and 4 only
    assert [row["round"] for row in artifacts["recovery"]] == [3, 3, 4, 4]
    assert _ckpt(ckpt_run.root / "run-a") == (4, pytest.approx(energy))
    (row,) = artifacts[postprocess.CHECKPOINT_ARTIFACT_KEY]
    assert row["completed_iterations"] == 4 and row["resumed_from"] is None
    assert -7.8825 < energy < -7.80


def test_resume_checkpoint_continues_another_run_with_more_rounds(ckpt_run):
    first, _, artifacts = ckpt_run("run-a", max_iterations=2)
    given = artifacts[postprocess.CHECKPOINT_ARTIFACT_KEY][0]["checkpoint"]
    before = (ckpt_run.root / "run-a" / "ckpt_t00_w0.npz").read_bytes()

    energy, solver, artifacts = ckpt_run("run-b", max_iterations=5, resume_checkpoint=given)
    assert len(solver.calls) == 6  # rounds 3 to 5
    assert [row["round"] for row in artifacts["recovery"]] == [3, 3, 4, 4, 5, 5]
    assert energy <= first
    assert (ckpt_run.root / "run-a" / "ckpt_t00_w0.npz").read_bytes() == before
    assert _ckpt(ckpt_run.root / "run-b")[0] == 5
    assert artifacts[postprocess.CHECKPOINT_ARTIFACT_KEY][0]["resumed_from"] == given


def test_resume_checkpoint_accepts_the_folder(ckpt_run):
    ckpt_run("run-a", max_iterations=1)
    _, solver, _ = ckpt_run(
        "run-b", max_iterations=2, resume_checkpoint=str(ckpt_run.root / "run-a")
    )
    assert len(solver.calls) == 2


def test_a_retry_before_the_first_new_round_still_resumes(ckpt_run):
    ckpt_run("run-a", max_iterations=2)
    given = str(ckpt_run.root / "run-a")
    with pytest.raises(recovery.RoundFailedError):
        ckpt_run("run-b", _CrashingSolve(crash_at=0), max_iterations=4, resume_checkpoint=given)
    assert _ckpt(ckpt_run.root / "run-b")[0] == 2
    # The retry of run-b finds its own copy, even without resume_checkpoint.
    _, solver, _ = ckpt_run("run-b", max_iterations=4)
    assert len(solver.calls) == 4


@pytest.mark.parametrize(
    "change, match",
    [(dict(subspace_dim=100), "subspace_dim"), (dict(num_batches=3), "num_batches")],
)
def test_resume_checkpoint_of_other_settings_is_refused_before_any_solve(ckpt_run, change, match):
    ckpt_run("run-a", max_iterations=1)
    solver = _SolveFermion()
    with pytest.raises(ValueError, match=match):
        ckpt_run(
            "run-b",
            solver,
            max_iterations=3,
            resume_checkpoint=str(ckpt_run.root / "run-a"),
            **change,
        )
    assert solver.calls == []


class _OtherSolver(_SolveFermion):
    _block_type_slug = "sbd_solver_job"


class _SameTypeOtherBlock(_SolveFermion):
    """Another block of the same type, for example with a longer walltime."""


def test_solver_kind_is_the_block_type_slug():
    from qcsc_prefect_fulqrum import FulqrumSolverJob

    assert postprocess.solver_kind(FulqrumSolverJob.model_construct()) == "fulqrum_solver_job"
    assert postprocess.solver_kind(_OtherSolver()) == "sbd_solver_job"


def test_resume_checkpoint_of_another_solver_is_refused_before_any_solve(ckpt_run):
    ckpt_run("run-a", max_iterations=1)
    solver = _OtherSolver()
    with pytest.raises(ValueError, match="other solver.*solver type"):
        ckpt_run("run-b", solver, max_iterations=3, resume_checkpoint=str(ckpt_run.root / "run-a"))
    assert solver.calls == []


def test_retry_with_another_solver_starts_over(ckpt_run, caplog):
    ckpt_run("run-a", max_iterations=1)
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        _, solver, _ = ckpt_run("run-a", _OtherSolver(), max_iterations=2)
    assert len(solver.calls) == 4  # rounds 1 and 2
    assert "other solver" in caplog.text
    assert any(p.name.startswith("run-a.prev-") for p in ckpt_run.root.iterdir())


def test_resume_with_another_block_of_the_same_type_continues(ckpt_run):
    ckpt_run("run-a", max_iterations=1)
    _, solver, _ = ckpt_run(
        "run-b",
        _SameTypeOtherBlock(),
        max_iterations=2,
        resume_checkpoint=str(ckpt_run.root / "run-a"),
    )
    assert len(solver.calls) == 2  # round 2 only


def test_resume_checkpoint_of_another_molecule_is_refused(ckpt_run, lih):
    ckpt_run("run-a", max_iterations=1)
    other = lih.model_copy(update={"nuclear_repulsion_energy": lih.nuclear_repulsion_energy + 1})
    with pytest.raises(ValueError, match="Hamiltonian"):
        ckpt_run(
            "run-b",
            molecule=other,
            max_iterations=2,
            resume_checkpoint=str(ckpt_run.root / "run-a"),
        )


def test_resume_checkpoint_without_source_json_is_refused(ckpt_run):
    ckpt_run("run-a", max_iterations=1)
    (ckpt_run.root / "run-a" / "source.json").unlink()
    with pytest.raises(ValueError, match="no source.json"):
        ckpt_run("run-b", max_iterations=2, resume_checkpoint=str(ckpt_run.root / "run-a"))


def test_own_checkpoint_of_other_settings_is_moved_aside(ckpt_run):
    ckpt_run("run-a", max_iterations=1)
    _, solver, _ = ckpt_run("run-a", max_iterations=1, subspace_dim=100)
    assert len(solver.calls) == 2  # started again at round 1
    assert [p.name.startswith("run-a.prev-") for p in ckpt_run.root.iterdir()].count(True) == 1


def test_checkpoint_off_writes_nothing(ckpt_run):
    _, _, artifacts = ckpt_run("run-a", max_iterations=1, checkpoint=False)
    assert not ckpt_run.root.exists()
    assert postprocess.CHECKPOINT_ARTIFACT_KEY not in artifacts
