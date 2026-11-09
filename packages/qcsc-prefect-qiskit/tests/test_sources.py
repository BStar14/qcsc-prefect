"""QuantumSource and its standard sources, used directly without any application."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass

import pytest
from prefect.client.schemas.objects import BlockDocument
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime
from qcsc_prefect.integrations.qiskit.qrmi import iqp as iqp_mod
from qcsc_prefect.integrations.qiskit.sources import (
    QFwResource,
    QFwSource,
    QrmiSource,
    QuantumSource,
    RandomSource,
)
from qcsc_prefect.integrations.qiskit.sources import qfw as qfw_mod
from qcsc_prefect.integrations.qiskit.sources.qfw import counts_to_result
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin

# Schema checksum of QFwResource before it was moved out of the SBD application.
QFW_RESOURCE_CHECKSUM = "sha256:63a1a7740786c51f127a62930f513d439d68552107048bb5f109f6f0cc0c76bb"


def measured_circuit(meas=3, test=None):
    """``meas`` first and then ``test``, as the SQD LUCJ circuits are built."""
    qreg = QuantumRegister(meas, "q")
    regs = [qreg, ClassicalRegister(meas, "meas")]
    if test is not None:
        regs.append(ClassicalRegister(test, "test"))
    circuit = QuantumCircuit(*regs)
    if test is not None:
        circuit.measure(qreg, circuit.cregs[1])
    circuit.h(0)
    circuit.measure(qreg, circuit.cregs[0])
    return circuit


# --- QuantumSource ---------------------------------------------------------------------


def test_sources_implement_the_interface():
    random = RandomSource(num_bits=4, seed=1)
    qfw = QFwSource("ibm_x", sampler="s.py")
    for source, name, needs in [(random, "random", False), (qfw, "ibm_x", True)]:
        assert isinstance(source, QuantumSource)
        assert (source.backend_name, source.requires_circuit) == (name, needs)


def test_a_dataclass_field_without_a_default_does_not_implement_a_property():
    # Why the sources use explicit @property: a dataclass field without a default does
    # not override the abstract property, so the class cannot be instantiated.
    @dataclass
    class NoDefault(QuantumSource):
        backend_name: str
        requires_circuit: bool

        def sample(self, circuit=None, *, shots):
            raise NotImplementedError

    with pytest.raises(TypeError, match="abstract"):
        NoDefault("x", False)


# --- RandomSource ----------------------------------------------------------------------


@pytest.mark.parametrize("num_bits,seed,shots", [(8, 26, 8), (13, 0, 50), (36, 24, 1000)])
def test_random_source_matches_generate_bit_array_uniform(num_bits, seed, shots):
    addon = pytest.importorskip("qiskit_addon_sqd.counts")
    result = RandomSource(num_bits=num_bits, seed=seed).sample(shots=shots)
    expected = addon.generate_bit_array_uniform(
        num_samples=shots, num_bits=num_bits, rand_seed=seed
    )
    assert list(result[0].data.keys()) == ["meas"]
    assert result[0].data.meas == expected


def test_random_source_repeats_and_keeps_leading_zeros():
    source = RandomSource(num_bits=13, seed=5)
    first, second = source.sample(shots=200), source.sample(None, shots=200)
    assert first[0].data.meas == second[0].data.meas
    bitstrings = first[0].data.meas.get_bitstrings()
    assert {len(b) for b in bitstrings} == {13}
    assert any(b.startswith("0") for b in bitstrings)
    assert RandomSource(num_bits=13, seed=6).sample(shots=200)[0].data.meas != first[0].data.meas


@pytest.mark.parametrize("shots", [0, -1, 2.0, "8", True, None])
def test_random_source_rejects_invalid_shots(shots):
    with pytest.raises(ValueError, match="positive integer"):
        RandomSource(num_bits=4, seed=1).sample(shots=shots)


@pytest.mark.parametrize("num_bits", [0, -1, 2.5, True])
def test_random_source_rejects_invalid_widths(num_bits):
    with pytest.raises(ValueError, match="num_bits"):
        RandomSource(num_bits=num_bits)


# --- QrmiSource ------------------------------------------------------------------------


def iqp_runtime(resource="ibm_test"):
    return QrmiRuntime(
        scheduler="iqp",
        resource=resource,
        resource_config={"QRMI_IBM_QCS_IAM_APIKEY": "secret"},
    )


@pytest.fixture
def fake_iqp(monkeypatch):
    calls = []

    async def sample_on_iqp(circuits, shots, registers, configs, runtime):
        from prefect.context import TagsContext

        calls.append({"shots": shots, "tags": set(TagsContext.get().current_tags)})
        data = {
            name: BitArray.from_samples(["0" * w] * shots, w) for name, w in registers[0].items()
        }
        return PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})], metadata={})

    monkeypatch.setattr(iqp_mod, "sample_on_iqp", sample_on_iqp)
    return calls


def test_qrmi_source_samples_one_circuit_with_the_given_shots_and_tags(fake_iqp):
    source = QrmiSource(iqp_runtime(), tags=("app: quantum",))
    result = source.sample(measured_circuit(), shots=7)
    assert source.backend_name == "ibm_test" and source.requires_circuit
    assert result[0].data.meas.num_shots == 7
    assert fake_iqp == [{"shots": 7, "tags": {"app: quantum"}}]


def test_sources_reject_a_string_where_a_sequence_is_expected():
    with pytest.raises(TypeError, match="tags"):
        QrmiSource(iqp_runtime(), tags="res: quantum")
    with pytest.raises(TypeError, match="sampler_args"):
        QFwSource("ibm_fez", sampler="s.py", sampler_args="--optimization-level 3")


@pytest.mark.parametrize("resource", ["", "ibm x", " ibm"])
def test_qrmi_source_validates_the_resource_when_built(resource):
    runtime = iqp_runtime()
    runtime.resource = resource
    with pytest.raises(ValueError, match="resource"):
        QrmiSource(runtime)


def test_qrmi_source_rejects_bad_circuits_before_submission(fake_iqp):
    source = QrmiSource(iqp_runtime())
    unmeasured = QuantumCircuit(QuantumRegister(1), ClassicalRegister(1, "meas"))
    unbound = measured_circuit()
    unbound.rx(Parameter("t"), 0)
    for circuit, message in [
        (None, "requires a circuit"),
        (unmeasured, "measured"),
        (unbound, "Bind"),
    ]:
        with pytest.raises(ValueError, match=message):
            source.sample(circuit, shots=4)
    assert fake_iqp == []


def test_qrmi_source_refuses_a_running_event_loop(fake_iqp):
    source = QrmiSource(iqp_runtime())

    async def inside_loop():
        source.sample(measured_circuit(), shots=4)

    with pytest.raises(RuntimeError):
        asyncio.run(inside_loop())
    assert fake_iqp == []


# --- QFwSource -------------------------------------------------------------------------


class FakeLauncher:
    def __init__(self, counts, **extra):
        self.counts = counts
        self.extra = extra
        self.calls = []

    def __call__(self, circuit, *, shots, qpu, work_dir, sampler, sampler_args):
        self.calls.append(
            dict(
                circuit=circuit,
                shots=shots,
                qpu=qpu,
                work_dir=work_dir,
                sampler=sampler,
                sampler_args=sampler_args,
            )
        )
        return {"counts": self.counts, **self.extra}


def qfw_source(monkeypatch, launcher, **kwargs):
    monkeypatch.setattr(qfw_mod, "_launcher", lambda: launcher)
    return QFwSource("ibm_fez", sampler="/node/sampler.py", **kwargs)


def test_qfw_source_calls_the_launcher_with_its_configuration(monkeypatch, tmp_path):
    launcher = FakeLauncher({"001": 3, "100": 1}, job_id="j1", backend="ibm_fez")
    args = ["--optimization-level", 3, "--sabre-layout-trials", 8]
    source = qfw_source(monkeypatch, launcher, sampler_args=args, work_dir=tmp_path)

    result = source.sample(measured_circuit(), shots=4)

    (call,) = launcher.calls
    assert (call["shots"], call["qpu"], call["sampler"]) == (4, "ibm_fez", "/node/sampler.py")
    assert call["work_dir"] == str(tmp_path)
    assert call["sampler_args"] == args and type(call["sampler_args"][1]) is int
    assert result.metadata["qcsc_qfw"] == {"job_id": "j1", "backend": "ibm_fez"}
    assert Counter(result[0].data.meas.get_bitstrings()) == {"001": 3, "100": 1}


def test_qfw_source_default_work_dir_is_qfw_jobs_in_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    launcher = FakeLauncher({"000": 2})
    qfw_source(monkeypatch, launcher).sample(measured_circuit(), shots=2)
    assert launcher.calls[0]["work_dir"] == str(tmp_path / "qfw_jobs")


def test_qfw_source_prepares_once_and_keeps_the_registers(monkeypatch):
    prepared = []

    def prepare(circuit):
        prepared.append(circuit)
        circuit.metadata = {"prepared": True}
        return circuit

    launcher = FakeLauncher({"000 111": 2})
    original = measured_circuit(test=3)
    qfw_source(monkeypatch, launcher, prepare=prepare).sample(original, shots=2)

    assert len(prepared) == 1 and prepared[0] is not original  # runs on a copy
    assert launcher.calls[0]["circuit"].metadata == {"prepared": True}

    def drop_test(circuit):
        return measured_circuit()

    with pytest.raises(ValueError, match="preserve classical register"):
        qfw_source(monkeypatch, launcher, prepare=drop_test).sample(original, shots=2)
    assert len(launcher.calls) == 1


@pytest.mark.parametrize(
    "circuit,shots,message",
    [
        (None, 2, "requires a circuit"),
        (QuantumCircuit(1, 1), 2, "measured"),
        (measured_circuit(), 0, "positive integer"),
    ],
)
def test_qfw_source_rejects_bad_input_before_launching(monkeypatch, circuit, shots, message):
    launcher = FakeLauncher({"000": 2})
    with pytest.raises(ValueError, match=message):
        qfw_source(monkeypatch, launcher).sample(circuit, shots=shots)
    assert launcher.calls == []


def test_qfw_source_reports_a_missing_launcher(monkeypatch):
    monkeypatch.delenv("QFW_SLURM_BIN", raising=False)
    monkeypatch.setitem(sys.modules, "qfw_slurm_sampler", None)
    with pytest.raises(RuntimeError, match="QFW_SLURM_BIN"):
        QFwSource("ibm_fez", sampler="s.py").sample(measured_circuit(), shots=2)


def test_qfw_launcher_import_adds_qfw_slurm_bin_once(monkeypatch, tmp_path):
    (tmp_path / "qfw_slurm_sampler.py").write_text("def run_sampler_job(*a, **k):\n    return a\n")
    monkeypatch.setenv("QFW_SLURM_BIN", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    # setitem first, so that monkeypatch removes the imported module afterwards.
    monkeypatch.setitem(sys.modules, "qfw_slurm_sampler", None)
    monkeypatch.delitem(sys.modules, "qfw_slurm_sampler")
    first, second = qfw_mod._launcher(), qfw_mod._launcher()
    assert first is second
    assert sys.path.count(str(tmp_path)) == 1


# --- QFw counts ------------------------------------------------------------------------


def test_counts_keep_test_and_meas_from_the_same_shot():
    # Keys list registers in reverse order: "test meas".
    counts = {"000 110": 2, "100 011": 3, "000 001": 1}
    result = counts_to_result({"counts": counts}, measured_circuit(test=3), 6)
    data = result[0].data
    pairs = Counter(zip(data.test.get_bitstrings(), data.meas.get_bitstrings()))
    assert pairs == {("000", "110"): 2, ("100", "011"): 3, ("000", "001"): 1}


def test_counts_keep_leading_zeros_and_widths():
    result = counts_to_result({"counts": {"000 001": 2, "000 010": 1}}, measured_circuit(test=3), 3)
    meas = result[0].data.meas
    assert (meas.num_bits, meas.num_shots) == (3, 3)
    assert set(meas.get_bitstrings()) == {"001", "010"}


def test_counts_accept_keys_without_spaces_and_a_single_register():
    joined = counts_to_result({"counts": {"100011": 2}}, measured_circuit(test=3), 2)
    assert joined[0].data.test.get_bitstrings() == ["100"] * 2
    assert joined[0].data.meas.get_bitstrings() == ["011"] * 2
    single = counts_to_result({"counts": {"001": 1, "000": 1}}, measured_circuit(), 2)
    assert list(single[0].data.keys()) == ["meas"]


@pytest.mark.parametrize(
    "counts,message",
    [
        ({"000 001": 1}, "1 shots; expected 2"),
        ({"000 001": 3}, "3 shots; expected 2"),
        ({"0x1": 2}, "binary|bits"),
        ({"000 0x1": 2}, "not binary"),
        ({"000 01": 2}, "2 bits for meas"),
        ({"000 001 000": 2}, "registers"),
        ({"00001": 2}, "does not have 6 bits"),
        ({"000 001": -1, "000 000": 3}, "Invalid QFw count"),
        ({"000 001": 1.5, "000 000": 0.5}, "Invalid QFw count"),
    ],
)
def test_counts_reject_malformed_records(counts, message):
    with pytest.raises(RuntimeError, match=message):
        counts_to_result({"counts": counts}, measured_circuit(test=3), 2)


def test_counts_reject_records_without_counts():
    with pytest.raises(RuntimeError, match="no counts"):
        counts_to_result({"job_id": "j"}, measured_circuit(), 2)


def test_qfw_counts_match_the_qrmi_layout_after_the_same_filter():
    # The same measured shots, as a QRMI BitArray result and as QFw counts.
    shots = [("000", "110"), ("100", "011"), ("000", "110"), ("010", "001")]
    test, meas = zip(*shots)
    qrmi = DataBin(
        meas=BitArray.from_samples(list(meas), 3), test=BitArray.from_samples(list(test), 3)
    )
    counts = Counter(f"{t} {m}" for t, m in shots)
    qfw = counts_to_result({"counts": dict(counts)}, measured_circuit(test=3), len(shots))[0].data

    def kept(data):
        return Counter(data.meas.get_bitstrings(data.test.bitcount() == 0))

    assert kept(qrmi) == kept(qfw) == {"110": 2}


def test_qfw_registers_are_in_circuit_order_like_qiskit_samplers():
    from qiskit.primitives import StatevectorSampler

    # meas (2 bits) before test (3 bits), with different widths so the layout matters.
    qreg = QuantumRegister(2, "q")
    circuit = QuantumCircuit(qreg, ClassicalRegister(2, "meas"), ClassicalRegister(3, "test"))
    circuit.x(0)
    circuit.measure(qreg, circuit.cregs[0])
    circuit.measure(qreg, circuit.cregs[1][:2])
    expected = StatevectorSampler().run([circuit], shots=4).result()[0]
    counts = expected.join_data().get_counts()
    # Counts list the registers in reverse order: "test meas".
    split = {f"{key[:3]} {key[3:]}": value for key, value in counts.items()}
    qfw = counts_to_result({"counts": split}, circuit, 4)[0]
    assert list(qfw.data.keys()) == list(expected.data.keys()) == ["meas", "test"]
    assert qfw.join_data().get_counts() == counts


# --- QFwResource -----------------------------------------------------------------------


def test_qfw_resource_keeps_its_slug_and_schema():
    assert QFwResource.get_block_type_slug() == "qfwresource"
    assert QFwResource._calculate_schema_checksum() == QFW_RESOURCE_CHECKSUM
    assert QFwResource.__doc__ is None


def test_saved_qfw_resource_documents_load():
    block_type = QFwResource._to_block_type()
    schema = QFwResource._to_block_schema(block_type_id=block_type.id)
    document = BlockDocument(
        name="qfw-runner",
        data={"device_id": "ibm_fez", "device_access_config": "/cfg.yaml", "interface": "qrmi"},
        block_schema_id=schema.id,
        block_schema=schema,
        block_type_id=block_type.id,
        block_type=block_type,
        block_type_name=block_type.name,
    )
    block = QFwResource._from_block_document(document)
    assert (block.device_id, block.device_access_config, block.interface) == (
        "ibm_fez",
        "/cfg.yaml",
        "qrmi",
    )


def test_qfw_resource_run_sampler_is_deprecated(monkeypatch):
    block = QFwResource(device_id="ibm_fez")
    monkeypatch.setattr(
        QFwResource, "run_circuit", lambda self, c, num_shots: {"result": {"counts": {"01": 2}}}
    )
    with pytest.warns(DeprecationWarning, match="QFwSource"):
        assert block.run_sampler(None, num_shots=2).num_shots == 2


GUARD = textwrap.dedent(
    """
    import builtins, threading
    real = builtins.__import__
    BLOCKED = {"defw", "svc_lib_qpm", "util", "qfw_qiskit", "qfw_slurm_sampler", "qrmi"}

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.split(".")[0] in BLOCKED:
            raise ImportError("blocked import: " + name)
        return real(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded
    """
)


def test_importing_sources_and_loading_qfw_resource_off_the_main_thread_skips_qfw(tmp_path):
    # Runs in a fresh interpreter, so modules imported by other tests do not count.
    script = GUARD + textwrap.dedent(
        """
        import sys
        errors = []

        def worker():
            try:
                from qcsc_prefect.integrations.qiskit.sources import QFwResource, RandomSource
                QFwResource(device_id="ibm_fez")
                RandomSource(num_bits=4, seed=1).sample(shots=3)
            except BaseException as exc:
                errors.append(repr(exc))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert not errors, errors
        loaded = {"defw", "svc_lib_qpm", "qfw_slurm_sampler", "ffsim"} & set(sys.modules)
        assert not loaded, loaded
        print("ok")
        """
    )
    env = {"PREFECT_HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120
    )
    assert result.stdout.strip().endswith("ok"), result.stderr[-2000:]
