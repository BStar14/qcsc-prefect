import os
from types import SimpleNamespace

import defw  # noqa: F401
from prefect.blocks.core import Block
from pydantic import Field
from qiskit.primitives.containers import BitArray
from svc_lib_qpm.descriptor import resolve_descriptor
from svc_lib_qpm.drivers.qrmi_driver import QrmiDriver
from svc_lib_qpm.frontend import Frontend
from util.circuit_payload import dump_qpy


class QFwResource(Block):
    device_id: str = Field(description="QFw device identifier, for example ibm_fez.")

    device_access_config: str = Field(
        default="/workspace/qfw-container-base/device-access.yaml",
        description="Path to the QFw device-access configuration.",
    )

    interface: str = Field(
        default="qrmi",
        description="QFw quantum interface library.",
    )

    def _create_driver(self):
        os.environ["QFW_DEVICE_ACCESS_CFG"] = self.device_access_config
        os.environ["QFW_QPU_IFACE_PREF"] = self.interface

        descriptor = resolve_descriptor(self.device_id)
        driver = QrmiDriver(descriptor)

        return descriptor, driver

    def get_target(self):
        _, driver = self._create_driver()

        target, _ = driver._build_backend_target(
            SimpleNamespace(
                info={},
                provider_credential=None,
            )
        )

        return target

    def run_circuit(self, circuit, num_shots=1024):
        descriptor, driver = self._create_driver()
        frontend = Frontend([driver], descriptor)

        info = {
            "circuit": {
                "format": "qpy+gzip",
                "data": dump_qpy(circuit, compress=True),
            },
            "num_shots": num_shots,
        }

        source = SimpleNamespace(
            info=info,
            provider_credential=None,
        )

        return frontend.run_circuit(source)

    def run_sampler(self, circuit, num_shots=1024, use_reset_mitigation=False):
        result = self.run_circuit(circuit, num_shots=num_shots)

        counts = result["result"]["counts"]

        if use_reset_mitigation:
            samples = []

            for bitstring, count in counts.items():
                registers = bitstring.split()

                if len(registers) != 2:
                    raise RuntimeError(
                        "Expected QFw reset-mitigation counts to contain "
                        f"two classical registers, got {bitstring!r}."
                    )

                # Qiskit count strings list classical registers
                # in reverse register order.
                test_bits = registers[0]
                meas_bits = registers[1]

                if int(test_bits, 2).bit_count() == 0:
                    samples.extend([meas_bits] * count)

            if not samples:
                raise RuntimeError(
                    "Reset mitigation removed all samples from the QFw sampler result."
                )

            return BitArray.from_samples(samples)

        samples = []
        for bitstring, count in counts.items():
            samples.extend([bitstring] * count)

        return BitArray.from_samples(samples)
