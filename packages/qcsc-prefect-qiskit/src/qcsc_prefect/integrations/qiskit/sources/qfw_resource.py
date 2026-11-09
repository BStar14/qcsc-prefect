# Saved QFw device settings. Importing or loading this block does not import QFw:
# defw, svc_lib_qpm and util.circuit_payload are imported only inside the methods that
# use them, because DEFw installs signal handlers on import, which Python allows only in
# the main thread. QFwSource uses only device_id.
#
# Keep the class without a docstring and keep the fields, descriptions and defaults:
# they define the schema checksum of the saved "qfwresource" blocks.

import os
import warnings
from types import SimpleNamespace

from prefect.blocks.core import Block
from pydantic import Field


class QFwResource(Block):
    _block_type_slug = "qfwresource"

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
        import defw  # noqa: F401
        from svc_lib_qpm.descriptor import resolve_descriptor
        from svc_lib_qpm.drivers.qrmi_driver import QrmiDriver

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
        from svc_lib_qpm.frontend import Frontend
        from util.circuit_payload import dump_qpy

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
        # Deprecated: filters inside the block and loses leading zeros. Use QFwSource,
        # which returns all registers, and filter in the application.
        warnings.warn(
            "QFwResource.run_sampler is deprecated; use QFwSource and filter the result "
            "in the application",
            DeprecationWarning,
            stacklevel=2,
        )
        from qiskit.primitives.containers import BitArray

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
