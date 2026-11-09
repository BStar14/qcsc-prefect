"""Former location of ``QFwResource``; it is now in qcsc_prefect.integrations.qiskit.sources.

Re-exported as the same class, so saved ``qfwresource`` blocks, ``create_qfw_block.py``
and existing imports keep working. Importing it does not import QFw or DEFw.
"""

from qcsc_prefect.integrations.qiskit.sources.qfw_resource import QFwResource

__all__ = ["QFwResource"]
