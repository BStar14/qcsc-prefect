"""Run an SQD flow once from a FlowParameters JSON file, without a deployment.

Usage (from the repository root)::

    python algorithms/sqd/exec.py fulqrum algorithms/sqd/default_params/fulqrum_n2_random.json
    python algorithms/sqd/exec.py dice algorithms/sqd/default_params/dice_n2.json
"""

import asyncio
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in {"fulqrum", "dice"}:
        raise SystemExit("Usage: python algorithms/sqd/exec.py {fulqrum|dice} <params.json>")

    if sys.argv[1] == "fulqrum":
        from sqd_fulqrum.main import FlowParameters
        from sqd_fulqrum.main import sqd_2405_05068_fulqrum as sqd_flow
    else:
        from sqd_dice.main import FlowParameters
        from sqd_dice.main import sqd_2405_05068 as sqd_flow

    json_file = Path(sys.argv[2]).expanduser().resolve()
    params = FlowParameters.model_validate_json(json_file.read_text())
    print(asyncio.run(sqd_flow(parameters=params)))


if __name__ == "__main__":
    main()
