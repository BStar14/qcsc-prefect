"""Run the ``sqd`` flow once from a FlowParameters JSON file, without a deployment.

Usage::

    cd algorithms/open_loop/sqd
    python exec.py default_params/n2_random.json
"""

import asyncio
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python exec.py <params.json>")

    from sqd_common.flow_params import FlowParameters
    from sqd_common.main import sqd

    json_file = Path(sys.argv[1]).expanduser().resolve()
    params = FlowParameters.model_validate_json(json_file.read_text())
    print(asyncio.run(sqd(parameters=params)))


if __name__ == "__main__":
    main()
