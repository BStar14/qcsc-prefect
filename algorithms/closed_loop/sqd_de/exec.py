import sys
from pathlib import Path

from sbd.flow_params import FlowParameters
from sbd.main import riken_sqd_de


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python algorithms/closed_loop/sqd_de/exec.py <params.json>")

    json_file = Path(sys.argv[1]).expanduser().resolve()
    params = FlowParameters.model_validate_json(json_file.read_text())
    riken_sqd_de(parameters=params)


if __name__ == "__main__":
    main()
