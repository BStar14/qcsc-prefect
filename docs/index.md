# QCSC Prefect

QCSC Prefect is a monorepo for portable HPC workflow orchestration with
[Prefect](https://www.prefect.io/). The same workflow code can run across
multiple HPC systems by switching reusable execution blocks.

## What You Can Find Here

- The core architecture and execution model in [Architecture](./concept.md)
- Step-by-step tutorials for Miyabi and Fugaku in
  [Tutorials](./tutorials/create_qcsc_workflow_for_miyabi.md)
- Operational setup guides in [How-to](./howto/howto_setup_prefect_qiskit.md)

## Repository Layout

```text
qcsc-prefect/
├── packages/
│   ├── qcsc-prefect-core/
│   ├── qcsc-prefect-blocks/
│   ├── qcsc-prefect-adapters/
│   ├── qcsc-prefect-executor/
│   ├── qcsc-prefect-qiskit/
│   ├── qcsc-prefect-dice/
│   └── qcsc-prefect-fulqrum/
├── algorithms/
├── examples/
└── docs/
```

## Quick Start

```bash
git clone https://github.com/qiskit-community/qcsc-prefect.git
cd qcsc-prefect
uv sync
```

To preview this documentation locally:

```bash
uv run --with mkdocs-material --with "mkdocstrings[python]" --with ruff mkdocs serve
```
