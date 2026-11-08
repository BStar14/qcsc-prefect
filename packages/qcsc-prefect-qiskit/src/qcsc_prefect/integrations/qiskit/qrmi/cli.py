"""``qcsc-qrmi``: set up IBM Quantum access for QRMI, like ``prefect cloud login``.

Usage::

    qcsc-qrmi login [--config PATH] [--block qrmi-runtime] [--no-discover]
    qcsc-qrmi refresh [--config PATH] [--block qrmi-runtime]
    qcsc-qrmi list [--config PATH]
    qcsc-qrmi check [--block qrmi-runtime]

``login`` asks for the IBM Cloud API key and finds the instances it can use and
their backends on IBM Quantum (``--no-discover`` asks for them instead). It writes
``qrmi_config.json`` (``--config``, else ``$QRMI_CONFIG``, else
``~/.qiskit/qrmi_config.json``) and saves the ``qrmi-runtime`` block with dropdowns
of the instances and backends. ``refresh`` repeats the discovery with the stored
key. ``list`` prints the instances and backends, never the key. ``check`` checks
that the block's instance can use its backend.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

from .config_file import Instance as Instance
from .config_file import (
    QrmiConfig,
    config_path,
    describe,
    instance_backends,
    instance_crns,
    merge_instances,
    read_config,
    write_config,
)

DEFAULT_BLOCK_NAME = "qrmi-runtime"
# Slurm settings of a new block, as sqd-create-blocks saves them.
NEW_BLOCK_SLURM_DEFAULTS = {
    "command_block_name": "cmd-sqd-qrmi",
    "execution_profile_block_name": "exec-sqd-qrmi",
    "hpc_profile_block_name": "hpc-slurm-sqd-qrmi",
}


def discover_instances(api_key: str) -> list[Instance]:
    """The instances ``api_key`` can use on IBM Quantum, with their backends."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
    found = []
    for item in service.instances():
        crn = item["crn"]
        backends = sorted({backend.name for backend in service.backends(instance=crn)})
        found.append(
            Instance(name=item["name"], crn=crn, plan=item.get("plan") or "", backends=backends)
        )
    return found


def load_block(name: str):
    """The saved block ``name``, or None."""
    from .block import QrmiRuntimeBlock

    try:
        return QrmiRuntimeBlock.load(name, _sync=True)
    except ValueError:
        return None


def save_block(block_name: str, settings: dict[str, Any], labels, backends) -> None:
    """Save ``settings`` as block ``block_name``, with dropdowns of ``labels`` and ``backends``."""
    from .block import qrmi_runtime_block_class

    qrmi_runtime_block_class(labels, backends)(**settings).save(block_name, overwrite=True)


class _Prompt:
    """Interactive input; replaced in tests."""

    def __init__(self, ask: Callable[[str], str] = input, secret=getpass.getpass, out=print):
        self.ask = ask
        self.secret = secret
        self.out = out

    def choose(self, question: str, options: Sequence[str], default: str) -> str:
        for index, option in enumerate(options, start=1):
            self.out(f"  {index}. {option}")
        while True:
            answer = self.ask(f"{question} [{default}]: ").strip()
            if not answer:
                return default
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return options[int(answer) - 1]
            if answer in options:
                return answer
            self.out(f"Choose a number from 1 to {len(options)}, or a name from the list.")


def _manual_instances(prompt: _Prompt) -> list[Instance]:
    """Instances typed in, as many as the user wants."""
    instances = []
    prompt.out("Add each IBM Quantum instance; leave the name empty to finish.")
    while True:
        name = prompt.ask("Instance name: ").strip()
        if not name:
            break
        plan = prompt.ask("  Plan (for example premium or open): ").strip()
        crn = prompt.ask("  Service CRN: ").strip()
        backends = prompt.ask("  Backends, comma separated: ")
        names = sorted({item.strip() for item in backends.split(",") if item.strip()})
        if not crn or not names:
            prompt.out("  A CRN and at least one backend are needed; skipped.")
            continue
        instances.append(Instance(name=name, crn=crn, plan=plan, backends=names))
    return instances


def _api_key(args: argparse.Namespace, prompt: _Prompt, stored: str | None) -> str:
    if args.api_key_env:
        key = os.environ.get(args.api_key_env, "").strip()
        if not key:
            raise SystemExit(f"Environment variable {args.api_key_env} is empty or unset.")
        return key
    hint = " (Enter keeps the stored key)" if stored else ""
    key = prompt.secret(f"IBM Cloud API key{hint}: ").strip()
    if not key and stored:
        return stored
    if not key:
        raise SystemExit("An IBM Cloud API key is needed.")
    return key


def _select(
    prompt: _Prompt, instances: list[Instance], current: QrmiConfig
) -> tuple[list[Instance], Instance, str]:
    """Ask which instances to keep, the default instance and the default backend."""
    labels = [instance.label for instance in instances]
    prompt.out(f"Found {len(instances)} instance(s):")
    for instance in instances:
        prompt.out(f"  {instance.label}  {len(instance.backends)} backend(s)")
    keep = prompt.ask("Use which instances? Numbers or names, comma separated [all]: ").strip()
    if keep:
        chosen = []
        for token in (item.strip() for item in keep.split(",")):
            if token.isdigit() and 1 <= int(token) <= len(instances):
                chosen.append(instances[int(token) - 1])
            elif token in labels or any(token == i.name for i in instances):
                chosen.extend(i for i in instances if token in (i.label, i.name))
        instances = list({instance.name: instance for instance in chosen}.values()) or instances
    labels = [instance.label for instance in instances]
    previous = current.instance(current.default_instance or "")
    default_label = previous.label if previous and previous.label in labels else labels[0]
    label = prompt.choose("Default instance", labels, default_label)
    default = next(instance for instance in instances if instance.label == label)
    if not default.backends:
        raise SystemExit(f"{default.label} has no backends; choose another default instance.")
    names = sorted(set(default.backends))
    backend = prompt.choose("Default backend", names, names[0])
    return instances, default, backend


def _block_settings(block_name: str, config: QrmiConfig, label: str, backend: str) -> dict:
    """The block's settings with this login's instances; its other settings are kept."""
    block = load_block(block_name)
    settings = (block.settings() if block is not None else {}) or {
        "scheduler": "slurm",
        **NEW_BLOCK_SLURM_DEFAULTS,
    }
    settings.update(
        instance=label,
        resource=backend,
        instances=instance_crns(config.instances),
        instance_backends=instance_backends(config.instances),
        service_crn="",
    )
    if block is None:
        settings["root_dir"] = os.environ.get("QCSC_QRMI_ROOT_DIR", "/data/sqd/qrmi")
    return settings


def _print_next_steps(out, path, block_name, label, backend, instance_name, new_block) -> None:
    out("")
    out(f"Wrote {path} (mode 600).")
    out(f"Saved block {block_name}: IBM Quantum Instance {label}, QPU Backend {backend}.")
    if new_block:
        out("Run sqd-create-blocks next to save the QRMI job blocks this block names.")
    out("")
    out("For Slurm, the SPANK plugin reads the qrmi_config.json named in plugstack.conf on")
    out(f"each compute node; install {path} there (often /etc/slurm/qrmi_config.json).")
    out("")
    out("Copy these lines into the [qrmi] table of sqd_blocks.toml to keep the choice:")
    out("[qrmi]")
    out(f'resource = "{backend}"')
    out(f'instance = "{instance_name}"')


def cmd_login(args: argparse.Namespace, prompt: _Prompt | None = None) -> int:
    prompt = prompt or _Prompt()
    path = config_path(args.config)
    current = read_config(path)
    key = _api_key(args, prompt, current.api_key)
    if args.no_discover:
        instances = _manual_instances(prompt)
    else:
        prompt.out("Finding your IBM Quantum instances and backends...")
        instances = discover_instances(key)
    if not instances:
        raise SystemExit("No instances: check the API key, or add them with --no-discover.")
    instances, default, backend = _select(prompt, instances, current)
    config = merge_instances(current, instances)
    config.api_key = key
    config.default_instance = default.name
    write_config(config, path)
    new_block = load_block(args.block) is None
    save_block(
        args.block,
        _block_settings(args.block, config, default.label, backend),
        [instance.label for instance in config.instances],
        config.backends,
    )
    _print_next_steps(prompt.out, path, args.block, default.label, backend, default.name, new_block)
    return 0


def cmd_refresh(args: argparse.Namespace, prompt: _Prompt | None = None) -> int:
    prompt = prompt or _Prompt()
    path = config_path(args.config)
    current = read_config(path)
    if not current.api_key:
        raise SystemExit(f"No API key in {path}; run qcsc-qrmi login first.")
    kept = {instance.name for instance in current.instances}
    found = discover_instances(current.api_key)
    instances = [instance for instance in found if not kept or instance.name in kept]
    if not instances:
        raise SystemExit("IBM Quantum returned none of the stored instances.")
    config = merge_instances(current, instances)
    if not config.instance(config.default_instance or ""):
        config.default_instance = instances[0].name
    write_config(config, path)

    block = load_block(args.block)
    if block is None:
        prompt.out(f"Wrote {path}; there is no block {args.block} to update (run login).")
        return 0
    label = block.instance_label()
    chosen = config.instance(label) or config.instance(config.default_instance)
    backend = block.resource
    if backend not in chosen.backends:
        prompt.out(f"{backend} is no longer available to {chosen.label}.")
        backend = chosen.backends[0]
    settings = block.settings()
    settings.update(
        instance=chosen.label,
        resource=backend,
        instances=instance_crns(config.instances),
        instance_backends=instance_backends(config.instances),
    )
    save_block(args.block, settings, [i.label for i in config.instances], config.backends)
    prompt.out(f"Wrote {path}; block {args.block}: {chosen.label}, {backend}.")
    return 0


def cmd_list(args: argparse.Namespace, prompt: _Prompt | None = None) -> int:
    prompt = prompt or _Prompt()
    config = read_config(config_path(args.config))
    if args.json:
        prompt.out(json.dumps(describe(config), indent=2))
        return 0
    prompt.out(f"API key: {'set' if config.api_key else 'unset'}")
    for instance in config.instances:
        mark = " (default)" if instance.name == config.default_instance else ""
        prompt.out(f"{instance.label}{mark}: {', '.join(instance.backends)}")
    return 0


def cmd_check(args: argparse.Namespace, prompt: _Prompt | None = None) -> int:
    prompt = prompt or _Prompt()
    block = load_block(args.block)
    if block is None:
        raise SystemExit(f"No QRMI Runtime block {args.block}.")
    block.check_backend_access(live=not args.offline)
    prompt.out(f"{block.instance_label() or 'Service CRN'} can use {block.resource}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qcsc-qrmi", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, *, block=True, config=True):
        if config:
            p.add_argument("--config", help="qrmi_config.json (default ~/.qiskit/...)")
        if block:
            p.add_argument("--block", default=DEFAULT_BLOCK_NAME, help="QRMI Runtime block")

    login = sub.add_parser("login", help="Store the API key, instances and backends.")
    common(login)
    login.add_argument("--no-discover", action="store_true", help="Type the instances in.")
    login.add_argument("--api-key-env", help="Read the API key from this environment variable.")
    login.set_defaults(func=cmd_login)
    refresh = sub.add_parser("refresh", help="Update the instances and backends.")
    common(refresh)
    refresh.set_defaults(func=cmd_refresh)
    listing = sub.add_parser("list", help="Print the instances and backends.")
    common(listing, block=False)
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_list)
    check = sub.add_parser("check", help="Check the block's instance and backend.")
    common(check, config=False)
    check.add_argument("--offline", action="store_true", help="Use the stored lists only.")
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        print(f"qcsc-qrmi {args.command}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
