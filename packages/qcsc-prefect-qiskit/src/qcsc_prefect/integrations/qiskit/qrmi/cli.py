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
import contextlib
import getpass
import json
import logging
import os
import sys
from collections.abc import Callable, Iterator, Sequence
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


class DiscoveryError(RuntimeError):
    """IBM Quantum could not list the instances or backends of an API key."""


@contextlib.contextmanager
def quiet_ibm_runtime() -> Iterator[None]:
    """Hide qiskit-ibm-runtime's own warnings, such as "Loading instance ...".

    They describe its account selection, which this command does on its own. Set
    ``QISKIT_IBM_RUNTIME_LOG_LEVEL`` to see them.
    """
    if os.environ.get("QISKIT_IBM_RUNTIME_LOG_LEVEL"):
        yield
        return
    logger = logging.getLogger("qiskit_ibm_runtime")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(level)


def discover_instances(api_key: str, out: Callable[[str], None] = print) -> list[Instance]:
    """The instances ``api_key`` can use on IBM Quantum, with their backends.

    Each instance is opened by its CRN, so instances with the same name in different
    regions stay apart.

    Raises:
        DiscoveryError: If IBM Quantum rejects the key or cannot be reached.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    found = []
    with quiet_ibm_runtime():
        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
            items = service.instances()
        except Exception as exc:  # noqa: BLE001 - reported as one readable line
            raise DiscoveryError(_discovery_message(exc)) from exc
        for item in items:
            instance = Instance(name=item["name"], crn=item["crn"], plan=item.get("plan") or "")
            try:
                # By CRN: the service talks to the instance's regional endpoint.
                backends = service.backends(instance=instance.crn)
            except Exception as exc:  # noqa: BLE001 - one instance does not stop the rest
                out(f"  Skipped {instance.label}: {_discovery_message(exc)}")
                continue
            instance.backends = sorted({backend.name for backend in backends})
            found.append(instance)
    return found


def _discovery_message(exc: Exception) -> str:
    """One readable line for a failed IBM Quantum request."""
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    lowered = text.lower()
    if any(word in lowered for word in ("token", "api key", "unauthorized", "401", "403")):
        return "IBM Quantum rejected the API key. Check it, or create a new one on IBM Cloud."
    if any(
        word in lowered for word in ("connection", "timed out", "timeout", "resolve", "network")
    ):
        return "IBM Quantum could not be reached. Check the network connection of this host."
    if len(text) > 160:
        text = text[:157] + "..."
    return f"IBM Quantum request failed ({type(exc).__name__}): {text}"


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


def _unique_labels(instances: list[Instance]) -> list[Instance]:
    """``instances``, with a CRN suffix on labels that would still be equal."""
    seen: dict[str, int] = {}
    for instance in instances:
        seen[instance.label] = seen.get(instance.label, 0) + 1
    result = []
    for instance in instances:
        if seen[instance.label] > 1:
            # Same name, plan and region: tell them apart by the end of the CRN.
            tail = instance.crn.rstrip(":").rsplit(":", 1)[-1][-6:]
            instance = Instance(
                name=f"{instance.name} [{tail}]",
                crn=instance.crn,
                plan=instance.plan,
                backends=instance.backends,
            )
        result.append(instance)
    return result


def _pick(prompt: _Prompt, question: str, count: int, default: int) -> int:
    """Ask for one number from 1 to ``count``; return its index."""
    while True:
        answer = prompt.ask(f"{question} [{default + 1}]: ").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= count:
            return int(answer) - 1
        prompt.out(f"  Enter a number from 1 to {count}.")


def _select(
    prompt: _Prompt, instances: list[Instance], current: QrmiConfig
) -> tuple[list[Instance], Instance, str]:
    """Ask which instances to keep, the default instance and the default backend.

    Instances are chosen by number, so two with the same name are never confused.
    """
    instances = _unique_labels(instances)
    prompt.out(f"Found {len(instances)} instance(s):")
    for index, instance in enumerate(instances, start=1):
        prompt.out(f"  {index}. {instance.label}  {len(instance.backends)} backend(s)")
    while True:
        keep = prompt.ask("Use which instances? Numbers, comma separated [all]: ").strip()
        tokens = [token.strip() for token in keep.split(",") if token.strip()]
        if all(t.isdigit() and 1 <= int(t) <= len(instances) for t in tokens):
            break
        prompt.out(f"  Enter numbers from 1 to {len(instances)}, or nothing for all.")
    if tokens:
        instances = [instances[i] for i in sorted({int(t) - 1 for t in tokens})]
    if len(instances) > 1:
        prompt.out("Default instance:")
        for index, instance in enumerate(instances, start=1):
            prompt.out(f"  {index}. {instance.label}")
    previous = current.instance(current.default_instance or "")
    crns = [instance.crn for instance in instances]
    start = crns.index(previous.crn) if previous and previous.crn in crns else 0
    default = instances[_pick(prompt, "Default instance", len(instances), start)]
    if len(instances) == 1:
        prompt.out(f"Default instance: {default.label}")
    if not default.backends:
        raise SystemExit(f"{default.label} has no backends; choose another default instance.")
    names = sorted(set(default.backends))
    prompt.out(f"Backends of {default.label}:")
    for index, name in enumerate(names, start=1):
        prompt.out(f"  {index}. {name}")
    while True:
        answer = prompt.ask(f"Default backend [{names[0]}]: ").strip()
        if not answer:
            return instances, default, names[0]
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return instances, default, names[int(answer) - 1]
        if answer in names:
            return instances, default, answer
        others = [i.label for i in instances if answer in i.backends]
        hint = f" It is available to: {', '.join(others)}." if others else ""
        prompt.out(f"  {answer} is not available to {default.label}.{hint}")


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
        instances = discover_instances(key, prompt.out)
    if not instances:
        raise SystemExit("No instances: check the API key, or add them with --no-discover.")
    instances, default, backend = _select(prompt, instances, current)
    config = merge_instances(current, instances)
    config.api_key = key
    config.default_instance = default.crn
    write_config(config, path)
    new_block = load_block(args.block) is None
    save_block(
        args.block,
        _block_settings(args.block, config, default.label, backend),
        [instance.label for instance in config.instances],
        config.backends,
    )
    _print_next_steps(
        prompt.out, path, args.block, default.label, backend, default.label, new_block
    )
    return 0


def cmd_refresh(args: argparse.Namespace, prompt: _Prompt | None = None) -> int:
    prompt = prompt or _Prompt()
    path = config_path(args.config)
    current = read_config(path)
    if not current.api_key:
        raise SystemExit(f"No API key in {path}; run qcsc-qrmi login first.")
    kept = {instance.crn for instance in current.instances}
    found = discover_instances(current.api_key, prompt.out)
    labels = {instance.crn: instance.name for instance in current.instances}
    instances = [instance for instance in found if not kept or instance.crn in kept]
    for instance in instances:
        # Keep a name made unique at login, such as 'name [abc123]'.
        instance.name = labels.get(instance.crn, instance.name)
    if not instances:
        raise SystemExit("IBM Quantum returned none of the stored instances.")
    config = merge_instances(current, instances)
    if not config.instance(config.default_instance or ""):
        config.default_instance = instances[0].crn
    write_config(config, path)

    block = load_block(args.block)
    if block is None:
        prompt.out(f"Wrote {path}; there is no block {args.block} to update (run login).")
        return 0
    crn = block.instances.get(block.instance_label(), "")
    chosen = config.instance(crn) or config.instance(config.default_instance)
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
        mark = " (default)" if instance.crn == config.default_instance else ""
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
    except (ValueError, OSError, DiscoveryError) as exc:
        print(f"qcsc-qrmi {args.command}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"\nqcsc-qrmi {args.command}: cancelled; nothing was changed.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
