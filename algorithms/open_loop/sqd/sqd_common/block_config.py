"""Load the ``sqd-create-blocks`` config: your settings, a system profile, and ``[advanced]``.

``sqd_blocks.example.toml`` holds the settings each user edits. Its ``profile`` names
a bundled system profile (``slurm``, ``miyabi``, ``fugaku``, or the GPU profiles
``slurm-gpu`` and ``miyabi-gpu`` for fulqrum, under ``sqd_common/profiles/``) with the
scheduler and resource settings of that system, or gives a path to your own profile
file. Rarely changed settings (block names, timeouts,
solver tuning) go in an optional ``[advanced]`` table.

``load_block_config`` merges these into the flat settings the solver scripts read.
Later layers win:

1. the profile, then its ``[<solver>]`` table (``[dice]`` with ``solver = "dice"``)
2. ``[advanced]``
3. the top level of your file, then its ``[<solver>]`` table

``[qrmi]`` merges the profile's QRMI job defaults under your keys, and only when your
file has a ``[qrmi]`` table. ``[runtime]`` is taken from your file as it is. A config
without ``profile``, ``solver`` or these layer tables is used unchanged, so flat
configs keep working.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
SOLVERS = ("fulqrum", "dice", "sbd")
# Keys that select the layers; they are not block settings.
_SELECTORS = ("profile", "solver")
# Tables that are layers, not settings of their own.
_LAYER_TABLES = ("advanced", *SOLVERS)
# Tables that the solver scripts read as tables.
_SETTING_TABLES = ("qrmi", "runtime", "environments", "sqd_advanced")
# Keys that only [qrmi] has. Elsewhere they mean its header line is commented out.
_QRMI_ONLY_KEYS = ("resource",)


def read_config_file(path: Path) -> dict[str, Any]:
    """Read a TOML or JSON config, with a hint for a commented-out table header."""
    if not path.exists():
        raise FileNotFoundError(f"Config file was not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as f:
        try:
            return tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            hint = ""
            if "Cannot overwrite a value" in str(exc):
                # Table keys left without their header redefine top-level keys such as
                # walltime, ompthreads or timeout_seconds.
                hint = (
                    " A key is set twice in one table. Keys after a table header such as "
                    "[qrmi] or [advanced] belong to that table, so check that the header "
                    "line is not commented out and that top-level keys come before it."
                )
            raise RuntimeError(f"Invalid TOML in {path}: {exc}.{hint}") from exc


def profile_names() -> list[str]:
    """Names of the bundled system profiles."""
    return sorted(path.stem for path in PROFILES_DIR.glob("*.toml"))


def profile_path(profile: str, base: Path | None = None) -> Path:
    """The path of a bundled profile name, or of a profile file relative to ``base``."""
    if profile in profile_names():
        return PROFILES_DIR / f"{profile}.toml"
    path = Path(profile).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    if not path.exists():
        raise RuntimeError(
            f"Unknown profile {profile!r}: use one of {', '.join(profile_names())} or a path "
            "to a profile TOML file."
        )
    return path


def solver_of(config: dict[str, Any]) -> str:
    """The solver a config selects (``fulqrum`` by default)."""
    solver = str(config.get("solver", "fulqrum")).strip().lower()
    if solver not in SOLVERS:
        raise RuntimeError(f"'solver' must be one of {', '.join(SOLVERS)}; got {solver!r}.")
    return solver


def is_layered(config: dict[str, Any]) -> bool:
    return any(key in config for key in (*_SELECTORS, *_LAYER_TABLES))


def merge_layers(
    config: dict[str, Any], base: Path | None = None, solver: str | None = None
) -> dict[str, Any]:
    """Flatten a layered config into the settings of ``solver`` (default: its ``solver``)."""
    if not is_layered(config):
        return dict(config)
    solver = solver or solver_of(config)
    profile: dict[str, Any] = {}
    if "profile" in config:
        profile = read_config_file(profile_path(str(config["profile"]), base))

    merged: dict[str, Any] = {}
    for layer in (
        profile,
        _table(profile, solver),
        _table(config, "advanced"),
        config,
        _table(config, solver),
    ):
        merged.update(_flat(layer))
    if "environments" in profile or "environments" in config:
        merged["environments"] = {
            **_table(profile, "environments"),
            **_table(config, "environments"),
        }
    if "qrmi" in config:
        merged["qrmi"] = {**_table(profile, "qrmi"), **_table(config, "qrmi")}
    if "runtime" in config:
        merged["runtime"] = config["runtime"]
    if "sqd_advanced" in config:
        merged["sqd_advanced"] = config["sqd_advanced"]
    return merged


def load_block_config(path: Path | None, solver: str | None = None) -> dict[str, Any]:
    """Read and flatten the config file of ``sqd-create-blocks`` for ``solver``.

    ``solver`` is the solver whose blocks are created, which ``--solver`` may set
    to another than the file's ``solver``; its ``[<solver>]`` table applies.
    """
    if path is None:
        return {}
    config = read_config_file(path)
    check_qrmi_keys(config, path)
    return merge_layers(config, path.parent, solver)


def check_qrmi_keys(config: dict[str, Any], path: Path | None = None) -> None:
    """Reject [qrmi] keys outside [qrmi], which a commented-out header would leave there."""
    where = f" of {path}" if path is not None else ""
    places = [("the top level", config)]
    places += [(f"[{name}]", config[name]) for name in _LAYER_TABLES if name in config]
    for place, table in places:
        found = sorted(set(table) & set(_QRMI_ONLY_KEYS)) if isinstance(table, dict) else []
        if found:
            raise RuntimeError(
                f"{', '.join(found)} in {place}{where} belongs to the [qrmi] table. Check that "
                "the [qrmi] header line is not commented out."
            )


def _table(config: dict[str, Any], name: str) -> dict[str, Any]:
    table = config.get(name, {})
    if not isinstance(table, dict):
        raise RuntimeError(f"'{name}' must be a table.")
    return dict(table)


def _flat(layer: dict[str, Any]) -> dict[str, Any]:
    """The plain settings of one layer, without selectors and tables."""
    skip = {*_SELECTORS, *_LAYER_TABLES, *_SETTING_TABLES}
    return {key: value for key, value in layer.items() if key not in skip}
