from __future__ import annotations

from typing import Any


def resolve_sampler_options_and_work_dir(
    raw_value: Any,
    *,
    default_shots: int,
) -> tuple[dict[str, Any], str | None]:
    """
    Resolve Prefect Variable payload into:
    - sampler options for QrmiRuntime.sampler(...)
    - optional workflow base work_dir

    Supported payload shapes:
    1) Legacy (existing):
       {"params": {"shots": 100000}}
    2) New (recommended):
       {
         "sampler_options": {"params": {"shots": 100000}},
         "work_dir": "/work/.../miyabi_tutorial"
       }
    3) Mixed:
       {"params": {...}, "work_dir": "..."}  # work_dir is stripped before sampler call
    """
    default_options: dict[str, Any] = {"params": {"shots": int(default_shots)}}

    if raw_value is None:
        return default_options, None

    if not isinstance(raw_value, dict):
        raise TypeError(f"Prefect Variable payload must be a mapping. Got: {type(raw_value)!r}")

    raw_work_dir = raw_value.get("work_dir")
    work_dir: str | None = None
    if raw_work_dir is not None:
        if not isinstance(raw_work_dir, str):
            raise TypeError("'work_dir' in options variable must be a string.")
        stripped = raw_work_dir.strip()
        work_dir = stripped or None

    if "sampler_options" in raw_value:
        sampler_options = raw_value["sampler_options"]
        if not isinstance(sampler_options, dict):
            raise TypeError("'sampler_options' in options variable must be a mapping.")
    else:
        sampler_options = dict(raw_value)
        sampler_options.pop("work_dir", None)

    if not sampler_options:
        sampler_options = default_options

    return sampler_options, work_dir
