"""Configuration: settings, mode presets and YAML loading.

`PipelineConfig` (the contract) lives in `domain/schemas.py`. This package holds
the runtime settings object and the mode presets.

Imports here are lazy because `domain.schemas` needs the presets at import
time; eagerly importing `settings` from this `__init__` would create a cycle.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "Settings",
    "settings",
    "get_pipeline_config",
    "load_config_from_yaml",
    "MODE_PRESETS",
]

_MODE = "settings"


def __getattr__(name: str) -> Any:
    if name in ("Settings", "settings", "get_pipeline_config", "load_config_from_yaml"):
        from importlib import import_module

        module = import_module(f"{__name__}.{_MODE}")
        return getattr(module, name)
    if name == "MODE_PRESETS":
        from importlib import import_module

        return import_module(f"{__name__}.defaults").MODE_PRESETS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
