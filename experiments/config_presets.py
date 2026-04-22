"""Named config presets for experiment entrypoints."""

from __future__ import annotations

from typing import Any

_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "train": {
        "baseline": {},
        "smoke": {
            "max_patches": 16,
            "batch_size": 2,
            "workers": 0,
        },
        "fast-debug": {
            "max_patches": 64,
            "batch_size": 4,
            "workers": 1,
        },
        "plan09-composite": {
            "loss_system": "composite",
            "lambda_elev": 1.0,
            "lambda_slope": 0.5,
            "batch_size": 4,
            "workers": 1,
        },
    },
    "eval": {
        "baseline": {},
        "smoke": {
            "max_patches": 16,
            "batch_size": 4,
            "workers": 0,
        },
        "fast-debug": {
            "max_patches": 64,
            "batch_size": 8,
            "workers": 1,
        },
    },
}


def list_presets(mode: str) -> list[str]:
    key = mode.strip().lower()
    if key not in _PRESETS:
        raise KeyError(f"unknown preset mode: {mode}")
    return sorted(_PRESETS[key])


def get_preset(mode: str, name: str) -> dict[str, Any]:
    mode_key = mode.strip().lower()
    preset_key = name.strip().lower()
    if mode_key not in _PRESETS:
        raise KeyError(f"unknown preset mode: {mode}")
    if preset_key not in _PRESETS[mode_key]:
        available = ", ".join(list_presets(mode_key))
        raise KeyError(f"unknown preset '{name}' for mode '{mode}', available: {available}")
    return dict(_PRESETS[mode_key][preset_key])

