"""Aggregate named preset fragments registered by individual experiment packages."""

from __future__ import annotations

from typing import Any


def collect_train_preset_entries() -> dict[str, dict[str, Any]]:
    from experiments.baseline.presets import TRAIN_PRESET_ENTRIES as baseline_entries
    from experiments.frequency_domain.presets import TRAIN_PRESET_ENTRIES as frequency_entries
    from experiments.hydrology.presets import TRAIN_PRESET_ENTRIES as hydrology_entries
    from experiments.mixture_specialists.presets import TRAIN_PRESET_ENTRIES as mixture_entries
    from experiments.two_stage.presets import TRAIN_PRESET_ENTRIES as two_stage_entries

    merged: dict[str, dict[str, Any]] = {}
    for chunk in (
        baseline_entries,
        hydrology_entries,
        frequency_entries,
        mixture_entries,
        two_stage_entries,
    ):
        merged.update(chunk)
    return merged
