"""Experiment registry."""

from __future__ import annotations

from experiments.base import Experiment
from experiments.baseline_film import BaselineFilmExperiment
from experiments.hydrology import HydrologyExperiment

_EXPERIMENTS: dict[str, type[Experiment]] = {
    "baseline": BaselineFilmExperiment,
    "hydrology": HydrologyExperiment,
}


def list_experiments() -> list[str]:
    return sorted(_EXPERIMENTS)


def create_experiment(name: str) -> Experiment:
    key = name.strip().lower()
    if key not in _EXPERIMENTS:
        available = ", ".join(list_experiments())
        raise KeyError(f"unknown experiment '{name}', available: {available}")
    return _EXPERIMENTS[key]()

