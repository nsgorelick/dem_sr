"""Experiment registry."""

from __future__ import annotations

from experiments.base import Experiment
from experiments.baseline_film import BaselineFilmExperiment
from experiments.hydrology import HydrologyExperiment
from experiments.two_stage import TwoStageResidualExperiment

_EXPERIMENTS: dict[str, type[Experiment]] = {
    "baseline": BaselineFilmExperiment,
    "hydrology": HydrologyExperiment,
    "two_stage": TwoStageResidualExperiment,
}


def list_experiments() -> list[str]:
    return sorted(_EXPERIMENTS)


def create_experiment(name: str) -> Experiment:
    key = name.strip().lower()
    if key not in _EXPERIMENTS:
        available = ", ".join(list_experiments())
        raise KeyError(f"unknown experiment '{name}', available: {available}")
    return _EXPERIMENTS[key]()

