"""Experiment registry."""

from __future__ import annotations

from experiments.base import Experiment
from experiments.baseline.plan import BaselineFilmExperiment
from experiments.frequency_domain.plan import FrequencyDomainExperiment
from experiments.hydrology.plan import HydrologyExperiment
from experiments.mixture_specialists.plan import MixtureSpecialistsExperiment
from experiments.two_stage.plan import TwoStageResidualExperiment

_EXPERIMENTS: dict[str, type[Experiment]] = {
    "baseline": BaselineFilmExperiment,
    "frequency_domain": FrequencyDomainExperiment,
    "hydrology": HydrologyExperiment,
    "mixture_specialists": MixtureSpecialistsExperiment,
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
