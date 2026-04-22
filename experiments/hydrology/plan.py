"""Hydrology experiment placeholder on top of baseline FiLM."""

from __future__ import annotations

import argparse

from experiments.baseline.plan import BaselineFilmExperiment


class HydrologyExperiment(BaselineFilmExperiment):
    """Placeholder experiment key for hydrology-focused variants."""

    name = "hydrology"

    @classmethod
    def add_train_args(cls, _parser: argparse.ArgumentParser) -> None:
        return
