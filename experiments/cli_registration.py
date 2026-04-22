"""Register all experiment-specific CLI flags on shared train/eval parsers."""

from __future__ import annotations

import argparse

from experiments.baseline.plan import BaselineFilmExperiment
from experiments.frequency_domain.plan import FrequencyDomainExperiment
from experiments.mixture_specialists.plan import MixtureSpecialistsExperiment
from experiments.two_stage.plan import TwoStageResidualExperiment


def add_all_train_experiment_args(parser: argparse.ArgumentParser) -> None:
    """Append every experiment train flag group (union of all plans)."""
    BaselineFilmExperiment.add_train_args(parser)
    TwoStageResidualExperiment.add_train_args(parser)
    MixtureSpecialistsExperiment.add_train_args(parser)
    FrequencyDomainExperiment.add_train_args(parser)


def add_all_eval_experiment_args(parser: argparse.ArgumentParser) -> None:
    """Append every experiment eval flag group (union of all plans)."""
    TwoStageResidualExperiment.add_eval_args(parser)
