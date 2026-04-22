"""Thin compatibility wrapper over legacy model factory."""

from __future__ import annotations

import torch

from dem_film_unet import create_model


def create_experiment_model(arch: str) -> torch.nn.Module:
    return create_model(arch)

