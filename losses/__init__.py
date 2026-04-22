"""Composable loss system for experiment entrypoints."""

from .components import (
    ContourIndicatorL1Loss,
    ContourSDFL1Loss,
    CurvatureL1Loss,
    ElevationSmoothL1Loss,
    FlowDirectionProxyLoss,
    GradientL1Loss,
    MultiScaleElevationSmoothL1Loss,
    PitSpikePenaltyLoss,
    SlopeL1Loss,
)
from .composite import CompositeLoss, build_composite_loss_from_config

__all__ = [
    "CompositeLoss",
    "build_composite_loss_from_config",
    "ElevationSmoothL1Loss",
    "SlopeL1Loss",
    "GradientL1Loss",
    "CurvatureL1Loss",
    "MultiScaleElevationSmoothL1Loss",
    "ContourSDFL1Loss",
    "ContourIndicatorL1Loss",
    "FlowDirectionProxyLoss",
    "PitSpikePenaltyLoss",
]

