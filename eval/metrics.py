"""Eval-facing metrics module backed by shared core metrics."""

from __future__ import annotations

from core.metrics import (
    compute_per_patch_metrics,
    finalize_metric_sums,
    init_metric_sums,
    set_contour_interval,
    update_metric_sums,
)

__all__ = [
    "set_contour_interval",
    "init_metric_sums",
    "update_metric_sums",
    "finalize_metric_sums",
    "compute_per_patch_metrics",
]

