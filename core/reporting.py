"""Shared payload builders for experiment train/eval entrypoints."""

from __future__ import annotations

from typing import Any


def build_train_payload(
    *,
    experiment: str,
    checkpoint_out: str,
    data_root: str,
    epochs: int,
    history: dict[str, list[float]],
    train_size: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "train",
        "experiment": experiment,
        "checkpoint_out": checkpoint_out,
        "data_root": data_root,
        "epochs": int(epochs),
        "train_size": int(train_size),
        "history": history,
        "config": config,
    }


def build_eval_payload(
    *,
    experiment: str,
    prediction_sources: list[str],
    checkpoint: str | None,
    data_root: str,
    manifest: str | None,
    list_from_root: bool,
    contour_interval_m: float,
    metrics_by_source: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "eval",
        "experiment": experiment,
        "prediction_source": prediction_sources,
        "checkpoint": checkpoint,
        "data_root": data_root,
        "manifest": manifest,
        "list_from_root": list_from_root,
        "contour_interval_m": contour_interval_m,
        "metrics_by_source": metrics_by_source,
        "config": config,
    }

