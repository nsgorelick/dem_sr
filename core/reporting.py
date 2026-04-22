"""Shared payload builders for experiment train/eval entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


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
        "history": _json_safe(history),
        "config": _json_safe(config),
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
        "metrics_by_source": _json_safe(metrics_by_source),
        "config": _json_safe(config),
    }

