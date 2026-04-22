"""Shared evaluation loop utilities for experiment entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader

from core.data_schema import validate_batch, validate_model_outputs
from core.metrics import (
    add_customer_example_fields,
    compute_per_patch_metrics,
    finalize_metric_sums,
    init_metric_sums,
    parse_patch_stem,
    update_metric_sums,
)
from eval.predictors import predict_model, predict_stage_a, predict_z_lr


@torch.no_grad()
def run_eval_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_forward: Callable[[torch.nn.Module, dict[str, Any]], dict[str, torch.Tensor]],
    amp_enabled: bool,
) -> dict[str, float]:
    """Run one evaluation pass and return weighted DEM metrics."""
    model.eval()
    sums = init_metric_sums(device)
    n_patches = 0

    for batch_raw in loader:
        batch: dict[str, Any] = dict(batch_raw)
        validate_batch(batch, require_ae=True)
        batch_tensors = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model_forward(model, batch_tensors)
            z_hat = validate_model_outputs(outputs)
        if z_hat.shape != batch_tensors["z_gt"].shape:
            raise ValueError(
                f"prediction/ground-truth shape mismatch: {tuple(z_hat.shape)} vs {tuple(batch_tensors['z_gt'].shape)}"
            )
        update_metric_sums(z_hat, batch_tensors["z_gt"], batch_tensors["w"], sums)
        n_patches += int(batch_tensors["z_gt"].shape[0])

    return finalize_metric_sums(sums, n_patches=n_patches)


@torch.no_grad()
def run_eval_epoch_multi_source(
    *,
    model: torch.nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    model_forward: Callable[[torch.nn.Module, dict[str, Any]], dict[str, torch.Tensor]],
    amp_enabled: bool,
    prediction_sources: list[str],
    sliding_window_tile_size: int | None = None,
    sliding_window_overlap: int = 0,
) -> dict[str, dict[str, float]]:
    """Run one evaluation pass for one or more prediction sources."""
    sources = list(dict.fromkeys(prediction_sources))
    if "model" in sources and model is None:
        raise ValueError("model is required when prediction_sources includes 'model'")
    if model is not None:
        model.eval()

    sums_by_source = {source: init_metric_sums(device) for source in sources}
    n_patches = 0
    for batch_raw in loader:
        batch: dict[str, Any] = dict(batch_raw)
        validate_batch(batch, require_ae=True)
        batch_tensors = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            for source in sources:
                if source == "model":
                    assert model is not None
                    pred = predict_model(
                        model,
                        batch_tensors,
                        model_forward,
                        sliding_window_tile_size=sliding_window_tile_size,
                        sliding_window_overlap=sliding_window_overlap,
                        amp_enabled=amp_enabled,
                    )
                    pred = validate_model_outputs({"z_hat": pred})
                elif source == "z_lr":
                    pred = predict_z_lr(batch_tensors)
                elif source == "stage_a":
                    assert model is not None
                    pred = predict_stage_a(model, batch_tensors, model_forward)
                else:
                    raise ValueError(f"unsupported prediction source: {source}")
                if pred.shape != batch_tensors["z_gt"].shape:
                    raise ValueError(
                        f"prediction/ground-truth shape mismatch: {tuple(pred.shape)} vs {tuple(batch_tensors['z_gt'].shape)}"
                    )
                update_metric_sums(pred, batch_tensors["z_gt"], batch_tensors["w"], sums_by_source[source])
        n_patches += int(batch_tensors["z_gt"].shape[0])

    return {source: finalize_metric_sums(sums, n_patches=n_patches) for source, sums in sums_by_source.items()}


@torch.no_grad()
def run_eval_epoch_multi_source_with_rows(
    *,
    model: torch.nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    model_forward: Callable[[torch.nn.Module, dict[str, Any]], dict[str, torch.Tensor]],
    amp_enabled: bool,
    prediction_sources: list[str],
    sliding_window_tile_size: int | None = None,
    sliding_window_overlap: int = 0,
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    """Run multi-source eval and collect per-patch rows."""
    sources = list(dict.fromkeys(prediction_sources))
    if "model" in sources and model is None:
        raise ValueError("model is required when prediction_sources includes 'model'")
    if model is not None:
        model.eval()

    sums_by_source = {source: init_metric_sums(device) for source in sources}
    n_patches = 0
    per_patch_rows: list[dict[str, object]] = []

    for batch_raw in loader:
        batch: dict[str, Any] = dict(batch_raw)
        validate_batch(batch, require_ae=True)
        batch_tensors = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

        pred_by_source: dict[str, torch.Tensor] = {}
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            for source in sources:
                if source == "model":
                    assert model is not None
                    pred = predict_model(
                        model,
                        batch_tensors,
                        model_forward,
                        sliding_window_tile_size=sliding_window_tile_size,
                        sliding_window_overlap=sliding_window_overlap,
                        amp_enabled=amp_enabled,
                    )
                    pred = validate_model_outputs({"z_hat": pred})
                elif source == "z_lr":
                    pred = predict_z_lr(batch_tensors)
                elif source == "stage_a":
                    assert model is not None
                    pred = predict_stage_a(model, batch_tensors, model_forward)
                else:
                    raise ValueError(f"unsupported prediction source: {source}")
                if pred.shape != batch_tensors["z_gt"].shape:
                    raise ValueError(
                        f"prediction/ground-truth shape mismatch: {tuple(pred.shape)} vs {tuple(batch_tensors['z_gt'].shape)}"
                    )
                pred_by_source[source] = pred
                update_metric_sums(pred, batch_tensors["z_gt"], batch_tensors["w"], sums_by_source[source])

        per_source_patch_metrics = {
            source: compute_per_patch_metrics(pred_by_source[source], batch_tensors["z_gt"], batch_tensors["w"])
            for source in sources
        }
        stems_raw = batch.get("stem")
        if stems_raw is None:
            batch_n = int(batch_tensors["z_gt"].shape[0])
            stems = [f"patch_{n_patches + idx}" for idx in range(batch_n)]
        else:
            stems = list(stems_raw)
        for idx, stem in enumerate(stems):
            row: dict[str, object] = {"stem": stem}
            parsed = parse_patch_stem(stem)
            if parsed is not None:
                row.update(parsed)
            for source in sources:
                for metric_name, values in per_source_patch_metrics[source].items():
                    row[f"{source}_{metric_name}"] = float(values[idx])
            if "model" in sources and "z_lr" in sources:
                add_customer_example_fields(row, baseline_source="z_lr", improved_source="model")
            per_patch_rows.append(row)

        n_patches += int(batch_tensors["z_gt"].shape[0])

    metrics_by_source = {source: finalize_metric_sums(sums, n_patches=n_patches) for source, sums in sums_by_source.items()}
    return metrics_by_source, per_patch_rows

