"""Reusable DEM metric and stratification helpers."""

from __future__ import annotations

import math
from typing import Any

import torch

from dem_film_unet import (
    DEFAULT_CONTOUR_INTERVAL_M,
    contour_sdf,
    slope_to_degrees,
    terrain_grad,
    terrain_laplacian,
    terrain_slope,
)

METRIC_SUM_KEYS: tuple[str, ...] = (
    "sum_w",
    "sum_e",
    "sum_abs_e",
    "sum_sq_e",
    "sum_abs_ds",
    "sum_sq_ds",
    "sum_abs_ds_deg",
    "sum_sq_ds_deg",
    "sum_abs_dgx",
    "sum_sq_dgx",
    "sum_abs_dgy",
    "sum_sq_dgy",
    "sum_abs_dlap",
    "sum_sq_dlap",
    "sum_abs_dsdf",
    "sum_sq_dsdf",
)

STRATA_FIELDS: tuple[str, ...] = ("slope_bin", "hydrology_bin", "building_bin", "uncertainty_bin")

_contour_interval_m: float = DEFAULT_CONTOUR_INTERVAL_M


def set_contour_interval(interval_m: float) -> None:
    """Override contour interval used for SDF-based metrics."""
    global _contour_interval_m
    _contour_interval_m = float(interval_m)


def init_metric_sums(device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.zeros((), device=device, dtype=torch.float64) for key in METRIC_SUM_KEYS}


def update_metric_sums(
    pred: torch.Tensor,
    z_gt: torch.Tensor,
    w: torch.Tensor,
    sums: dict[str, torch.Tensor],
) -> None:
    pred_f = pred.float()
    z_gt_f = z_gt.float()
    s_pred = terrain_slope(pred_f)
    s_gt = terrain_slope(z_gt_f)
    s_pred_deg = slope_to_degrees(s_pred)
    s_gt_deg = slope_to_degrees(s_gt)
    gx_pred, gy_pred = terrain_grad(pred_f)
    gx_gt, gy_gt = terrain_grad(z_gt_f)
    lap_pred = terrain_laplacian(pred_f)
    lap_gt = terrain_laplacian(z_gt_f)
    sdf_pred = contour_sdf(pred_f, _contour_interval_m)
    sdf_gt = contour_sdf(z_gt_f, _contour_interval_m)

    w64 = w.double()
    e = (pred.double() - z_gt.double()).squeeze(1)
    ds = (s_pred.double() - s_gt.double()).squeeze(1)
    ds_deg = (s_pred_deg.double() - s_gt_deg.double()).squeeze(1)
    dgx = (gx_pred.double() - gx_gt.double()).squeeze(1)
    dgy = (gy_pred.double() - gy_gt.double()).squeeze(1)
    dlap = (lap_pred.double() - lap_gt.double()).squeeze(1)
    dsdf = (sdf_pred.double() - sdf_gt.double()).squeeze(1)
    if w64.shape[1] == 1:
        w64 = w64.squeeze(1)

    sums["sum_w"] = sums["sum_w"] + w64.sum()
    sums["sum_e"] = sums["sum_e"] + (e * w64).sum()
    sums["sum_abs_e"] = sums["sum_abs_e"] + (e.abs() * w64).sum()
    sums["sum_sq_e"] = sums["sum_sq_e"] + (e * e * w64).sum()
    sums["sum_abs_ds"] = sums["sum_abs_ds"] + (ds.abs() * w64).sum()
    sums["sum_sq_ds"] = sums["sum_sq_ds"] + (ds * ds * w64).sum()
    sums["sum_abs_ds_deg"] = sums["sum_abs_ds_deg"] + (ds_deg.abs() * w64).sum()
    sums["sum_sq_ds_deg"] = sums["sum_sq_ds_deg"] + (ds_deg * ds_deg * w64).sum()
    sums["sum_abs_dgx"] = sums["sum_abs_dgx"] + (dgx.abs() * w64).sum()
    sums["sum_sq_dgx"] = sums["sum_sq_dgx"] + (dgx * dgx * w64).sum()
    sums["sum_abs_dgy"] = sums["sum_abs_dgy"] + (dgy.abs() * w64).sum()
    sums["sum_sq_dgy"] = sums["sum_sq_dgy"] + (dgy * dgy * w64).sum()
    sums["sum_abs_dlap"] = sums["sum_abs_dlap"] + (dlap.abs() * w64).sum()
    sums["sum_sq_dlap"] = sums["sum_sq_dlap"] + (dlap * dlap * w64).sum()
    sums["sum_abs_dsdf"] = sums["sum_abs_dsdf"] + (dsdf.abs() * w64).sum()
    sums["sum_sq_dsdf"] = sums["sum_sq_dsdf"] + (dsdf * dsdf * w64).sum()


def finalize_metric_sums(sums: dict[str, torch.Tensor], n_patches: int) -> dict[str, float]:
    sw = float(sums["sum_w"].clamp(min=1e-12))
    return {
        "n_patches": float(n_patches),
        "sum_weights": sw,
        "elev_bias_w": float(sums["sum_e"] / sw),
        "elev_mae_w": float(sums["sum_abs_e"] / sw),
        "elev_rmse_w": float(torch.sqrt(sums["sum_sq_e"] / sw)),
        "slope_mae_w": float(sums["sum_abs_ds"] / sw),
        "slope_rmse_w": float(torch.sqrt(sums["sum_sq_ds"] / sw)),
        "slope_mae_deg_w": float(sums["sum_abs_ds_deg"] / sw),
        "slope_rmse_deg_w": float(torch.sqrt(sums["sum_sq_ds_deg"] / sw)),
        "grad_x_mae_w": float(sums["sum_abs_dgx"] / sw),
        "grad_x_rmse_w": float(torch.sqrt(sums["sum_sq_dgx"] / sw)),
        "grad_y_mae_w": float(sums["sum_abs_dgy"] / sw),
        "grad_y_rmse_w": float(torch.sqrt(sums["sum_sq_dgy"] / sw)),
        "laplacian_mae_w": float(sums["sum_abs_dlap"] / sw),
        "laplacian_rmse_w": float(torch.sqrt(sums["sum_sq_dlap"] / sw)),
        "sdf_mae_w": float(sums["sum_abs_dsdf"] / sw),
        "sdf_rmse_w": float(torch.sqrt(sums["sum_sq_dsdf"] / sw)),
        "contour_interval_m": float(_contour_interval_m),
    }


def compute_per_patch_metrics(pred: torch.Tensor, z_gt: torch.Tensor, w: torch.Tensor) -> dict[str, list[float]]:
    pred_f = pred.float()
    z_gt_f = z_gt.float()
    s_pred = terrain_slope(pred_f)
    s_gt = terrain_slope(z_gt_f)
    s_pred_deg = slope_to_degrees(s_pred)
    s_gt_deg = slope_to_degrees(s_gt)
    gx_pred, gy_pred = terrain_grad(pred_f)
    gx_gt, gy_gt = terrain_grad(z_gt_f)
    lap_pred = terrain_laplacian(pred_f)
    lap_gt = terrain_laplacian(z_gt_f)
    sdf_pred = contour_sdf(pred_f, _contour_interval_m)
    sdf_gt = contour_sdf(z_gt_f, _contour_interval_m)

    w64 = w.double()
    e = (pred.double() - z_gt.double()).squeeze(1)
    ds = (s_pred.double() - s_gt.double()).squeeze(1)
    ds_deg = (s_pred_deg.double() - s_gt_deg.double()).squeeze(1)
    dgx = (gx_pred.double() - gx_gt.double()).squeeze(1)
    dgy = (gy_pred.double() - gy_gt.double()).squeeze(1)
    dlap = (lap_pred.double() - lap_gt.double()).squeeze(1)
    dsdf = (sdf_pred.double() - sdf_gt.double()).squeeze(1)
    if w64.shape[1] == 1:
        w64 = w64.squeeze(1)
    sum_w = w64.flatten(1).sum(dim=1).clamp(min=1e-12)

    def weighted_mean(x: torch.Tensor) -> torch.Tensor:
        return (x * w64).flatten(1).sum(dim=1) / sum_w

    return {
        "sum_weights": sum_w.tolist(),
        "elev_bias_w": weighted_mean(e).tolist(),
        "elev_mae_w": weighted_mean(e.abs()).tolist(),
        "elev_rmse_w": torch.sqrt(weighted_mean(e * e)).tolist(),
        "slope_mae_w": weighted_mean(ds.abs()).tolist(),
        "slope_rmse_w": torch.sqrt(weighted_mean(ds * ds)).tolist(),
        "slope_mae_deg_w": weighted_mean(ds_deg.abs()).tolist(),
        "slope_rmse_deg_w": torch.sqrt(weighted_mean(ds_deg * ds_deg)).tolist(),
        "grad_x_mae_w": weighted_mean(dgx.abs()).tolist(),
        "grad_x_rmse_w": torch.sqrt(weighted_mean(dgx * dgx)).tolist(),
        "grad_y_mae_w": weighted_mean(dgy.abs()).tolist(),
        "grad_y_rmse_w": torch.sqrt(weighted_mean(dgy * dgy)).tolist(),
        "laplacian_mae_w": weighted_mean(dlap.abs()).tolist(),
        "laplacian_rmse_w": torch.sqrt(weighted_mean(dlap * dlap)).tolist(),
        "sdf_mae_w": weighted_mean(dsdf.abs()).tolist(),
        "sdf_rmse_w": torch.sqrt(weighted_mean(dsdf * dsdf)).tolist(),
    }


def finalize_python_metric_sums(sums: dict[str, float], n_patches: int) -> dict[str, float]:
    sw = max(float(sums["sum_w"]), 1e-12)
    return {
        "n_patches": float(n_patches),
        "sum_weights": sw,
        "elev_bias_w": sums["sum_e"] / sw,
        "elev_mae_w": sums["sum_abs_e"] / sw,
        "elev_rmse_w": math.sqrt(max(sums["sum_sq_e"] / sw, 0.0)),
        "slope_mae_w": sums["sum_abs_ds"] / sw,
        "slope_rmse_w": math.sqrt(max(sums["sum_sq_ds"] / sw, 0.0)),
        "slope_mae_deg_w": sums["sum_abs_ds_deg"] / sw,
        "slope_rmse_deg_w": math.sqrt(max(sums["sum_sq_ds_deg"] / sw, 0.0)),
        "grad_x_mae_w": sums.get("sum_abs_dgx", 0.0) / sw,
        "grad_x_rmse_w": math.sqrt(max(sums.get("sum_sq_dgx", 0.0) / sw, 0.0)),
        "grad_y_mae_w": sums.get("sum_abs_dgy", 0.0) / sw,
        "grad_y_rmse_w": math.sqrt(max(sums.get("sum_sq_dgy", 0.0) / sw, 0.0)),
        "laplacian_mae_w": sums.get("sum_abs_dlap", 0.0) / sw,
        "laplacian_rmse_w": math.sqrt(max(sums.get("sum_sq_dlap", 0.0) / sw, 0.0)),
        "sdf_mae_w": sums.get("sum_abs_dsdf", 0.0) / sw,
        "sdf_rmse_w": math.sqrt(max(sums.get("sum_sq_dsdf", 0.0) / sw, 0.0)),
    }


def _row_value(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def compute_stratified_metrics(
    per_patch_rows: list[dict[str, object]],
    prediction_sources: list[str],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for source in prediction_sources:
        field_map: dict[str, dict[str, dict[str, float]]] = {}
        for field in STRATA_FIELDS:
            grouped: dict[str, dict[str, float]] = {}
            counts: dict[str, int] = {}
            for row in per_patch_rows:
                group = str(row.get(field, "missing"))
                sums = grouped.setdefault(group, {key: 0.0 for key in METRIC_SUM_KEYS})
                counts[group] = counts.get(group, 0) + 1
                sum_w = float(row[f"{source}_sum_weights"])
                sums["sum_w"] += sum_w
                sums["sum_e"] += float(row[f"{source}_elev_bias_w"]) * sum_w
                sums["sum_abs_e"] += float(row[f"{source}_elev_mae_w"]) * sum_w
                sums["sum_sq_e"] += float(row[f"{source}_elev_rmse_w"]) ** 2 * sum_w
                sums["sum_abs_ds"] += float(row[f"{source}_slope_mae_w"]) * sum_w
                sums["sum_sq_ds"] += float(row[f"{source}_slope_rmse_w"]) ** 2 * sum_w
                sums["sum_abs_ds_deg"] += float(row[f"{source}_slope_mae_deg_w"]) * sum_w
                sums["sum_sq_ds_deg"] += float(row[f"{source}_slope_rmse_deg_w"]) ** 2 * sum_w
                sums["sum_abs_dgx"] += _row_value(row, f"{source}_grad_x_mae_w") * sum_w
                sums["sum_sq_dgx"] += _row_value(row, f"{source}_grad_x_rmse_w") ** 2 * sum_w
                sums["sum_abs_dgy"] += _row_value(row, f"{source}_grad_y_mae_w") * sum_w
                sums["sum_sq_dgy"] += _row_value(row, f"{source}_grad_y_rmse_w") ** 2 * sum_w
                sums["sum_abs_dlap"] += _row_value(row, f"{source}_laplacian_mae_w") * sum_w
                sums["sum_sq_dlap"] += _row_value(row, f"{source}_laplacian_rmse_w") ** 2 * sum_w
                sums["sum_abs_dsdf"] += _row_value(row, f"{source}_sdf_mae_w") * sum_w
                sums["sum_sq_dsdf"] += _row_value(row, f"{source}_sdf_rmse_w") ** 2 * sum_w
            field_map[field] = {
                group: finalize_python_metric_sums(group_sums, counts[group])
                for group, group_sums in sorted(grouped.items())
            }
        out[source] = field_map
    return out


def parse_patch_stem(stem: str) -> dict[str, int | str] | None:
    parts = stem.split("_")
    if len(parts) != 5:
        return None
    x, y, zone, country, year = parts
    try:
        return {"x": int(x), "y": int(y), "zone": int(zone), "country": country, "year": int(year)}
    except ValueError:
        return None


def pct_improvement(baseline: float, improved: float) -> float:
    denom = max(abs(baseline), 1e-12)
    return 100.0 * (baseline - improved) / denom


def add_customer_example_fields(
    row: dict[str, Any],
    *,
    baseline_source: str,
    improved_source: str,
) -> None:
    metric_pairs = (
        ("elev_mae_w", 0.35),
        ("elev_rmse_w", 0.15),
        ("slope_mae_deg_w", 0.35),
        ("slope_rmse_deg_w", 0.15),
    )
    score = 0.0
    for metric_name, weight in metric_pairs:
        baseline = float(row[f"{baseline_source}_{metric_name}"])
        improved = float(row[f"{improved_source}_{metric_name}"])
        delta = baseline - improved
        pct = pct_improvement(baseline, improved)
        row[f"{improved_source}_vs_{baseline_source}_{metric_name}_improvement"] = delta
        row[f"{improved_source}_vs_{baseline_source}_{metric_name}_improvement_pct"] = pct
        score += weight * pct
    row[f"{improved_source}_vs_{baseline_source}_customer_example_score"] = score

