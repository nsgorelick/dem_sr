#!/usr/bin/env python3
"""Evaluate model predictions, the raw ``z_lr`` baseline, and comparison rasters.

Reports weighted elevation / slope metrics aligned with ``loss_dem`` so the
trained model, the input baseline, and external DTMs can be compared on the same
holdout set.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dem_film_unet import (
    ARCH_CHOICES,
    ARCH_FILM,
    DEFAULT_CONTOUR_INTERVAL_M,
    contour_sdf,
    create_model,
    slope_to_degrees,
    terrain_grad,
    terrain_laplacian,
    terrain_slope,
)
from local_patch_dataset import (
    LocalDemPatchDataset,
    collate_dem_batch,
    list_patch_stems,
    load_patch_stems_manifest,
)
from patch_table import load_patch_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_dem")

PATCH_TABLE_FIELDS = (
    "p90_slope",
    "frac_shore",
    "frac_water",
    "has_edge",
    "frac_building",
    "mean_uncert",
    "mean_W",
    "valid_frac",
    "gt_coverage_mean",
    "resid_scale",
    "relief",
    "stratum_id",
)
STRATA_FIELDS = ("slope_bin", "hydrology_bin", "building_bin", "uncertainty_bin")


_EVAL_CONTOUR_INTERVAL_M: float = DEFAULT_CONTOUR_INTERVAL_M


def set_eval_contour_interval(interval_m: float) -> None:
    """Override the contour interval used by geometry/contour metrics."""
    global _EVAL_CONTOUR_INTERVAL_M
    _EVAL_CONTOUR_INTERVAL_M = float(interval_m)


def update_metric_sums(
    pred: torch.Tensor,
    z_gt: torch.Tensor,
    w: torch.Tensor,
    sums: dict[str, torch.Tensor],
) -> None:
    """Accumulate weighted elevation, slope, gradient, curvature, and SDF errors."""
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
    sdf_pred = contour_sdf(pred_f, _EVAL_CONTOUR_INTERVAL_M)
    sdf_gt = contour_sdf(z_gt_f, _EVAL_CONTOUR_INTERVAL_M)

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


_METRIC_SUM_KEYS: tuple[str, ...] = (
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


def init_metric_sums(device: torch.device) -> dict[str, torch.Tensor]:
    """Allocate accumulator tensors for one prediction source."""
    return {
        key: torch.zeros((), device=device, dtype=torch.float64) for key in _METRIC_SUM_KEYS
    }


def finalize_metric_sums(sums: dict[str, torch.Tensor], n_patches: int) -> dict[str, float]:
    """Convert accumulator tensors into scalar metrics."""
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
        "contour_interval_m": float(_EVAL_CONTOUR_INTERVAL_M),
    }


def parse_patch_stem(stem: str) -> dict[str, int | str] | None:
    """Parse ``x_y_zone_country_year`` metadata when available."""
    parts = stem.split("_")
    if len(parts) != 5:
        return None
    x, y, zone, country, year = parts
    try:
        return {
            "x": int(x),
            "y": int(y),
            "zone": int(zone),
            "country": country,
            "year": int(year),
        }
    except ValueError:
        return None


def get_numeric(row: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric
    return None


def canonicalize_patch_table_row(row: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    out["p90_slope"] = get_numeric(row, "p90_slope", "slope_p90")
    out["frac_shore"] = get_numeric(row, "frac_shore", "shore_mean")
    out["frac_water"] = get_numeric(row, "frac_water", "water_mean")
    out["has_edge"] = get_numeric(row, "has_edge")
    out["frac_building"] = get_numeric(row, "frac_building", "bld_mean")
    out["mean_uncert"] = get_numeric(row, "mean_uncert", "U_lr10_mean")
    out["mean_W"] = get_numeric(row, "mean_W", "weight_mean")
    out["valid_frac"] = get_numeric(row, "valid_frac", "weight_valid_mean")
    out["gt_coverage_mean"] = get_numeric(row, "gt_coverage_mean")
    out["resid_scale"] = get_numeric(row, "resid_scale", "residAbs_p95")
    out["relief"] = get_numeric(row, "relief")
    if row.get("stratum_id") not in (None, ""):
        out["stratum_id"] = row["stratum_id"]
    return out


def compute_quantile_cutpoints(values: list[float], num_bins: int = 4) -> list[float]:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean or num_bins < 2:
        return []
    cutpoints: list[float] = []
    n = len(clean)
    for idx in range(1, num_bins):
        rank = idx * (n - 1) / num_bins
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            value = clean[lo]
        else:
            frac = rank - lo
            value = clean[lo] * (1.0 - frac) + clean[hi] * frac
        cutpoints.append(value)
    return cutpoints


def assign_uncertainty_bin(value: float | None, cutpoints: list[float]) -> str:
    if value is None or not math.isfinite(value):
        return "missing"
    for idx, cutpoint in enumerate(cutpoints, start=1):
        if value <= cutpoint:
            return f"q{idx}"
    return f"q{len(cutpoints) + 1}"


def assign_slope_bin(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "missing"
    if value <= 2.0:
        return "0-2"
    if value <= 5.0:
        return "2-5"
    if value <= 10.0:
        return "5-10"
    if value <= 20.0:
        return "10-20"
    return ">20"


def assign_hydrology_bin(frac_shore: float | None, frac_water: float | None, has_edge: float | None) -> str:
    shore = frac_shore if frac_shore is not None and math.isfinite(frac_shore) else 0.0
    water = frac_water if frac_water is not None and math.isfinite(frac_water) else 0.0
    edge = has_edge if has_edge is not None and math.isfinite(has_edge) else 0.0
    if shore > 0:
        return "shore"
    if water > 0:
        return "water"
    if edge > 0:
        return "edge"
    return "dry"


def assign_building_bin(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "missing"
    if value <= 0:
        return "0"
    if value <= 0.05:
        return "0-0.05"
    if value <= 0.25:
        return "0.05-0.25"
    return ">0.25"


def build_patch_table_context(
    patch_table: dict[str, dict[str, object]],
    stems: list[str],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    matched_rows = {
        stem: canonicalize_patch_table_row(patch_table[stem]) for stem in stems if stem in patch_table
    }
    uncertainty_cutpoints = compute_quantile_cutpoints(
        [float(row["mean_uncert"]) for row in matched_rows.values() if row.get("mean_uncert") is not None],
        num_bins=4,
    )
    context: dict[str, dict[str, object]] = {}
    for stem in stems:
        row = matched_rows.get(stem, {})
        enriched = dict(row)
        enriched["slope_bin"] = assign_slope_bin(get_numeric(row, "p90_slope"))
        enriched["hydrology_bin"] = assign_hydrology_bin(
            get_numeric(row, "frac_shore"),
            get_numeric(row, "frac_water"),
            get_numeric(row, "has_edge"),
        )
        enriched["building_bin"] = assign_building_bin(get_numeric(row, "frac_building"))
        enriched["uncertainty_bin"] = assign_uncertainty_bin(get_numeric(row, "mean_uncert"), uncertainty_cutpoints)
        context[stem] = enriched
    return context, {"uncertainty_cutpoints": uncertainty_cutpoints, "strata_fields": list(STRATA_FIELDS)}


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


_STRATA_SUM_KEYS: tuple[str, ...] = (
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
                sums = grouped.setdefault(
                    group,
                    {key: 0.0 for key in _STRATA_SUM_KEYS},
                )
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
                group: finalize_python_metric_sums(sums, counts[group])
                for group, sums in sorted(grouped.items())
            }
        out[source] = field_map
    return out


def compute_per_patch_metrics(
    pred: torch.Tensor,
    z_gt: torch.Tensor,
    w: torch.Tensor,
) -> dict[str, list[float]]:
    """Return weighted metrics for each patch in the batch."""
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
    sdf_pred = contour_sdf(pred_f, _EVAL_CONTOUR_INTERVAL_M)
    sdf_gt = contour_sdf(z_gt_f, _EVAL_CONTOUR_INTERVAL_M)

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


def pct_improvement(baseline: float, improved: float) -> float:
    """Positive values indicate lower error than baseline."""
    denom = max(abs(baseline), 1e-12)
    return 100.0 * (baseline - improved) / denom


def candidate_path_for_stem(
    stem: str,
    *,
    candidate_root: Path,
    candidate_product: str | None,
) -> Path:
    """Match ``LocalDemPatchDataset`` candidate path resolution."""
    if candidate_product:
        return candidate_root / candidate_product / f"{stem}.tif"
    direct = candidate_root / f"{stem}.tif"
    if direct.is_file():
        return direct
    return candidate_root / stem / f"{stem}.tif"


def filter_stems_with_candidates(
    stems: list[str],
    *,
    candidate_root: Path,
    candidate_product: str | None,
) -> tuple[list[str], int]:
    """Keep only stems that have a candidate raster on disk."""
    kept: list[str] = []
    skipped = 0
    for stem in stems:
        if candidate_path_for_stem(
            stem,
            candidate_root=candidate_root,
            candidate_product=candidate_product,
        ).is_file():
            kept.append(stem)
        else:
            skipped += 1
    return kept, skipped


def add_customer_example_fields(
    row: dict[str, object],
    *,
    baseline_source: str,
    improved_source: str,
) -> None:
    """Attach improvement columns and a simple composite ranking score."""
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


@torch.no_grad()
def run_eval(
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    prediction_sources: list[str],
    model: torch.nn.Module | None = None,
    blend_model: torch.nn.Module | None = None,
    blend_weight: float = 0.25,
    collect_per_patch: bool = False,
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    """Aggregate weighted MAE/RMSE and optionally collect per-patch rows."""
    use_cuda_amp = use_amp and device.type == "cuda"
    sources = list(dict.fromkeys(prediction_sources))
    if "model" in sources:
        if model is None:
            raise ValueError("model prediction source requires a loaded model")
        model.eval()
        if blend_model is not None:
            blend_model.eval()

    sums_by_source = {source: init_metric_sums(device) for source in sources}
    n_patches = 0
    per_patch_rows: list[dict[str, object]] = []

    progress = tqdm(
        loader,
        desc="eval " + ",".join(sources),
        total=len(loader),
        dynamic_ncols=True,
    )
    for batch in progress:
        z_lr = batch["z_lr"].to(device, non_blocking=True)
        z_gt = batch["z_gt"].to(device, non_blocking=True)
        w = batch["w"].to(device, non_blocking=True)
        pred_by_source: dict[str, torch.Tensor] = {}
        weight_by_source: dict[str, torch.Tensor] = {}

        if "model" in sources:
            x_dem = batch["x_dem"].to(device, non_blocking=True)
            x_ae = batch["x_ae"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                z_hat = model(x_dem, x_ae, z_lr)
                if blend_model is not None:
                    z_hat_blend = blend_model(x_dem, x_ae, z_lr)
                    residual_a = z_hat - z_lr
                    residual_b = z_hat_blend - z_lr
                    z_hat = z_lr + (1.0 - blend_weight) * residual_a + blend_weight * residual_b
                pred_by_source["model"] = z_hat
            weight_by_source["model"] = w
        if "z_lr" in sources:
            pred_by_source["z_lr"] = z_lr
            weight_by_source["z_lr"] = w
        if "raster" in sources:
            pred_by_source["raster"] = batch["z_candidate"].to(device, non_blocking=True)
            weight_by_source["raster"] = w * batch["z_candidate_valid"].to(device, non_blocking=True)

        for source in sources:
            update_metric_sums(pred_by_source[source], z_gt, weight_by_source[source], sums_by_source[source])

        if collect_per_patch:
            per_source_patch_metrics = {
                source: compute_per_patch_metrics(pred_by_source[source], z_gt, weight_by_source[source])
                for source in sources
            }
            stems = list(batch["stem"])
            for idx, stem in enumerate(stems):
                row: dict[str, object] = {"stem": stem}
                meta = parse_patch_stem(stem)
                if meta is not None:
                    row.update(meta)
                for source in sources:
                    for metric_name, values in per_source_patch_metrics[source].items():
                        row[f"{source}_{metric_name}"] = float(values[idx])
                if "model" in sources and "z_lr" in sources:
                    add_customer_example_fields(row, baseline_source="z_lr", improved_source="model")
                if "model" in sources and "raster" in sources:
                    add_customer_example_fields(row, baseline_source="raster", improved_source="model")
                if "raster" in sources and "z_lr" in sources:
                    add_customer_example_fields(row, baseline_source="z_lr", improved_source="raster")
                per_patch_rows.append(row)

        n_patches += z_gt.shape[0]

        postfix: dict[str, str | int] = {"patches": n_patches}
        for source in sources:
            sw = float(sums_by_source[source]["sum_w"].clamp(min=1e-12))
            postfix[f"{source}_rmse"] = f"{float(torch.sqrt(sums_by_source[source]['sum_sq_e'] / sw)):.4f}"
        progress.set_postfix(postfix)

    metrics_by_source = {
        source: finalize_metric_sums(sums, n_patches) for source, sums in sums_by_source.items()
    }
    return metrics_by_source, per_patch_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prediction-source",
        choices=("model", "z_lr", "raster"),
        nargs="+",
        default=["model"],
        help="One or more sources to evaluate in a single pass",
    )
    p.add_argument("--checkpoint", type=Path, default=Path("dem_film_unet.pt"))
    p.add_argument(
        "--blend-checkpoint",
        type=Path,
        default=None,
        help="Optional second checkpoint for residual blending with --checkpoint",
    )
    p.add_argument(
        "--blend-weight",
        type=float,
        default=0.25,
        help="Residual blend weight for --blend-checkpoint (default: 0.25)",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Override training root (default: read from checkpoint)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="One patch stem per line (recommended for holdout evaluation)",
    )
    p.add_argument(
        "--list-from-root",
        action="store_true",
        help="Use all stems found under data-root (expensive on large datasets)",
    )
    p.add_argument("--max-patches", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--candidate-root",
        type=Path,
        default=None,
        help="Root directory for per-patch comparison rasters",
    )
    p.add_argument(
        "--candidate-product",
        default=None,
        help="Optional subdirectory name under --candidate-root (for example: fabdem)",
    )
    p.add_argument(
        "--candidate-band",
        type=int,
        default=1,
        help="1-based band index for comparison rasters",
    )
    p.add_argument(
        "--precomputed-weight",
        action="store_true",
        help="Match training if you trained with precomputed stack weight band",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write metrics to this path",
    )
    p.add_argument(
        "--per-patch-json",
        type=Path,
        default=None,
        help="Write one JSON row per patch with per-source metrics and improvement columns",
    )
    p.add_argument(
        "--patch-table",
        type=Path,
        default=None,
        help="Optional CSV/JSON/GeoJSON patch-summary table keyed by patch stem",
    )
    p.add_argument(
        "--stratified-json",
        type=Path,
        default=None,
        help="Optional JSON file for stratified summaries derived from --patch-table",
    )
    p.add_argument(
        "--arch",
        choices=ARCH_CHOICES,
        default=None,
        help="Model architecture override (default: checkpoint args.arch or film_unet)",
    )
    p.add_argument(
        "--contour-interval",
        type=float,
        default=DEFAULT_CONTOUR_INTERVAL_M,
        help="Contour interval in meters for SDF metric (default: 10.0)",
    )
    p.add_argument(
        "--blend-arch",
        choices=ARCH_CHOICES,
        default=None,
        help="Architecture override for --blend-checkpoint (default: checkpoint args.arch or film_unet)",
    )
    args = p.parse_args()
    prediction_sources = list(dict.fromkeys(args.prediction_source))
    set_eval_contour_interval(args.contour_interval)

    ckpt = None
    state = None
    ckpt_arch = None
    if "model" in prediction_sources:
        if not args.checkpoint.is_file():
            log.error("Checkpoint not found: %s", args.checkpoint)
            sys.exit(1)
        load_kw: dict = {"map_location": "cpu"}
        try:
            ckpt = torch.load(args.checkpoint, weights_only=False, **load_kw)
        except TypeError:
            ckpt = torch.load(args.checkpoint, **load_kw)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if isinstance(ckpt, dict):
            ckpt_args = ckpt.get("args")
            if isinstance(ckpt_args, dict):
                ckpt_arch = ckpt_args.get("arch")
        if args.blend_checkpoint is not None and not args.blend_checkpoint.is_file():
            log.error("Blend checkpoint not found: %s", args.blend_checkpoint)
            sys.exit(1)
    if not (0.0 <= args.blend_weight <= 1.0):
        log.error("--blend-weight must be in [0, 1], got %.6f", args.blend_weight)
        sys.exit(1)

    data_root = args.data_root or (
        ckpt.get("data_root") if isinstance(ckpt, dict) else None
    )
    if data_root is None:
        log.error(
            "Set --data-root explicitly, or use --prediction-source model with a checkpoint "
            "that includes data_root."
        )
        sys.exit(1)
    if "raster" in prediction_sources and args.candidate_root is None:
        log.error("Set --candidate-root when using --prediction-source raster.")
        sys.exit(1)
    if args.stratified_json is not None and args.patch_table is None:
        log.error("Set --patch-table when using --stratified-json.")
        sys.exit(1)

    stems = None
    if args.manifest is not None:
        stems = load_patch_stems_manifest(args.manifest)
        log.info("Manifest %s: %d stems", args.manifest, len(stems))
    elif args.list_from_root:
        if "raster" in prediction_sources:
            stems = list_patch_stems(data_root)
            log.info("Listed %d stems from %s", len(stems), data_root)
        else:
            stems = None
            log.info("Listing stems from %s", data_root)
    else:
        log.error("Provide --manifest (holdout list) or --list-from-root.")
        sys.exit(1)

    if "raster" in prediction_sources:
        candidate_root = Path(args.candidate_root)
        if args.candidate_product is not None:
            product_dir = candidate_root / args.candidate_product
            if not product_dir.is_dir():
                log.error("Candidate product directory not found: %s", product_dir)
                sys.exit(1)
        elif not candidate_root.is_dir():
            log.error("Candidate root directory not found: %s", candidate_root)
            sys.exit(1)

        if stems is None:
            stems = list_patch_stems(data_root)
            log.info("Listed %d stems from %s for candidate filtering", len(stems), data_root)

        before = len(stems)
        stems, skipped = filter_stems_with_candidates(
            stems,
            candidate_root=candidate_root,
            candidate_product=args.candidate_product,
        )
        log.info(
            "Raster candidate filtering kept %d/%d stems (%d missing candidate files)",
            len(stems),
            before,
            skipped,
        )
        if not stems:
            log.error("No stems have candidate rasters under %s", candidate_root)
            sys.exit(1)

    ds = LocalDemPatchDataset(
        data_root,
        patch_stems=stems,
        use_precomputed_weight=args.precomputed_weight,
        load_ae=("model" in prediction_sources),
        candidate_root=args.candidate_root,
        candidate_product=args.candidate_product,
        candidate_band=args.candidate_band,
        max_patches=args.max_patches,
    )
    if len(ds) == 0:
        log.error("No patches to evaluate.")
        sys.exit(1)
    log.info("Evaluating %d patches", len(ds))

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_dem_batch,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    blend_model = None
    if "model" in prediction_sources:
        arch = args.arch or ckpt_arch or ARCH_FILM
        model = create_model(arch).to(device)
        model.load_state_dict(state, strict=True)
        log.info("Model architecture: %s", arch)
        if args.blend_checkpoint is not None:
            load_kw: dict = {"map_location": "cpu"}
            try:
                blend_ckpt = torch.load(args.blend_checkpoint, weights_only=False, **load_kw)
            except TypeError:
                blend_ckpt = torch.load(args.blend_checkpoint, **load_kw)
            blend_state = (
                blend_ckpt["model"]
                if isinstance(blend_ckpt, dict) and "model" in blend_ckpt
                else blend_ckpt
            )
            blend_ckpt_arch = None
            if isinstance(blend_ckpt, dict):
                blend_ckpt_args = blend_ckpt.get("args")
                if isinstance(blend_ckpt_args, dict):
                    blend_ckpt_arch = blend_ckpt_args.get("arch")
            blend_arch = args.blend_arch or blend_ckpt_arch or ARCH_FILM
            blend_model = create_model(blend_arch).to(device)
            blend_model.load_state_dict(blend_state, strict=True)
            log.info(
                "Blend enabled: checkpoint=%s arch=%s weight=%.3f",
                args.blend_checkpoint,
                blend_arch,
                args.blend_weight,
            )
    log.info("Device: %s", device)
    log.info("Prediction sources: %s", ", ".join(prediction_sources))

    collect_per_patch = (
        args.per_patch_json is not None or args.patch_table is not None or args.stratified_json is not None
    )
    metrics_by_source, per_patch_rows = run_eval(
        loader,
        device,
        use_amp=args.amp,
        prediction_sources=prediction_sources,
        model=model,
        blend_model=blend_model,
        blend_weight=args.blend_weight,
        collect_per_patch=collect_per_patch,
    )

    patch_table_match_summary = None
    stratification_payload = None
    stratified_metrics_by_source = None
    if args.patch_table is not None:
        eval_stems = stems if stems is not None else list_patch_stems(data_root)
        patch_table = load_patch_table(args.patch_table, allowed_stems=set(eval_stems))
        patch_context, stratification_payload = build_patch_table_context(
            patch_table,
            [str(row["stem"]) for row in per_patch_rows],
        )
        matched = 0
        for row in per_patch_rows:
            stem = str(row["stem"])
            context = patch_context.get(stem, {})
            if stem in patch_table:
                matched += 1
            for field in PATCH_TABLE_FIELDS:
                row[field] = context.get(field)
            for field in STRATA_FIELDS:
                row[field] = context.get(field, "missing")
        patch_table_match_summary = {
            "matched_rows": matched,
            "missing_rows": len(per_patch_rows) - matched,
        }
        stratified_metrics_by_source = compute_stratified_metrics(per_patch_rows, prediction_sources)
        log.info(
            "Patch table join matched %d/%d evaluation rows",
            matched,
            len(per_patch_rows),
        )

    for source in prediction_sources:
        metrics = metrics_by_source[source]
        log.info("--- metrics for %s (weighted by W, same as training loss) ---", source)
        log.info("patches:     %d", int(metrics["n_patches"]))
        log.info("sum(W):      %.6g", metrics["sum_weights"])
        log.info("elev bias_w: %.6f m", metrics["elev_bias_w"])
        log.info("elev MAE_w:  %.6f m", metrics["elev_mae_w"])
        log.info("elev RMSE_w: %.6f m", metrics["elev_rmse_w"])
        log.info("slope MAE_w: %.6f (dz/dhoriz, same as loss_dem)", metrics["slope_mae_w"])
        log.info("slope RMSE_w:%.6f", metrics["slope_rmse_w"])
        log.info("slope MAE_deg_w:  %.6f deg", metrics["slope_mae_deg_w"])
        log.info("slope RMSE_deg_w: %.6f deg", metrics["slope_rmse_deg_w"])
        log.info("grad_x RMSE_w:   %.6f (rise/run)", metrics["grad_x_rmse_w"])
        log.info("grad_y RMSE_w:   %.6f (rise/run)", metrics["grad_y_rmse_w"])
        log.info("laplacian RMSE_w:%.6f (m/m^2)", metrics["laplacian_rmse_w"])
        log.info(
            "sdf RMSE_w:      %.6f m (contour interval=%.3g m)",
            metrics["sdf_rmse_w"],
            metrics["contour_interval_m"],
        )

    payload = {
        "prediction_source": prediction_sources,
        "checkpoint": str(args.checkpoint),
        "blend_checkpoint": str(args.blend_checkpoint) if args.blend_checkpoint else None,
        "blend_weight": args.blend_weight,
        "data_root": data_root,
        "manifest": str(args.manifest) if args.manifest else None,
        "list_from_root": bool(args.list_from_root),
        "candidate_root": str(args.candidate_root) if args.candidate_root else None,
        "candidate_product": args.candidate_product,
        "candidate_band": args.candidate_band,
        "precomputed_weight": args.precomputed_weight,
        "patch_table": str(args.patch_table) if args.patch_table else None,
        "patch_table_match_summary": patch_table_match_summary,
        "stratification": stratification_payload,
        "metrics_by_source": metrics_by_source,
        "stratified_metrics_by_source": stratified_metrics_by_source,
    }
    if len(prediction_sources) == 1:
        payload.update(metrics_by_source[prediction_sources[0]])
    if args.output_json:
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Wrote %s", args.output_json)
    if args.per_patch_json:
        args.per_patch_json.write_text(json.dumps(per_patch_rows, indent=2), encoding="utf-8")
        log.info("Wrote %s", args.per_patch_json)
    if args.stratified_json:
        stratified_payload = {
            "patch_table": str(args.patch_table) if args.patch_table else None,
            "patch_table_match_summary": patch_table_match_summary,
            "stratification": stratification_payload,
            "stratified_metrics_by_source": stratified_metrics_by_source,
        }
        args.stratified_json.write_text(json.dumps(stratified_payload, indent=2), encoding="utf-8")
        log.info("Wrote %s", args.stratified_json)


if __name__ == "__main__":
    main()
