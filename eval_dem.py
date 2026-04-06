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
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dem_film_unet import DemFilmUNet, slope_to_degrees, terrain_slope
from local_patch_dataset import (
    LocalDemPatchDataset,
    collate_dem_batch,
    list_patch_stems,
    load_patch_stems_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_dem")


def update_metric_sums(
    pred: torch.Tensor,
    z_gt: torch.Tensor,
    w: torch.Tensor,
    sums: dict[str, torch.Tensor],
) -> None:
    """Accumulate weighted elevation and slope errors for a prediction tensor."""
    s_pred = terrain_slope(pred.float())
    s_gt = terrain_slope(z_gt.float())
    s_pred_deg = slope_to_degrees(s_pred)
    s_gt_deg = slope_to_degrees(s_gt)

    w64 = w.double()
    e = (pred.double() - z_gt.double()).squeeze(1)
    ds = (s_pred.double() - s_gt.double()).squeeze(1)
    ds_deg = (s_pred_deg.double() - s_gt_deg.double()).squeeze(1)
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


def init_metric_sums(device: torch.device) -> dict[str, torch.Tensor]:
    """Allocate accumulator tensors for one prediction source."""
    return {
        "sum_w": torch.zeros((), device=device, dtype=torch.float64),
        "sum_e": torch.zeros((), device=device, dtype=torch.float64),
        "sum_abs_e": torch.zeros((), device=device, dtype=torch.float64),
        "sum_sq_e": torch.zeros((), device=device, dtype=torch.float64),
        "sum_abs_ds": torch.zeros((), device=device, dtype=torch.float64),
        "sum_sq_ds": torch.zeros((), device=device, dtype=torch.float64),
        "sum_abs_ds_deg": torch.zeros((), device=device, dtype=torch.float64),
        "sum_sq_ds_deg": torch.zeros((), device=device, dtype=torch.float64),
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


def compute_per_patch_metrics(
    pred: torch.Tensor,
    z_gt: torch.Tensor,
    w: torch.Tensor,
) -> dict[str, list[float]]:
    """Return weighted metrics for each patch in the batch."""
    s_pred = terrain_slope(pred.float())
    s_gt = terrain_slope(z_gt.float())
    s_pred_deg = slope_to_degrees(s_pred)
    s_gt_deg = slope_to_degrees(s_gt)

    w64 = w.double()
    e = (pred.double() - z_gt.double()).squeeze(1)
    ds = (s_pred.double() - s_gt.double()).squeeze(1)
    ds_deg = (s_pred_deg.double() - s_gt_deg.double()).squeeze(1)
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
    model: DemFilmUNet | None = None,
    collect_per_patch: bool = False,
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    """Aggregate weighted MAE/RMSE and optionally collect per-patch rows."""
    use_cuda_amp = use_amp and device.type == "cuda"
    sources = list(dict.fromkeys(prediction_sources))
    if "model" in sources:
        if model is None:
            raise ValueError("model prediction source requires a loaded model")
        model.eval()

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
                pred_by_source["model"] = model(x_dem, x_ae, z_lr)
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
    args = p.parse_args()
    prediction_sources = list(dict.fromkeys(args.prediction_source))

    ckpt = None
    state = None
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
    if "model" in prediction_sources:
        model = DemFilmUNet().to(device)
        model.load_state_dict(state, strict=True)
    log.info("Device: %s", device)
    log.info("Prediction sources: %s", ", ".join(prediction_sources))

    metrics_by_source, per_patch_rows = run_eval(
        loader,
        device,
        use_amp=args.amp,
        prediction_sources=prediction_sources,
        model=model,
        collect_per_patch=(args.per_patch_json is not None),
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

    payload = {
        "prediction_source": prediction_sources,
        "checkpoint": str(args.checkpoint),
        "data_root": data_root,
        "manifest": str(args.manifest) if args.manifest else None,
        "list_from_root": bool(args.list_from_root),
        "candidate_root": str(args.candidate_root) if args.candidate_root else None,
        "candidate_product": args.candidate_product,
        "candidate_band": args.candidate_band,
        "precomputed_weight": args.precomputed_weight,
        "metrics_by_source": metrics_by_source,
    }
    if len(prediction_sources) == 1:
        payload.update(metrics_by_source[prediction_sources[0]])
    if args.output_json:
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Wrote %s", args.output_json)
    if args.per_patch_json:
        args.per_patch_json.write_text(json.dumps(per_patch_rows, indent=2), encoding="utf-8")
        log.info("Wrote %s", args.per_patch_json)


if __name__ == "__main__":
    main()
