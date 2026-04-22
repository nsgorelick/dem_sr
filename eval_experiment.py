#!/usr/bin/env python3
"""Evaluate DEM experiments (model, z_lr baseline, stage_a, etc.)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from core.checkpoints import extract_model_state, load_checkpoint
from core.config import (
    add_shared_experiment_args,
    apply_namespace_preset_defaults,
    config_to_dict,
    export_experiment_cli_config,
    resolve_config,
)
from core.metrics import PATCH_TABLE_FIELDS, set_contour_interval
from core.reporting import build_eval_payload
from core.run_config import (
    load_run_config,
    resolve_description,
    section_defaults,
    standardized_eval_output_path,
)
from eval.engine import run_eval_epoch_multi_source, run_eval_epoch_multi_source_with_rows
from experiments.cli_registration import add_all_eval_experiment_args
from experiments.config_presets import get_preset, list_presets
from experiments.registry import create_experiment, list_experiments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_experiment")


def _as_path(value: str | Path | None) -> Path | None:
    if value is None or isinstance(value, Path):
        return value
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    experiments = list_experiments()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="baseline",
        choices=experiments,
        help="Experiment key.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Shared JSON run config file")
    parser.add_argument("--description", default=None, help="Human-readable run description")
    parser.add_argument(
        "--preset",
        default="baseline",
        choices=list_presets("eval"),
        help="Named eval config preset (applies only to default-valued fields).",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("experiment_model.pt"))
    parser.add_argument(
        "--prediction-source",
        choices=("model", "z_lr", "stage_a"),
        nargs="+",
        default=["model"],
        help="One or more prediction sources to evaluate in a single pass.",
    )
    add_shared_experiment_args(parser)
    parser.add_argument(
        "--list-from-root",
        action="store_true",
        help="Use all stems found under data-root",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write metrics to this path",
    )
    parser.add_argument(
        "--per-patch-json",
        type=Path,
        default=None,
        help="Write one JSON row per patch with per-source metrics and optional strata fields",
    )
    parser.add_argument(
        "--patch-table",
        type=Path,
        default=None,
        help="Optional CSV/JSON/GeoJSON patch-summary table keyed by patch stem",
    )
    parser.add_argument(
        "--stratified-json",
        type=Path,
        default=None,
        help="Optional JSON file for stratified summaries derived from --patch-table",
    )
    parser.add_argument(
        "--sliding-window-tile-size",
        type=int,
        default=None,
        help="Optional model inference tile size for overlap/blend stitching.",
    )
    parser.add_argument(
        "--sliding-window-overlap",
        type=int,
        default=0,
        help="Pixel overlap between inference windows when sliding-window inference is enabled.",
    )
    add_all_eval_experiment_args(parser)
    return parser


def main() -> None:
    from local_patch_dataset import (
        LocalDemPatchDataset,
        collate_dem_batch,
        list_patch_stems,
        load_patch_stems_manifest,
    )
    from core.patch_table import load_patch_table
    from core.metrics import STRATA_FIELDS, build_patch_table_context, compute_stratified_metrics

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    run_config: dict[str, object] = {}
    if pre_args.config is not None:
        run_config = load_run_config(pre_args.config)

    parser = build_parser()
    if run_config:
        parser.set_defaults(**section_defaults(run_config, "eval"))
    args, _ = parser.parse_known_args()
    args.config = _as_path(args.config)
    args.manifest = _as_path(args.manifest)
    args.checkpoint = _as_path(args.checkpoint)
    args.output_json = _as_path(args.output_json)
    args.per_patch_json = _as_path(args.per_patch_json)
    args.patch_table = _as_path(args.patch_table)
    args.stratified_json = _as_path(args.stratified_json)
    apply_namespace_preset_defaults(args, parser, get_preset("eval", args.preset))
    experiment = create_experiment(args.experiment)
    experiment.coerce_eval_arg_paths(args)

    prediction_sources = list(dict.fromkeys(args.prediction_source))
    if args.sliding_window_tile_size is not None:
        if args.sliding_window_tile_size <= 0:
            raise SystemExit("--sliding-window-tile-size must be positive")
        if args.sliding_window_overlap < 0:
            raise SystemExit("--sliding-window-overlap must be >= 0")
        if args.sliding_window_overlap >= args.sliding_window_tile_size:
            raise SystemExit("--sliding-window-overlap must be smaller than --sliding-window-tile-size")
    checkpoint = None
    model_state = None
    checkpoint_args: dict[str, object] = {}
    requires_model = "model" in prediction_sources or "stage_a" in prediction_sources
    if requires_model:
        checkpoint = load_checkpoint(args.checkpoint)
        model_state = extract_model_state(checkpoint)
        checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        if not isinstance(checkpoint_args, dict):
            checkpoint_args = {}

    checkpoint_root = checkpoint.get("data_root") if isinstance(checkpoint, dict) else None
    cfg = resolve_config(args, default_data_root=checkpoint_root)
    data_root = cfg.data_root
    if data_root is None:
        raise SystemExit("Set --data-root or provide checkpoint with data_root.")

    if cfg.manifest is not None:
        stems = load_patch_stems_manifest(cfg.manifest)
        log.info("Manifest %s: %d stems", cfg.manifest, len(stems))
    elif cfg.list_from_root:
        stems = list_patch_stems(str(data_root))
        log.info("Listed %d stems from %s", len(stems), data_root)
    else:
        raise SystemExit("Provide --manifest or --list-from-root.")

    ds = LocalDemPatchDataset(
        str(data_root),
        patch_stems=stems,
        use_precomputed_weight=cfg.precomputed_weight,
        load_ae=True,
        max_patches=cfg.max_patches,
        tile_size=cfg.tile_size,
        supervision_crop_size=None,
    )
    if len(ds) == 0:
        raise SystemExit("No patches to evaluate.")

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=collate_dem_batch,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    set_contour_interval(cfg.contour_interval)

    model = None
    if requires_model:
        model_cfg = dict(checkpoint_args)
        model_cfg.update(vars(args) | config_to_dict(cfg))
        model = experiment.build_model(model_cfg).to(device)
        assert model_state is not None
        model.load_state_dict(model_state, strict=True)

    collect_per_patch = args.per_patch_json is not None or args.patch_table is not None or args.stratified_json is not None
    per_patch_rows: list[dict[str, object]] = []
    if collect_per_patch:
        metrics_by_source, per_patch_rows = run_eval_epoch_multi_source_with_rows(
            model=model,
            loader=loader,
            device=device,
            model_forward=experiment.model_forward,
            amp_enabled=amp_enabled,
            prediction_sources=prediction_sources,
            sliding_window_tile_size=args.sliding_window_tile_size,
            sliding_window_overlap=args.sliding_window_overlap,
        )
    else:
        metrics_by_source = run_eval_epoch_multi_source(
            model=model,
            loader=loader,
            device=device,
            model_forward=experiment.model_forward,
            amp_enabled=amp_enabled,
            prediction_sources=prediction_sources,
            sliding_window_tile_size=args.sliding_window_tile_size,
            sliding_window_overlap=args.sliding_window_overlap,
        )
    for source in prediction_sources:
        metrics = metrics_by_source[source]
        log.info(
            "eval[%s] patches=%d elev_rmse_w=%.6f slope_rmse_w=%.6f sdf_rmse_w=%.6f",
            source,
            int(metrics["n_patches"]),
            metrics["elev_rmse_w"],
            metrics["slope_rmse_w"],
            metrics["sdf_rmse_w"],
        )

    patch_table_match_summary = None
    stratification_payload = None
    stratified_metrics_by_source = None
    if args.patch_table is not None:
        eval_stems = stems if stems is not None else list_patch_stems(str(data_root))
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

    payload = build_eval_payload(
        experiment=args.experiment,
        prediction_sources=prediction_sources,
        checkpoint=str(args.checkpoint) if requires_model else None,
        data_root=str(data_root),
        manifest=str(cfg.manifest) if cfg.manifest else None,
        list_from_root=bool(cfg.list_from_root),
        contour_interval_m=float(cfg.contour_interval),
        metrics_by_source=metrics_by_source,
        config=export_experiment_cli_config(args, cfg),
    )
    payload["patch_table"] = str(args.patch_table) if args.patch_table else None
    payload["patch_table_match_summary"] = patch_table_match_summary
    payload["stratification"] = stratification_payload
    payload["stratified_metrics_by_source"] = stratified_metrics_by_source
    resolved_description = resolve_description(
        run_config if run_config else {},
        args.config if args.config is not None else Path("eval_experiment"),
        args.description,
    )
    payload["description"] = resolved_description
    output_json = args.output_json
    if output_json is None:
        output_json = standardized_eval_output_path(
            config_path=args.config,
            description=resolved_description,
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Wrote %s", output_json)
    if args.per_patch_json is not None:
        args.per_patch_json.write_text(json.dumps(per_patch_rows, indent=2), encoding="utf-8")
        log.info("Wrote %s", args.per_patch_json)
    if args.stratified_json is not None:
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

