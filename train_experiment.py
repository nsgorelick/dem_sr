#!/usr/bin/env python3
"""Train DEM experiments (baseline, two-stage, frequency-domain, etc.)."""

from __future__ import annotations

import argparse
import logging
import random
import time
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from core.checkpoints import load_checkpoint, make_training_checkpoint_payload, save_training_checkpoint
from core.config import (
    add_shared_experiment_args,
    apply_namespace_preset_defaults,
    config_to_dict,
    export_experiment_cli_config,
    resolve_config,
)
from core.pretraining import load_pretrained_encoder
from core.reporting import build_train_payload
from core.run_config import load_run_config, resolve_description, section_defaults
from experiments.cli_registration import add_all_train_experiment_args
from experiments.config_presets import get_preset, list_presets
from experiments.registry import create_experiment, list_experiments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_experiment")


def _as_path(value: str | Path | None) -> Path | None:
    if value is None or isinstance(value, Path):
        return value
    return Path(value)


def _apply_rotflip_augmentation(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply orientation-preserving rot/flip augmentations in-place."""
    k = int(torch.randint(0, 4, ()).item())
    flip_y = bool(torch.rand(()) < 0.5)
    flip_x = bool(torch.rand(()) < 0.5)
    for key, value in sample.items():
        if key == "stem" or not isinstance(value, torch.Tensor) or value.ndim < 3:
            continue
        out = value
        if k:
            out = torch.rot90(out, k=k, dims=(-2, -1))
        if flip_y:
            out = torch.flip(out, dims=(-2,))
        if flip_x:
            out = torch.flip(out, dims=(-1,))
        sample[key] = out.contiguous()
    return sample


def _resume_from_checkpoint(
    *,
    resume_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    history: dict[str, list[float]],
) -> tuple[int, dict[str, list[float]]]:
    """Restore training state from checkpoint and return (start_epoch, history)."""
    if not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    checkpoint = load_checkpoint(resume_path)
    if "model" not in checkpoint:
        raise TypeError(f"unsupported resume checkpoint format: {resume_path}")

    checkpoint_args = checkpoint.get("args", {})
    if isinstance(checkpoint_args, dict):
        checkpoint_arch = checkpoint_args.get("arch")
        if checkpoint_arch is not None and str(checkpoint_arch) != str(args.arch):
            raise ValueError(
                f"resume arch mismatch: checkpoint={checkpoint_arch} current={args.arch}"
            )

    model.load_state_dict(checkpoint["model"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except Exception:  # noqa: BLE001
            pass

    restored_history = history
    raw_history = checkpoint.get("history")
    if isinstance(raw_history, dict):
        train_loss = raw_history.get("train_loss")
        if isinstance(train_loss, list):
            restored_history["train_loss"] = [float(x) for x in train_loss]
        epoch_seconds = raw_history.get("epoch_seconds")
        if isinstance(epoch_seconds, list):
            restored_history["epoch_seconds"] = [float(x) for x in epoch_seconds]
        elif len(restored_history["epoch_seconds"]) < len(restored_history["train_loss"]):
            restored_history["epoch_seconds"] = restored_history["epoch_seconds"] + [0.0] * (
                len(restored_history["train_loss"]) - len(restored_history["epoch_seconds"])
            )

    start_epoch = int(checkpoint.get("epoch", 0))
    if start_epoch >= int(args.epochs):
        raise ValueError(
            f"resume epoch ({start_epoch}) must be less than requested epochs ({args.epochs})"
        )
    return start_epoch, restored_history


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
        choices=list_presets("train"),
        help="Named train config preset (applies only to default-valued fields).",
    )
    add_shared_experiment_args(parser)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", type=Path, default=Path("experiment_model.pt"))
    parser.add_argument("--output-json", type=Path, default=None, help="Write train payload to this path")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from experiment checkpoint")
    parser.add_argument(
        "--pretrained-encoder-checkpoint",
        type=Path,
        default=None,
        help="Optional SSL-pretrained encoder checkpoint to initialize model encoders.",
    )
    add_all_train_experiment_args(parser)
    parser.add_argument(
        "--augment-rotflip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply random 90-degree rotations and flips to training samples",
    )
    return parser


def main() -> None:
    from local_patch_dataset import (
        LocalDemPatchDataset,
        collate_dem_batch,
        list_patch_stems,
        load_patch_stems_manifest,
    )
    from train.engine import run_epoch

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    run_config: dict[str, object] = {}
    if pre_args.config is not None:
        run_config = load_run_config(pre_args.config)

    parser = build_parser()
    if run_config:
        parser.set_defaults(**section_defaults(run_config, "train"))
    args, _ = parser.parse_known_args()
    args.config = _as_path(args.config)
    args.manifest = _as_path(args.manifest)
    args.checkpoint_out = _as_path(args.checkpoint_out)
    args.output_json = _as_path(args.output_json)
    args.resume = _as_path(args.resume)
    args.pretrained_encoder_checkpoint = _as_path(args.pretrained_encoder_checkpoint)
    apply_namespace_preset_defaults(args, parser, get_preset("train", args.preset))
    cfg = resolve_config(args)
    experiment = create_experiment(args.experiment)
    experiment.coerce_train_arg_paths(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    resolved_description = resolve_description(
        run_config if run_config else {},
        args.config if args.config is not None else Path("train_experiment"),
        args.description,
    )

    if cfg.manifest is not None:
        stems = load_patch_stems_manifest(cfg.manifest)
        log.info("Loaded %d stems from %s", len(stems), cfg.manifest)
    else:
        stems = list_patch_stems(str(cfg.data_root))
        log.info("Listed %d stems under %s", len(stems), cfg.data_root)
    if cfg.max_patches is not None:
        stems = stems[: int(cfg.max_patches)]
    if not stems:
        raise SystemExit("No patches found for training.")

    train_subset = LocalDemPatchDataset(
        str(cfg.data_root),
        patch_stems=stems,
        use_precomputed_weight=cfg.precomputed_weight,
        load_ae=True,
        transform=_apply_rotflip_augmentation if args.augment_rotflip else None,
        tile_size=cfg.tile_size,
        supervision_crop_size=cfg.supervision_crop_size,
    )
    loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        collate_fn=collate_dem_batch,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    model_cfg = vars(args) | config_to_dict(cfg)
    model = experiment.build_model(model_cfg).to(device)
    if args.pretrained_encoder_checkpoint is not None:
        loaded, skipped = load_pretrained_encoder(model, args.pretrained_encoder_checkpoint)
        log.info(
            "Loaded pretrained encoder from %s (loaded=%d skipped=%d)",
            args.pretrained_encoder_checkpoint,
            loaded,
            skipped,
        )
    loss_fn = experiment.build_loss(model_cfg)
    optimizer = torch.optim.AdamW(experiment.iter_trainable_parameters(model), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: dict[str, list[float]] = {"train_loss": [], "epoch_seconds": []}
    start_epoch = 0
    if args.resume is not None:
        start_epoch, history = _resume_from_checkpoint(
            resume_path=args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            history=history,
        )
        log.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        metrics = run_epoch(
            model=model,
            loader=loader,
            device=device,
            model_forward=experiment.model_forward,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            train=True,
            progress_desc=f"train {epoch + 1}/{args.epochs}",
        )
        elapsed = time.perf_counter() - epoch_start
        history["train_loss"].append(float(metrics["loss"]))
        history["epoch_seconds"].append(float(elapsed))
        extra_log = experiment.train_epoch_log_suffix(metrics)
        log.info(
            "epoch %d/%d loss=%.6f elev=%.6f slope=%.6f %s(%.1fs)",
            epoch + 1,
            args.epochs,
            metrics.get("loss", float("nan")),
            metrics.get("elev", float("nan")),
            metrics.get("slope", float("nan")),
            extra_log,
            elapsed,
        )
        epoch_payload = make_training_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            data_root=str(cfg.data_root),
            epoch=epoch + 1,
            args=vars(args),
            history={"train_loss": history["train_loss"], "val_loss": []},
            train_size=len(train_subset),
            val_size=0,
        )
        epoch_out = args.checkpoint_out.with_name(
            f"{args.checkpoint_out.stem}_epoch_{epoch + 1:03d}{args.checkpoint_out.suffix}"
        )
        save_training_checkpoint(epoch_out, epoch_payload)
        log.info("Wrote %s", epoch_out)

    payload = make_training_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        data_root=str(cfg.data_root),
        epoch=args.epochs,
        args=vars(args),
        history={"train_loss": history["train_loss"], "val_loss": []},
        train_size=len(train_subset),
        val_size=0,
    )
    save_training_checkpoint(args.checkpoint_out, payload)
    log.info("Wrote %s", args.checkpoint_out)
    report = build_train_payload(
        experiment=args.experiment,
        checkpoint_out=str(args.checkpoint_out),
        data_root=str(cfg.data_root),
        epochs=args.epochs,
        history=history,
        train_size=len(train_subset),
        config=export_experiment_cli_config(args, cfg) | {"description": resolved_description},
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()

