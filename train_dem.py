#!/usr/bin/env python3
"""Train FiLM U-Net on GeoTIFF chips under a local training root."""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from dem_film_unet import (
    ARCH_CHOICES,
    ARCH_FILM,
    DEFAULT_CONTOUR_INTERVAL_M,
    LOSS_PRESET_BASELINE,
    LOSS_PRESET_CHOICES,
    create_model,
    loss_dem_preset,
)
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
log = logging.getLogger("train_dem")


def apply_rotflip_augmentation(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply simple orientation-preserving augmentations to every raster tensor."""
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


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    data_root: str,
    epoch: int,
    args: argparse.Namespace,
    history: dict[str, list[float | None]],
    train_size: int,
    val_size: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "data_root": data_root,
            "epoch": epoch,
            "args": vars(args),
            "history": history,
            "train_loss_curve": history["train_loss"],
            "val_loss_curve": history["val_loss"],
            "train_size": train_size,
            "val_size": val_size,
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        default="/data/training",
        help="Parent root with stack/ and ae/ subdirs",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional file: one patch stem per line",
    )
    p.add_argument("--max-patches", type=int, default=None, help="Cap dataset size (debug)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-slope", type=float, default=0.5)
    p.add_argument("--guidance-dropout", type=float, default=0.3)
    p.add_argument(
        "--augment-rotflip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply random 90-degree rotations and flips to training samples",
    )
    p.add_argument("--val-fraction", type=float, default=0.0, help="Random val split (0=no val)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="torch.cuda.amp")
    p.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("dem_film_unet.pt"),
        help="Final checkpoint path",
    )
    p.add_argument(
        "--precomputed-weight",
        action="store_true",
        help="Use stack weight band instead of recomputing W",
    )
    p.add_argument(
        "--arch",
        choices=ARCH_CHOICES,
        default=ARCH_FILM,
        help="Model architecture variant",
    )
    p.add_argument(
        "--loss-preset",
        choices=LOSS_PRESET_CHOICES,
        default=LOSS_PRESET_BASELINE,
        help="Loss configuration: baseline | geom | multitask | contour",
    )
    p.add_argument(
        "--contour-interval",
        type=float,
        default=DEFAULT_CONTOUR_INTERVAL_M,
        help="Contour interval in meters for SDF / soft-contour losses",
    )
    p.add_argument("--lambda-grad", type=float, default=0.25, help="Weight for dx/dy gradient L1")
    p.add_argument("--lambda-curv", type=float, default=0.1, help="Weight for Laplacian L1")
    p.add_argument("--lambda-ms", type=float, default=0.5, help="Weight for 2x multi-scale elev L1")
    p.add_argument("--lambda-sdf", type=float, default=0.5, help="Weight for contour SDF L1")
    p.add_argument(
        "--lambda-contour",
        type=float,
        default=0.25,
        help="Weight for soft contour-indicator L1",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a checkpoint written by this script",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    stems = None
    if args.manifest is not None:
        stems = load_patch_stems_manifest(args.manifest)
        log.info("Loaded %d stems from %s", len(stems), args.manifest)
    else:
        stems = list_patch_stems(args.data_root)
        log.info("Listed %d stems under %s", len(stems), args.data_root)

    if args.max_patches is not None:
        stems = stems[: int(args.max_patches)]
    n_total = len(stems)
    if n_total == 0:
        log.error("No patches found. Check the local data root and manifests.")
        sys.exit(1)
    if args.max_patches is not None:
        log.info("Capped to %d patches", n_total)

    val_frac = max(0.0, min(0.5, args.val_fraction))
    if val_frac > 0 and n_total >= 2:
        rng = random.Random(args.seed)
        order = list(range(n_total))
        rng.shuffle(order)
        n_val = max(1, int(round(n_total * val_frac)))
        n_val = min(n_val, n_total - 1)
        val_set = frozenset(order[:n_val])
        train_stems = [stems[i] for i in range(n_total) if i not in val_set]
        val_stems = [stems[i] for i in range(n_total) if i in val_set]
        log.info("Random split: train=%d val=%d", len(train_stems), len(val_stems))
    else:
        train_stems = stems
        val_stems = []

    train_subset = LocalDemPatchDataset(
        args.data_root,
        patch_stems=train_stems,
        use_precomputed_weight=args.precomputed_weight,
        load_ae=True,
        transform=apply_rotflip_augmentation if args.augment_rotflip else None,
    )
    val_subset = (
        LocalDemPatchDataset(
            args.data_root,
            patch_stems=val_stems,
            use_precomputed_weight=args.precomputed_weight,
            load_ae=True,
        )
        if val_stems
        else None
    )
    log.info("Training augmentation rot/flip: %s", "on" if args.augment_rotflip else "off")

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_dem_batch,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kw)
    val_loader = (
        DataLoader(val_subset, shuffle=False, **loader_kw) if val_subset is not None else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model = create_model(args.arch).to(device)
    log.info("Architecture: %s", args.arch)
    log.info("Loss preset: %s (contour_interval=%.3g m)", args.loss_preset, args.contour_interval)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_component_names = ("elev", "slope", "grad", "curv", "ms_elev", "sdf", "contour")
    history: dict[str, list[float | None]] = {
        "train_loss": [],
        "train_elev_loss": [],
        "train_slope_loss": [],
        "val_loss": [],
        "val_elev_loss": [],
        "val_slope_loss": [],
        "epoch_seconds": [],
        "samples_per_second": [],
    }
    for _name in loss_component_names:
        history.setdefault(f"train_{_name}_loss", [])
        history.setdefault(f"val_{_name}_loss", [])
    start_epoch = 0

    if args.resume is not None:
        if not args.resume.is_file():
            log.error("Resume checkpoint not found: %s", args.resume)
            sys.exit(1)
        load_kw: dict = {"map_location": "cpu"}
        try:
            ckpt = torch.load(args.resume, weights_only=False, **load_kw)
        except TypeError:
            ckpt = torch.load(args.resume, **load_kw)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            log.error("Unsupported resume checkpoint format: %s", args.resume)
            sys.exit(1)

        ckpt_args = ckpt.get("args")
        if isinstance(ckpt_args, dict):
            ckpt_arch = ckpt_args.get("arch", ARCH_FILM)
            if ckpt_arch != args.arch:
                log.error(
                    "Resume arch mismatch: checkpoint=%s current=%s. "
                    "Pass --arch %s or use a matching checkpoint.",
                    ckpt_arch,
                    args.arch,
                    ckpt_arch,
                )
                sys.exit(1)
            ckpt_data_root = ckpt.get("data_root")
            if ckpt_data_root is not None and str(ckpt_data_root) != str(args.data_root):
                log.warning(
                    "Resume data_root differs: checkpoint=%s current=%s",
                    ckpt_data_root,
                    args.data_root,
                )

        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not restore AMP scaler state: %s", exc)

        ckpt_history = ckpt.get("history")
        if isinstance(ckpt_history, dict):
            history = ckpt_history
            ref_len = len(history.get("train_loss", []))
            for _name in loss_component_names:
                for prefix in ("train", "val"):
                    key = f"{prefix}_{_name}_loss"
                    if key not in history:
                        history[key] = [None] * ref_len

        if isinstance(ckpt_args, dict):
            ckpt_preset = ckpt_args.get("loss_preset", LOSS_PRESET_BASELINE)
            if ckpt_preset != args.loss_preset:
                log.warning(
                    "Resume loss-preset differs: checkpoint=%s current=%s",
                    ckpt_preset,
                    args.loss_preset,
                )

        start_epoch = int(ckpt.get("epoch", 0))
        if start_epoch >= args.epochs:
            log.error(
                "Resume epoch (%d) is not less than requested --epochs (%d).",
                start_epoch,
                args.epochs,
            )
            sys.exit(1)
        log.info("Resumed from %s at completed epoch %d", args.resume, start_epoch)

    loss_kwargs = dict(
        lambda_slope=args.lambda_slope,
        lambda_grad=args.lambda_grad,
        lambda_curv=args.lambda_curv,
        lambda_ms=args.lambda_ms,
        lambda_sdf=args.lambda_sdf,
        lambda_contour=args.lambda_contour,
        contour_interval=args.contour_interval,
    )

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        running = 0.0
        running_components: dict[str, float] = {name: 0.0 for name in loss_component_names}
        running_component_counts: dict[str, int] = {name: 0 for name in loss_component_names}
        n_batch = 0
        train_bar = tqdm(
            train_loader,
            desc=f"train {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
            leave=True,
        )
        for batch in train_bar:
            x_dem = batch["x_dem"].to(device, non_blocking=True)
            x_ae = batch["x_ae"].to(device, non_blocking=True)
            z_lr = batch["z_lr"].to(device, non_blocking=True)
            z_gt = batch["z_gt"].to(device, non_blocking=True)
            w = batch["w"].to(device, non_blocking=True)
            if args.guidance_dropout > 0:
                drop = torch.rand(x_ae.size(0), 1, 1, 1, device=device) < args.guidance_dropout
                x_ae = x_ae.masked_fill(drop, 0.0)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                z_hat = model(x_dem, x_ae, z_lr)
                loss, components = loss_dem_preset(
                    z_hat, z_gt, w, preset=args.loss_preset, **loss_kwargs
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_value = float(loss.detach())
            running += loss_value
            for name in loss_component_names:
                if name in components:
                    running_components[name] += float(components[name])
                    running_component_counts[name] += 1
            n_batch += 1
            elev_value = float(components["elev"])
            slope_value = float(components["slope"])
            train_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                elev=f"{elev_value:.4f}",
                slope=f"{slope_value:.4f}",
                mean=f"{running / max(n_batch, 1):.4f}",
            )

        epoch_seconds = time.perf_counter() - epoch_start
        samples_per_second = (n_batch * args.batch_size) / max(epoch_seconds, 1e-8)
        train_loss_mean = running / max(n_batch, 1)
        component_means: dict[str, float | None] = {}
        for name in loss_component_names:
            cnt = running_component_counts[name]
            component_means[name] = (running_components[name] / cnt) if cnt > 0 else None
        history["train_loss"].append(train_loss_mean)
        history["train_elev_loss"].append(component_means["elev"])
        history["train_slope_loss"].append(component_means["slope"])
        for name in loss_component_names:
            history[f"train_{name}_loss"].append(component_means[name])
        history["epoch_seconds"].append(epoch_seconds)
        history["samples_per_second"].append(samples_per_second)

        extra = " ".join(
            f"{name}={component_means[name]:.6f}"
            for name in ("grad", "curv", "ms_elev", "sdf", "contour")
            if component_means.get(name) is not None
        )
        log.info(
            "epoch %d train loss mean=%.6f elev=%.6f slope=%.6f %s(%d batches, %.2f samples/s, %.1fs)",
            epoch + 1,
            train_loss_mean,
            component_means["elev"] if component_means["elev"] is not None else float("nan"),
            component_means["slope"] if component_means["slope"] is not None else float("nan"),
            (extra + " ") if extra else "",
            n_batch,
            samples_per_second,
            epoch_seconds,
        )

        if val_loader is not None:
            model.eval()
            vsum = 0.0
            vn = 0
            v_components: dict[str, float] = {name: 0.0 for name in loss_component_names}
            v_component_counts: dict[str, int] = {name: 0 for name in loss_component_names}
            with torch.no_grad():
                val_bar = tqdm(
                    val_loader,
                    desc=f"val {epoch + 1}/{args.epochs}",
                    dynamic_ncols=True,
                    leave=False,
                )
                for batch in val_bar:
                    x_dem = batch["x_dem"].to(device, non_blocking=True)
                    x_ae = batch["x_ae"].to(device, non_blocking=True)
                    z_lr = batch["z_lr"].to(device, non_blocking=True)
                    z_gt = batch["z_gt"].to(device, non_blocking=True)
                    w = batch["w"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        z_hat = model(x_dem, x_ae, z_lr)
                        loss, components = loss_dem_preset(
                            z_hat, z_gt, w, preset=args.loss_preset, **loss_kwargs
                        )
                    loss_value = float(loss)
                    vsum += loss_value
                    for name in loss_component_names:
                        if name in components:
                            v_components[name] += float(components[name])
                            v_component_counts[name] += 1
                    vn += 1
                    val_bar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        elev=f"{float(components['elev']):.4f}",
                        slope=f"{float(components['slope']):.4f}",
                        mean=f"{vsum / max(vn, 1):.4f}",
                    )
            val_loss_mean = vsum / max(vn, 1)
            v_means: dict[str, float | None] = {}
            for name in loss_component_names:
                cnt = v_component_counts[name]
                v_means[name] = (v_components[name] / cnt) if cnt > 0 else None
            history["val_loss"].append(val_loss_mean)
            history["val_elev_loss"].append(v_means["elev"])
            history["val_slope_loss"].append(v_means["slope"])
            for name in loss_component_names:
                history[f"val_{name}_loss"].append(v_means[name])
            extra = " ".join(
                f"{name}={v_means[name]:.6f}"
                for name in ("grad", "curv", "ms_elev", "sdf", "contour")
                if v_means.get(name) is not None
            )
            log.info(
                "epoch %d val loss mean=%.6f elev=%.6f slope=%.6f %s",
                epoch + 1,
                val_loss_mean,
                v_means["elev"] if v_means["elev"] is not None else float("nan"),
                v_means["slope"] if v_means["slope"] is not None else float("nan"),
                extra,
            )
        else:
            history["val_loss"].append(None)
            history["val_elev_loss"].append(None)
            history["val_slope_loss"].append(None)
            for name in loss_component_names:
                history[f"val_{name}_loss"].append(None)

        epoch_out = args.checkpoint_out.with_name(
            f"{args.checkpoint_out.stem}_epoch_{epoch + 1:03d}{args.checkpoint_out.suffix}"
        )
        save_checkpoint(
            epoch_out,
            model,
            optimizer=opt,
            scaler=scaler,
            data_root=args.data_root,
            epoch=epoch + 1,
            args=args,
            history=history,
            train_size=len(train_subset),
            val_size=0 if val_subset is None else len(val_subset),
        )
        log.info("Wrote %s", epoch_out)

    save_checkpoint(
        args.checkpoint_out,
        model,
        optimizer=opt,
        scaler=scaler,
        data_root=args.data_root,
        epoch=args.epochs,
        args=args,
        history=history,
        train_size=len(train_subset),
        val_size=0 if val_subset is None else len(val_subset),
    )
    log.info("Wrote %s", args.checkpoint_out)


if __name__ == "__main__":
    main()
