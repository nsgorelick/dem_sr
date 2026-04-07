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

from dem_film_unet import DemFilmUNet, loss_dem
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
    model: DemFilmUNet,
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

    model = DemFilmUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
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

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        running = 0.0
        running_elev = 0.0
        running_slope = 0.0
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
                loss, le, ls = loss_dem(z_hat, z_gt, w, lambda_slope=args.lambda_slope)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_value = float(loss.detach())
            elev_value = float(le)
            slope_value = float(ls)
            running += loss_value
            running_elev += elev_value
            running_slope += slope_value
            n_batch += 1
            train_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                elev=f"{elev_value:.4f}",
                slope=f"{slope_value:.4f}",
                mean=f"{running / max(n_batch, 1):.4f}",
            )

        epoch_seconds = time.perf_counter() - epoch_start
        samples_per_second = (n_batch * args.batch_size) / max(epoch_seconds, 1e-8)
        train_loss_mean = running / max(n_batch, 1)
        train_elev_mean = running_elev / max(n_batch, 1)
        train_slope_mean = running_slope / max(n_batch, 1)
        history["train_loss"].append(train_loss_mean)
        history["train_elev_loss"].append(train_elev_mean)
        history["train_slope_loss"].append(train_slope_mean)
        history["epoch_seconds"].append(epoch_seconds)
        history["samples_per_second"].append(samples_per_second)

        log.info(
            "epoch %d train loss mean=%.6f elev=%.6f slope=%.6f (%d batches, %.2f samples/s, %.1fs)",
            epoch + 1,
            train_loss_mean,
            train_elev_mean,
            train_slope_mean,
            n_batch,
            samples_per_second,
            epoch_seconds,
        )

        if val_loader is not None:
            model.eval()
            vsum = 0.0
            velev = 0.0
            vslope = 0.0
            vn = 0
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
                        loss, le, ls = loss_dem(z_hat, z_gt, w, lambda_slope=args.lambda_slope)
                    loss_value = float(loss)
                    elev_value = float(le)
                    slope_value = float(ls)
                    vsum += loss_value
                    velev += elev_value
                    vslope += slope_value
                    vn += 1
                    val_bar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        elev=f"{elev_value:.4f}",
                        slope=f"{slope_value:.4f}",
                        mean=f"{vsum / max(vn, 1):.4f}",
                    )
            val_loss_mean = vsum / max(vn, 1)
            val_elev_mean = velev / max(vn, 1)
            val_slope_mean = vslope / max(vn, 1)
            history["val_loss"].append(val_loss_mean)
            history["val_elev_loss"].append(val_elev_mean)
            history["val_slope_loss"].append(val_slope_mean)
            log.info(
                "epoch %d val loss mean=%.6f elev=%.6f slope=%.6f",
                epoch + 1,
                val_loss_mean,
                val_elev_mean,
                val_slope_mean,
            )
        else:
            history["val_loss"].append(None)
            history["val_elev_loss"].append(None)
            history["val_slope_loss"].append(None)

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
