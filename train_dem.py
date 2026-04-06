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
from local_patch_dataset import LocalDemPatchDataset, collate_dem_batch, load_patch_stems_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_dem")


def save_checkpoint(
    path: Path, model: DemFilmUNet, *, data_root: str, epoch: int, args: argparse.Namespace
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "data_root": data_root,
            "epoch": epoch,
            "args": vars(args),
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

    ds = LocalDemPatchDataset(
        args.data_root,
        patch_stems=stems,
        use_precomputed_weight=args.precomputed_weight,
        max_patches=args.max_patches,
    )
    n_total = len(ds)
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
        train_indices = [i for i in range(n_total) if i not in val_set]
        val_indices = [i for i in range(n_total) if i in val_set]
        train_subset = Subset(ds, train_indices)
        val_subset = Subset(ds, val_indices)
        log.info("Random split: train=%d val=%d", len(train_subset), len(val_subset))
    else:
        train_subset = ds
        val_subset = None

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

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        running = 0.0
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
            n_batch += 1
            train_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                elev=f"{elev_value:.4f}",
                slope=f"{slope_value:.4f}",
                mean=f"{running / max(n_batch, 1):.4f}",
            )

        epoch_seconds = time.perf_counter() - epoch_start
        samples_per_second = (n_batch * args.batch_size) / max(epoch_seconds, 1e-8)

        log.info(
            "epoch %d train loss mean=%.6f (%d batches, %.2f samples/s, %.1fs)",
            epoch + 1,
            running / max(n_batch, 1),
            n_batch,
            samples_per_second,
            epoch_seconds,
        )

        if val_loader is not None:
            model.eval()
            vsum = 0.0
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
                        loss, _, _ = loss_dem(z_hat, z_gt, w, lambda_slope=args.lambda_slope)
                    loss_value = float(loss)
                    vsum += loss_value
                    vn += 1
                    val_bar.set_postfix(loss=f"{loss_value:.4f}", mean=f"{vsum / max(vn, 1):.4f}")
            log.info("epoch %d val loss mean=%.6f", epoch + 1, vsum / max(vn, 1))

        epoch_out = args.checkpoint_out.with_name(
            f"{args.checkpoint_out.stem}_epoch_{epoch + 1:03d}{args.checkpoint_out.suffix}"
        )
        save_checkpoint(epoch_out, model, data_root=args.data_root, epoch=epoch + 1, args=args)
        log.info("Wrote %s", epoch_out)

    save_checkpoint(args.checkpoint_out, model, data_root=args.data_root, epoch=args.epochs, args=args)
    log.info("Wrote %s", args.checkpoint_out)


if __name__ == "__main__":
    main()
