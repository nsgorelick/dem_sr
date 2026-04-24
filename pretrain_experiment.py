#!/usr/bin/env python3
"""Self-supervised masked-reconstruction pretraining for DEM+AE encoders."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from core.pretraining import extract_encoder_state_dict
from core.run_config import load_run_config, resolve_description, section_defaults
from dem_film_unet import ARCH_FILM
from models.wrappers.factory import create_experiment_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pretrain_experiment")


def _as_path(value: str | Path | None) -> Path | None:
    if value is None or isinstance(value, Path):
        return value
    return Path(value)


def _apply_rotflip_augmentation(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


class MaskedEncoderPretrainer(nn.Module):
    """Joint DEM+AE encoder pretrainer with masked reconstruction heads."""

    def __init__(self, arch: str = ARCH_FILM) -> None:
        super().__init__()
        self.backbone = create_experiment_model(arch)
        self.dem_head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 5, kernel_size=1),
        )
        self.ae_head = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=1),
        )

    def forward(self, x_dem: torch.Tensor, x_ae: torch.Tensor) -> dict[str, torch.Tensor]:
        dem_feat = self.backbone.dem_b0(x_dem)
        ae_feat = self.backbone.ae_b0(x_ae)
        recon_dem = self.dem_head(dem_feat)
        recon_ae = self.ae_head(ae_feat)
        return {"recon_dem": recon_dem, "recon_ae": recon_ae}

    def export_encoder_state(self) -> dict[str, torch.Tensor]:
        return extract_encoder_state_dict(self.backbone)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="JSON run config with pretrain section")
    parser.add_argument("--description", default=None, help="Optional run description")
    parser.add_argument("--data-root", default="/data/training")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--arch", default=ARCH_FILM)
    parser.add_argument("--mask-ratio-dem", type=float, default=0.4)
    parser.add_argument("--mask-ratio-ae", type=float, default=0.4)
    parser.add_argument("--augment-rotflip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-out", type=Path, default=Path("runs/pretrain/encoder_pretrain.pt"))
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def _random_mask_like(x: torch.Tensor, ratio: float) -> torch.Tensor:
    keep = 1.0 - float(max(0.0, min(1.0, ratio)))
    rand = torch.rand((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
    return (rand < keep).float()


def main() -> None:
    from local_patch_dataset import (
        LocalDemPatchDataset,
        collate_dem_batch,
        list_patch_stems,
        load_patch_stems_manifest,
    )

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    run_config: dict[str, object] = {}
    if pre_args.config is not None:
        run_config = load_run_config(pre_args.config)

    parser = build_parser()
    if run_config:
        parser.set_defaults(**section_defaults(run_config, "pretrain"))
    args = parser.parse_args()
    args.config = _as_path(args.config)
    args.manifest = _as_path(args.manifest)
    args.checkpoint_out = _as_path(args.checkpoint_out)
    args.output_json = _as_path(args.output_json)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.manifest is not None:
        stems = load_patch_stems_manifest(args.manifest)
        log.info("Loaded %d stems from %s", len(stems), args.manifest)
    else:
        stems = list_patch_stems(args.data_root)
        log.info("Listed %d stems under %s", len(stems), args.data_root)
    if args.max_patches is not None:
        stems = stems[: int(args.max_patches)]
    if not stems:
        raise SystemExit("No patches found for pretraining.")

    ds = LocalDemPatchDataset(
        args.data_root,
        patch_stems=stems,
        load_ae=True,
        transform=_apply_rotflip_augmentation if args.augment_rotflip else None,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_dem_batch,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    model = MaskedEncoderPretrainer(arch=str(args.arch)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: dict[str, list[float]] = {"loss": [], "dem_loss": [], "ae_loss": [], "epoch_seconds": []}
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        running = 0.0
        running_dem = 0.0
        running_ae = 0.0
        n_batch = 0
        bar = tqdm(loader, desc=f"pretrain {epoch + 1}/{args.epochs}", dynamic_ncols=True, leave=False)
        for batch in bar:
            x_dem = batch["x_dem"].to(device, non_blocking=True)
            x_ae = batch["x_ae"].to(device, non_blocking=True)
            dem_mask = _random_mask_like(x_dem, args.mask_ratio_dem)
            ae_mask = _random_mask_like(x_ae, args.mask_ratio_ae)
            x_dem_masked = x_dem * dem_mask
            x_ae_masked = x_ae * ae_mask

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(x_dem_masked, x_ae_masked)
                target_dem_mask = (1.0 - dem_mask).expand_as(x_dem)
                target_ae_mask = (1.0 - ae_mask).expand_as(x_ae)
                dem_loss = (F.smooth_l1_loss(out["recon_dem"], x_dem, reduction="none") * target_dem_mask).sum() / (
                    target_dem_mask.sum() + 1e-8
                )
                ae_loss = (F.smooth_l1_loss(out["recon_ae"], x_ae, reduction="none") * target_ae_mask).sum() / (
                    target_ae_mask.sum() + 1e-8
                )
                loss = dem_loss + ae_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach())
            running_dem += float(dem_loss.detach())
            running_ae += float(ae_loss.detach())
            n_batch += 1
            bar.set_postfix(loss=f"{running / max(n_batch, 1):.4f}")

        elapsed = time.perf_counter() - t0
        mean = running / max(n_batch, 1)
        mean_dem = running_dem / max(n_batch, 1)
        mean_ae = running_ae / max(n_batch, 1)
        history["loss"].append(mean)
        history["dem_loss"].append(mean_dem)
        history["ae_loss"].append(mean_ae)
        history["epoch_seconds"].append(float(elapsed))
        log.info(
            "epoch %d/%d ssl_loss=%.6f dem=%.6f ae=%.6f (%.1fs)",
            epoch + 1,
            args.epochs,
            mean,
            mean_dem,
            mean_ae,
            elapsed,
        )

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "ssl_pretrain",
        "epoch": args.epochs,
        "args": vars(args),
        "data_root": args.data_root,
        "history": history,
        "model": model.state_dict(),
        "encoder_state": model.export_encoder_state(),
        "train_size": len(ds),
    }
    torch.save(payload, args.checkpoint_out)
    log.info("Wrote %s", args.checkpoint_out)

    if args.output_json is not None:
        report = {
            "kind": "pretrain",
            "checkpoint_out": str(args.checkpoint_out),
            "data_root": args.data_root,
            "train_size": len(ds),
            "history": history,
            "config": vars(args) | {"description": resolve_description(run_config, args.config or Path("pretrain"), args.description)},
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()

