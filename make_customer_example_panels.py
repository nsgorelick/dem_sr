#!/usr/bin/env python3
"""Generate 8-panel PNGs for selected customer example chips.

Panels:
1. Input DEM (z_lr)
2. FABDEM
3. Model prediction
4. Ground truth (z_gt)
5. Hillshade of input DEM
6. Hillshade of FABDEM
7. Hillshade of model prediction
8. Hillshade of ground truth
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont

from local_patch_dataset import load_patch_stems_manifest


def read_single_band(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        arr = src.read(1, out_dtype="float32", masked=True)
        data = np.ma.getdata(arr).astype(np.float32, copy=False)
        mask = (~np.ma.getmaskarray(arr)) & np.isfinite(data)
        nodata = src.nodata
        if nodata is not None and math.isfinite(float(nodata)):
            mask &= data != np.float32(nodata)
    return data, mask


def read_z_lr(stack_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(stack_path) as src:
        arr = src.read(3, out_dtype="float32", masked=True)
        data = np.ma.getdata(arr).astype(np.float32, copy=False)
        mask = (~np.ma.getmaskarray(arr)) & np.isfinite(data)
        nodata = src.nodata
        if nodata is not None and math.isfinite(float(nodata)):
            mask &= data != np.float32(nodata)
    return data, mask


def read_z_gt(stack_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(stack_path) as src:
        z_gt = src.read(1, out_dtype="float32", masked=True)
        z_gt_mask = src.read(2, out_dtype="float32") > 0.5
        data = np.ma.getdata(z_gt).astype(np.float32, copy=False)
        mask = (~np.ma.getmaskarray(z_gt)) & np.isfinite(data) & z_gt_mask
        nodata = src.nodata
        if nodata is not None and math.isfinite(float(nodata)):
            mask &= data != np.float32(nodata)
    return data, mask


def compute_common_stretch(
    arrays: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    lo_pct: float,
    hi_pct: float,
) -> tuple[float, float]:
    vals = [arr[mask] for arr, mask in zip(arrays, masks) if mask.any()]
    if not vals:
        return 0.0, 1.0
    joined = np.concatenate(vals)
    lo = float(np.percentile(joined, lo_pct))
    hi = float(np.percentile(joined, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(joined.min())
        hi = float(joined.max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def to_grayscale(arr: np.ndarray, mask: np.ndarray, lo: float, hi: float) -> Image.Image:
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    img = np.round(scaled * 255.0).astype(np.uint8)
    img[~mask] = 0
    return Image.fromarray(img, mode="L")


def hillshade_array(
    arr: np.ndarray,
    mask: np.ndarray,
    *,
    pixel_size_m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    # Fill invalids with nearest-safe fallback to avoid NaN gradients.
    filled = arr.astype(np.float32, copy=True)
    if mask.any():
        fill_value = float(np.median(filled[mask]))
    else:
        fill_value = 0.0
    filled[~mask] = fill_value

    dz_dy, dz_dx = np.gradient(filled, pixel_size_m, pixel_size_m)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)

    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(45.0)
    shaded = (
        np.sin(altitude) * np.sin(slope)
        + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    )
    shaded = np.clip(shaded, 0.0, 1.0)
    return shaded.astype(np.float32, copy=False), mask


def add_labeled_panel(
    canvas: Image.Image,
    panel: Image.Image,
    *,
    x: int,
    y: int,
    label: str,
    font: ImageFont.ImageFont,
    panel_size: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), label, fill="black", font=font)
    panel_y = y + 16
    panel_rgb = panel.resize((panel_size, panel_size), resample=Image.Resampling.NEAREST).convert("RGB")
    canvas.paste(panel_rgb, (x, panel_y))


def make_panel_png(
    stem: str,
    *,
    stack_dir: Path,
    fabdem_dir: Path,
    model_dir: Path,
    output_dir: Path,
    lo_pct: float,
    hi_pct: float,
    panel_size: int,
) -> Path:
    z_lr, m_lr = read_z_lr(stack_dir / f"{stem}.tif")
    z_gt, m_gt = read_z_gt(stack_dir / f"{stem}.tif")
    fab, m_fab = read_single_band(fabdem_dir / f"{stem}.tif")
    model, m_model = read_single_band(model_dir / f"{stem}.tif")

    elev_lo, elev_hi = compute_common_stretch(
        [z_lr, fab, model, z_gt],
        [m_lr, m_fab, m_model, m_gt],
        lo_pct=lo_pct,
        hi_pct=hi_pct,
    )
    elev_imgs = [
        ("Input DEM", to_grayscale(z_lr, m_lr, elev_lo, elev_hi)),
        ("FABDEM", to_grayscale(fab, m_fab, elev_lo, elev_hi)),
        ("Model", to_grayscale(model, m_model, elev_lo, elev_hi)),
        ("Ground truth", to_grayscale(z_gt, m_gt, elev_lo, elev_hi)),
    ]
    hs_lr, hs_m_lr = hillshade_array(z_lr, m_lr)
    hs_fab, hs_m_fab = hillshade_array(fab, m_fab)
    hs_model, hs_m_model = hillshade_array(model, m_model)
    hs_gt, hs_m_gt = hillshade_array(z_gt, m_gt)
    hs_lo, hs_hi = compute_common_stretch(
        [hs_lr, hs_fab, hs_model, hs_gt],
        [hs_m_lr, hs_m_fab, hs_m_model, hs_m_gt],
        lo_pct=lo_pct,
        hi_pct=hi_pct,
    )
    hill_imgs = [
        ("Input hillshade", to_grayscale(hs_lr, hs_m_lr, hs_lo, hs_hi)),
        ("FABDEM hillshade", to_grayscale(hs_fab, hs_m_fab, hs_lo, hs_hi)),
        ("Model hillshade", to_grayscale(hs_model, hs_m_model, hs_lo, hs_hi)),
        ("GT hillshade", to_grayscale(hs_gt, hs_m_gt, hs_lo, hs_hi)),
    ]

    margin = 16
    gap = 16
    title_h = 28
    label_h = 16
    width = margin * 2 + panel_size * 4 + gap * 3
    height = margin * 2 + title_h + (label_h + panel_size) * 2 + gap
    canvas = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin), stem, fill="black", font=font)

    row1_y = margin + title_h
    row2_y = row1_y + label_h + panel_size + gap
    for idx, (label, img) in enumerate(elev_imgs):
        x = margin + idx * (panel_size + gap)
        add_labeled_panel(canvas, img, x=x, y=row1_y, label=label, font=font, panel_size=panel_size)
    for idx, (label, img) in enumerate(hill_imgs):
        x = margin + idx * (panel_size + gap)
        add_labeled_panel(canvas, img, x=x, y=row2_y, label=label, font=font, panel_size=panel_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.png"
    canvas.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("customer_example_chips_manifest.txt"),
        help="One selected patch stem per line",
    )
    parser.add_argument("--stack-dir", type=Path, default=Path("/data/training/stack"))
    parser.add_argument("--fabdem-dir", type=Path, default=Path("/data/comparison/fabdem"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("customer_example_predictions/tifs"),
        help="Directory containing model prediction TIFFs for the same stems",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("customer_example_predictions/panels"))
    parser.add_argument("--stretch-low-pct", type=float, default=2.0)
    parser.add_argument("--stretch-high-pct", type=float, default=98.0)
    parser.add_argument("--panel-size", type=int, default=384)
    args = parser.parse_args()

    stems = load_patch_stems_manifest(args.manifest)
    missing = []
    written = []
    for stem in stems:
        needed = [
            args.stack_dir / f"{stem}.tif",
            args.fabdem_dir / f"{stem}.tif",
            args.model_dir / f"{stem}.tif",
        ]
        absent = [str(p) for p in needed if not p.is_file()]
        if absent:
            missing.append({"stem": stem, "missing": absent})
            continue
        out_path = make_panel_png(
            stem,
            stack_dir=args.stack_dir,
            fabdem_dir=args.fabdem_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            lo_pct=args.stretch_low_pct,
            hi_pct=args.stretch_high_pct,
            panel_size=args.panel_size,
        )
        written.append(str(out_path))

    print(f"Wrote {len(written)} panel PNG(s) to {args.output_dir}")
    if missing:
        print("Skipped stems with missing inputs:")
        for item in missing:
            print(f"  {item['stem']}")
            for path in item["missing"]:
                print(f"    {path}")


if __name__ == "__main__":
    main()
