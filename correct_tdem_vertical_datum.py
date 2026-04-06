#!/usr/bin/env python3
"""Convert TanDEM-X EDEM patch TIFFs from WGS84 ellipsoidal to EGM2008 heights.

This is intentionally a standalone one-file utility so vertical-datum correction
can be tested on a single exported patch before touching any batch export path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer, datadir, network
from pyproj.transformer import TransformerGroup


def _configure_proj_data(proj_data: Path | None) -> list[str]:
    """Configure local PROJ data directories without enabling network fetches."""
    if proj_data is not None:
        if not proj_data.is_dir():
            raise FileNotFoundError(f"PROJ data directory not found: {proj_data}")
        datadir.append_data_dir(str(proj_data))
    network.set_network_enabled(False)
    return datadir.get_data_dir().split(os.pathsep)


def _missing_grids(group: TransformerGroup) -> list[str]:
    """Extract short names for missing PROJ grids when available."""
    names: list[str] = []
    for op in group.unavailable_operations:
        for grid in getattr(op, "grids", ()):
            short_name = getattr(grid, "short_name", None)
            if short_name and short_name not in names:
                names.append(short_name)
    return names


def build_transformers(
    src_crs,
    *,
    proj_data: Path | None,
) -> tuple[Transformer, Transformer, dict[str, object]]:
    """Return horizontal and vertical transformers, failing if the local geoid grid is unavailable."""
    data_dirs = _configure_proj_data(proj_data)
    to_lonlat = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    group = TransformerGroup(
        CRS.from_epsg(4979),
        CRS.from_user_input("EPSG:4326+3855"),
        always_xy=True,
    )
    if not group.best_available or not group.transformers:
        missing = _missing_grids(group)
        raise RuntimeError(
            "EGM2008 vertical transform is not available from local PROJ data. "
            f"Checked data directories: {data_dirs}. "
            + (f"Missing grids: {missing}. " if missing else "")
            + "Download the required grid locally and rerun with --proj-data if needed."
        )
    diagnostics = {
        "proj_data_dirs": data_dirs,
        "transformer_description": group.transformers[0].description,
        "missing_grids": _missing_grids(group),
    }
    return to_lonlat, group.transformers[0], diagnostics


def convert_one_tif(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
    proj_data: Path | None,
) -> dict[str, object]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input TIFF not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")

    with rasterio.open(input_path) as src:
        if src.count != 1:
            raise ValueError(f"Expected single-band TIFF, got {src.count} bands: {input_path}")
        to_lonlat, ellipsoid_to_egm2008, diagnostics = build_transformers(
            src.crs,
            proj_data=proj_data,
        )
        data = src.read(1, masked=True)
        rows, cols = np.indices(data.shape)
        xs, ys = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel(), offset="center")
        xs = np.asarray(xs, dtype=np.float64).reshape(data.shape)
        ys = np.asarray(ys, dtype=np.float64).reshape(data.shape)
        lon, lat = to_lonlat.transform(xs, ys)

        raw = np.ma.getdata(data).astype(np.float64, copy=True)
        mask = np.ma.getmaskarray(data)
        valid = (~mask) & np.isfinite(raw)
        corrected = raw.copy()
        if valid.any():
            _, _, z_egm = ellipsoid_to_egm2008.transform(
                lon[valid],
                lat[valid],
                raw[valid],
            )
            corrected[valid] = np.asarray(z_egm, dtype=np.float64)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            compress="ZSTD",
            ZSTD_LEVEL=1,
            predictor=3,
            tiled=True,
            bigtiff="IF_SAFER",
        )
        tags = src.tags()
        band_tags = src.tags(1)
        desc = src.descriptions[0]
        nodata = src.nodata

    corrected_f32 = corrected.astype(np.float32, copy=False)
    if nodata is not None:
        corrected_f32[mask] = np.float32(nodata)

    try:
        with rasterio.open(part_path, "w", **profile) as dst:
            dst.write(corrected_f32[np.newaxis, ...])
            dst.update_tags(**tags)
            dst.update_tags(
                vertical_datum_source="WGS84 ellipsoidal",
                vertical_datum_target="EGM2008 orthometric",
            )
            if desc:
                dst.set_band_description(1, desc)
            if band_tags:
                dst.update_tags(1, **band_tags)
        os.replace(part_path, output_path)
    except Exception:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    stats: dict[str, object] = {
        "input": str(input_path),
        "output": str(output_path),
        "pixels_corrected": int(valid.sum()),
    }
    stats.update(diagnostics)
    if valid.any():
        delta = corrected[valid] - raw[valid]
        stats.update(
            {
                "delta_min_m": float(delta.min()),
                "delta_max_m": float(delta.max()),
                "delta_mean_m": float(delta.mean()),
                "input_range_m": [float(raw[valid].min()), float(raw[valid].max())],
                "output_range_m": [float(corrected[valid].min()), float(corrected[valid].max())],
            }
        )
    return stats


def check_transform_only(input_path: Path, *, proj_data: Path | None) -> dict[str, object]:
    """Preflight local PROJ availability without transforming any TIFFs."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input TIFF not found: {input_path}")
    with rasterio.open(input_path) as src:
        _, _, diagnostics = build_transformers(src.crs, proj_data=proj_data)
    return {
        "input": str(input_path),
        "status": "ok",
        **diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tif", type=Path, help="Input tdem_edem TIFF")
    parser.add_argument("output_tif", type=Path, nargs="?", help="Output corrected TIFF")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    parser.add_argument(
        "--proj-data",
        type=Path,
        default=None,
        help="Optional local PROJ data directory containing EGM2008 grids",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify that the local EGM2008 transform is available; do not write output",
    )
    args = parser.parse_args()

    try:
        if args.check_only:
            stats = check_transform_only(args.input_tif, proj_data=args.proj_data)
        else:
            if args.output_tif is None:
                raise ValueError("output_tif is required unless --check-only is used")
            stats = convert_one_tif(
                args.input_tif,
                args.output_tif,
                overwrite=args.overwrite,
                proj_data=args.proj_data,
            )
        print(json.dumps(stats, indent=2))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
