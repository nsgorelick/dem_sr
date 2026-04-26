#!/usr/bin/env python3
"""Export 512x512 DEM/AE patches to a dedicated training subdirectory."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from export_patches_gcs import (
    EE_HTTP_POOL_MAXSIZE,
    configure_ee_http_pool,
    configure_export_layout,
    run_export,
)


DEFAULT_EXPORT_DIR = Path("./data/training/patches_512")
DEFAULT_PATCH_SIZE = 512


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest with one patch id per line (x_y_zone_country_year)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help="Parent output directory (contains stack/ and ae/)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help="Patch size in pixels at 10m resolution",
    )
    parser.add_argument("--pool-workers", type=int, default=None, help="Initial worker pool size")
    parser.add_argument(
        "--http-pool-size",
        type=int,
        default=EE_HTTP_POOL_MAXSIZE,
        help="EE requests-session HTTP pool size (default: %(default)s)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_ee_http_pool(args.http_pool_size)
    configure_export_layout(export_dir=args.export_dir, patch_size_pixels=args.patch_size)
    run_export(manifest=args.manifest, pool_workers=args.pool_workers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)
    main()

