"""Export comparison DTM patches from Earth Engine on the training 10 m grid.

Writes one single-band GeoTIFF per product per patch under ``comparison/<product>/``.
Both source DTMs are 30 m products. They are resampled with bilinear interpolation
and reprojected to the same 128x128, 10 m UTM patch grid used by the training data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import ee
import rasterio
from rasterio.io import MemoryFile

from adaptive_export_runner import AdaptiveThreadExportRunner
from export_progress import ExportProgressLine
from local_patch_dataset import load_patch_stems_manifest

COMPARE_EXPORT_DIR = Path("comparison")
log = logging.getLogger("export_comparison_dtms")

EE_INIT_KWARGS: dict[str, str] = {
    "project": "ngorelick",
    "opt_url": "https://earthengine-highvolume.googleapis.com",
}

DEFAULT_COLLECTION_IDS = (
    "users/ngorelick/DTM/tmp/sample_us_100k",
    "users/ngorelick/DTM/tmp/sample_100k",
)

FABDEM_ASSET = "projects/sat-io/open-datasets/FABDEM"
TDEM_EDEM_ASSET = "projects/earthengine-legacy/assets/users/ngorelick/DTM/TDEM_EDEM"
PRODUCT_ASSETS = {
    "fabdem": FABDEM_ASSET,
    "tdem_edem": TDEM_EDEM_ASSET,
}

DEFAULT_EXPORT_POOL_WORKERS = 50
_PROGRESS_RATE_WINDOW_SEC = 60.0
_TIF_EXT = ".tif"
URL_OPEN_TIMEOUT_SEC = 120.0


def patch_id_from_properties(props: dict[str, Any]) -> str:
    """Filename stem for exported patch rasters."""
    return f"{int(props['x'])}_{int(props['y'])}_{props['zone']}_{props['country']}_{props['year']}"


def output_path_for(product: str, patch_id: str) -> Path:
    """Return the final output path for one product/patch export."""
    return COMPARE_EXPORT_DIR / product / f"{patch_id}.tif"


def completed_export_keys_on_disk(products: Sequence[str]) -> set[tuple[str, str]]:
    """Return ``(product, patch_id)`` pairs that already have final outputs on disk."""
    done: set[tuple[str, str]] = set()
    for product in products:
        product_dir = COMPARE_EXPORT_DIR / product
        if not product_dir.is_dir():
            continue
        for name in os.listdir(product_dir):
            if ".part" in name or not name.endswith(_TIF_EXT):
                continue
            done.add((product, name[: -len(_TIF_EXT)]))
    return done


def write_geotiff_zstd(path: Path | str, tif_bytes: bytes, predictor: int) -> None:
    """Write a GeoTIFF with ZSTD compression and atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_suffix(path.suffix + ".part")
    _rio_env = logging.getLogger("rasterio._env")
    _saved_rio = _rio_env.level
    _rio_env.setLevel(logging.ERROR)
    try:
        try:
            with MemoryFile(tif_bytes) as mem:
                with mem.open() as src:
                    data = src.read()
                    profile = src.profile.copy()
                    profile.pop("extra_samples", None)
                    profile.update(
                        driver="GTiff",
                        compress="ZSTD",
                        ZSTD_LEVEL=1,
                        predictor=predictor,
                        photometric="MINISBLACK",
                    )
                    with rasterio.open(part_path, "w", **profile) as dst:
                        dst.write(data)
                        dst.update_tags(**src.tags())
                        for band_idx in range(1, src.count + 1):
                            desc = src.descriptions[band_idx - 1]
                            if desc:
                                dst.set_band_description(band_idx, desc)
                            band_tags = src.tags(band_idx)
                            if band_tags:
                                dst.update_tags(band_idx, **band_tags)
            os.replace(part_path, path)
        except Exception:
            try:
                part_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    finally:
        _rio_env.setLevel(_saved_rio)


def _utm_crs_code(signed_zone: int) -> str:
    z = int(signed_zone)
    if z < 0:
        return f"EPSG:{32700 + abs(z)}"
    return f"EPSG:{32600 + abs(z)}"


def make_utm_proj_signed(signed_zone: int) -> ee.Projection:
    """Return the UTM projection for a signed zone id."""
    z = int(signed_zone)
    if z < 0:
        return ee.Projection(f"EPSG:{32700 + abs(z)}")
    return ee.Projection(f"EPSG:{32600 + abs(z)}")


def grid_128x128_snap10m(x: int, y: int, scale_m: int = 10, size: int = 128, coarse_m: int = 1280) -> dict:
    """Pixel grid for Earth Engine ``computePixels`` on the training patch grid.

    ``x`` and ``y`` identify the southwest coarse-grid corner of the patch in UTM
    patch coordinates. The exported raster should therefore span exactly one
    ``coarse_m`` cell, starting at ``(x * coarse_m, y * coarse_m)``.
    """
    left = float(x) * coarse_m
    top = float(y + 1) * coarse_m
    return {
        "dimensions": {"width": size, "height": size},
        "affineTransform": {
            "scaleX": scale_m,
            "shearX": 0,
            "translateX": left,
            "shearY": 0,
            "scaleY": -scale_m,
            "translateY": top,
        },
    }


def fetch_patch_items(collection_ids: Sequence[str]) -> list[dict]:
    """Download minimal GeoJSON patch rows from Earth Engine feature collections."""
    selectors = ["system:index", "x", "y", "zone", "year", "country"]
    items: list[dict] = []
    for collection_path in collection_ids:
        log.info("Preparing patch catalog request for %s", collection_path)
        try:
            fc = ee.FeatureCollection(collection_path)
            log.info("Requesting download URL for %s", collection_path)
            url = fc.getDownloadURL(filetype="geojson", selectors=selectors)
            log.info("Opening catalog URL for %s", collection_path)
            with urllib.request.urlopen(url, timeout=URL_OPEN_TIMEOUT_SEC) as response:
                raw = response.read()
            log.info("Downloaded patch catalog bytes for %s", collection_path)
        except Exception:
            log.exception("Failed while fetching patch catalog for %s", collection_path)
            raise
        payload = json.loads(raw.decode("utf-8"))
        for feat in payload["features"]:
            feat["properties"]["path"] = f"{collection_path}/{feat['id']}"
            items.append(feat)
        log.info("Loaded %d patch rows so far", len(items))
    return items


def filter_items_by_manifest(items: list[dict], manifest_path: Path | None) -> list[dict]:
    """Keep only patch rows present in the optional manifest file."""
    if manifest_path is None:
        return items
    keep = frozenset(load_patch_stems_manifest(manifest_path))
    return [item for item in items if patch_id_from_properties(item["properties"]) in keep]


def make_product_image(product: str, zone: int) -> ee.Image:
    """Return one comparison DTM image on the training 10 m UTM grid."""
    proj10 = make_utm_proj_signed(zone).atScale(10)
    asset_id = PRODUCT_ASSETS[product]
    collection = ee.ImageCollection(asset_id).select([0], ["elevation"])
    collection = collection.cast({"elevation": "float"}, ["elevation"])
    first = ee.Image(collection.first()).select([0], [product])
    base = collection.mosaic().select([0], [product]).reproject(first.projection())
    return base.resample("bilinear").reproject(proj10).float()


def make_export_jobs(items: Sequence[dict], products: Sequence[str]) -> list[dict]:
    """Flatten patch rows into one export job per requested product."""
    jobs: list[dict] = []
    for item in items:
        patch_id = patch_id_from_properties(item["properties"])
        for product in products:
            jobs.append(
                {
                    "item": item,
                    "product": product,
                    "patch_id": patch_id,
                }
            )
    return jobs


def filter_pending_jobs(jobs: Sequence[dict], completed_keys: set[tuple[str, str]]) -> list[dict]:
    """Keep only product/patch jobs that are not already on disk."""
    return [job for job in jobs if (job["product"], job["patch_id"]) not in completed_keys]


def export_one_product_patch(job: dict) -> dict:
    """Export one product for one patch using the training patch grid."""
    item = job["item"]
    product = job["product"]
    patch_id = job["patch_id"]
    props = item["properties"]
    zone = int(props["zone"])
    ee_path = props.get("path", "")

    try:
        grid = grid_128x128_snap10m(int(props["x"]), int(props["y"]))
        grid["crsCode"] = _utm_crs_code(zone)
        payload = {
            "expression": make_product_image(product, zone),
            "fileFormat": "GEO_TIFF",
            "bandIds": [product],
            "grid": grid,
        }
        log.info(
            "Calling computePixels for product=%s patch=%s zone=%s feature=%s",
            product,
            patch_id,
            zone,
            ee_path or "<unknown>",
        )
        tif_bytes = ee.data.computePixels(payload)
        out_path = output_path_for(product, patch_id)
        log.info("Writing %s", out_path)
        write_geotiff_zstd(out_path, tif_bytes, predictor=3)
    except ee.EEException as exc:
        log.exception(
            "Earth Engine export failed for product=%s patch=%s zone=%s feature=%s",
            product,
            patch_id,
            zone,
            ee_path or "<unknown>",
        )
        prefix = f"[patch {patch_id}] [product {product}]"
        if ee_path:
            prefix += f" [ee_feature {ee_path}]"
        raise ee.EEException(f"{prefix} {exc}") from exc
    except Exception:
        log.exception("Unexpected export failure for product=%s patch=%s", product, patch_id)
        raise

    return {
        "id": f"{product}/{patch_id}",
        "path": str(out_path.resolve()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest of patch stems to export (recommended for holdout-only export)",
    )
    p.add_argument(
        "--collection-id",
        action="append",
        dest="collection_ids",
        default=None,
        help="Earth Engine patch feature collection ID; may be passed multiple times",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=COMPARE_EXPORT_DIR,
        help="Root directory for output comparison GeoTIFFs",
    )
    p.add_argument(
        "--product",
        action="append",
        choices=tuple(PRODUCT_ASSETS),
        default=None,
        help="Product to export; may be passed multiple times (default: all)",
    )
    p.add_argument(
        "--pool-workers",
        type=int,
        default=DEFAULT_EXPORT_POOL_WORKERS,
        help="Initial export worker count for adaptive concurrency",
    )
    p.add_argument(
        "--max-tries",
        type=int,
        default=5,
        help="Retries per product/patch export",
    )
    return p


def run_export(
    manifest: Path | None,
    collection_ids: Sequence[str],
    products: Sequence[str],
    pool_workers: int,
    max_tries: int,
) -> None:
    log.info("Fetching patch items from %d collection(s)", len(collection_ids))
    items = fetch_patch_items(collection_ids)
    log.info("Fetched %d patch items before manifest filtering", len(items))
    items = filter_items_by_manifest(items, manifest)
    log.info("Retained %d patch items after manifest filtering", len(items))
    jobs = make_export_jobs(items, products)
    completed = completed_export_keys_on_disk(products)
    n_all = len(jobs)
    jobs = filter_pending_jobs(jobs, completed)
    skipped = n_all - len(jobs)
    total = len(jobs)
    print(
        f"catalog {n_all} product-patch exports, {skipped} already on disk, {total} to export",
        flush=True,
    )
    if not jobs:
        print("done (nothing to run).", flush=True)
        return

    runner = AdaptiveThreadExportRunner(
        min_concurrent=1,
        max_concurrent=pool_workers * 2,
        initial_concurrent=pool_workers,
        quiet_before_scale_up_sec=10.0,
        scale_down_step=1,
        scale_up_step=1,
        max_tries=max_tries,
        retry_base_delay_sec=1.0,
        retry_backoff=2.0,
    )
    progress = ExportProgressLine(rate_window_sec=_PROGRESS_RATE_WINDOW_SEC)

    def on_export_error(exc: Exception) -> None:
        print(f"\n[export skipped] {exc}", file=sys.stderr, flush=True)

    for _ in runner.run_unordered(
        export_one_product_patch,
        jobs,
        on_each_done=progress.emit,
        rate_window_sec=_PROGRESS_RATE_WINDOW_SEC,
        continue_on_errors=False,
        on_task_error=on_export_error,
    ):
        pass
    progress.end_line()
    print("done.", flush=True)


def main() -> None:
    global COMPARE_EXPORT_DIR

    args = build_arg_parser().parse_args()
    COMPARE_EXPORT_DIR = args.output_dir
    COMPARE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Initializing Earth Engine with %s", EE_INIT_KWARGS)
    ee.Initialize(**EE_INIT_KWARGS)
    log.info("Earth Engine initialized")

    collection_ids = tuple(args.collection_ids or DEFAULT_COLLECTION_IDS)
    products = tuple(args.product or PRODUCT_ASSETS.keys())
    for product in products:
        (COMPARE_EXPORT_DIR / product).mkdir(parents=True, exist_ok=True)
    run_export(
        args.manifest,
        collection_ids,
        products,
        pool_workers=args.pool_workers,
        max_tries=args.max_tries,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)
    main()
