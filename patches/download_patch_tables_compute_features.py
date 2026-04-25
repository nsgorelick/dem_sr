#!/usr/bin/env python3
"""Download patch-site FeatureCollections from Earth Engine using computeFeatures.

Lists assets under a folder (default ``users/ngorelick/DTM/tmp``), keeps names
matching ``{country}_sites_{year}_{utm_zone}`` (e.g. ``de_sites_2017_32``,
``au_sites_2022_-56``), and writes each as a local GeoJSON FeatureCollection.

Default output directory is this script's directory (``patches/``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import ee

_SCRIPT_DIR = Path(__file__).resolve().parent

EE_INIT_KWARGS: dict[str, str] = {
    "opt_url": "https://earthengine-highvolume.googleapis.com",
}

# Basenames like: de_sites_2017_32, au_sites_2022_-56, nordic_sites_2024_33,
# and legacy US tables like us_sites4_2019_17.
_PATCH_TABLE_RE = re.compile(
    r"^[a-z]+_sites(?:\d+)?_(?P<year>\d{4})_(?P<zone>-?\d+)$",
    re.IGNORECASE,
)


def _asset_name_to_id(name: str) -> str:
    """Turn API ``name`` (projects/.../assets/users/...) into EE asset id."""
    marker = "/assets/"
    if marker in name:
        return name.split(marker, 1)[1]
    return name


def _list_folder_assets(parent: str) -> list[dict[str, Any]]:
    """All assets under *parent*."""
    return ee.data.listAssets({"parent": parent}).get("assets", [])


def _is_patch_table_basename(basename: str) -> bool:
    return bool(_PATCH_TABLE_RE.match(basename))


def download_feature_collection_geojson(
    asset_id: str,
    *,
    page_size: int,
    workload_tag: str | None,
) -> dict[str, Any]:
    """Return a GeoJSON FeatureCollection dict for *asset_id*."""
    fc = ee.FeatureCollection(asset_id)
    features: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {
            "expression": fc,
            "pageSize": page_size,
        }
        if page_token:
            params["pageToken"] = page_token
        if workload_tag:
            params["workloadTag"] = workload_tag

        resp = ee.data.computeFeatures(params)
        if not isinstance(resp, dict):
            raise TypeError(
                f"computeFeatures expected dict for GeoJSON mode, got {type(resp)}"
            )
        batch = resp.get("features") or []
        features.extend(batch)
        page_token = resp.get("nextPageToken") or resp.get("next_page_token")
        if not page_token:
            break

    return {"type": "FeatureCollection", "features": features}


def _download_one_asset(
    *,
    asset_id: str,
    basename: str,
    out_dir: Path,
    overwrite: bool,
    page_size: int,
    workload_tag: str | None,
) -> tuple[str, str]:
    """Download one asset and return (status, message)."""
    dest = out_dir / f"{basename}.geojson"
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and not overwrite:
        # Clean up stale partials from interrupted prior runs.
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass
        return ("skip", f"skip (exists): {dest}")

    # If overwriting or destination absent, clear any stale partial before retrying.
    try:
        part.unlink(missing_ok=True)
    except OSError:
        pass

    try:
        fc_geojson = download_feature_collection_geojson(
            asset_id,
            page_size=page_size,
            workload_tag=workload_tag,
        )
        part.write_text(json.dumps(fc_geojson, indent=2), encoding="utf-8")
        os.replace(part, dest)
        nfeat = len(fc_geojson.get("features", []))
        return ("ok", f"download: {asset_id} -> {dest}\n  wrote {nfeat} features")
    except Exception as exc:
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass
        return ("error", f"ERROR {asset_id}: {exc}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parent",
        default="users/ngorelick/DTM/tmp",
        help="Earth Engine folder id to list",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR,
        help="Directory to write ``<basename>.geojson`` files (default: this script's directory)",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=2000,
        help="computeFeatures page size (max features per request)",
    )
    p.add_argument(
        "--workload-tag",
        default=None,
        help="Optional workloadTag for computeFeatures",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download workers (default: 8)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching assets only; do not download",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing GeoJSON files",
    )
    args = p.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    ee.Initialize(**EE_INIT_KWARGS)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = _list_folder_assets(args.parent)
    matches: list[tuple[str, str]] = []
    for a in assets:
        name = a.get("name") or a.get("id") or ""
        if not name:
            continue
        asset_id = _asset_name_to_id(str(name))
        basename = asset_id.rstrip("/").split("/")[-1]
        if not _is_patch_table_basename(basename):
            continue
        matches.append((asset_id, basename))

    matches.sort(key=lambda t: t[1])
    print(
        f"folder={args.parent} listed={len(assets)} patch_tables={len(matches)}",
        flush=True,
    )

    if args.dry_run:
        for asset_id, basename in matches:
            print(f"  {basename}  ({asset_id})", flush=True)
        return

    n_ok = 0
    n_skip = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                _download_one_asset,
                asset_id=asset_id,
                basename=basename,
                out_dir=out_dir,
                overwrite=args.overwrite,
                page_size=args.page_size,
                workload_tag=args.workload_tag,
            )
            for asset_id, basename in matches
        ]
        for fut in as_completed(futs):
            status, msg = fut.result()
            if status == "ok":
                n_ok += 1
                print(msg, flush=True)
            elif status == "skip":
                n_skip += 1
                print(msg, flush=True)
            else:
                n_err += 1
                print(msg, file=sys.stderr, flush=True)

    print(
        f"done. ok={n_ok} skipped={n_skip} errors={n_err} total={len(matches)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

