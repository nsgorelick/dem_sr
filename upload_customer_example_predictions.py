#!/usr/bin/env python3
"""Generate model prediction TIFFs for selected chips and upload them to Earth Engine."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import ee
import rasterio
import torch
from torch.utils.data import DataLoader

from dem_film_unet import DemFilmUNet
from ingest_tdem_edem import (
    ensure_ee_collection,
    join_gcs_object,
    list_gcs_tile_ids,
    manifest_name_for_asset,
    parse_gcs_uri,
    run_command,
)
from local_patch_dataset import LocalDemPatchDataset, collate_dem_batch, load_patch_stems_manifest

log = logging.getLogger("upload_customer_example_predictions")

EE_INIT_KWARGS: dict[str, str] = {
    "project": "ngorelick",
    "opt_url": "https://earthengine-highvolume.googleapis.com",
}


def list_ee_asset_ids(collection_id: str) -> set[str]:
    """List existing asset basenames in an ImageCollection via the EE Python API."""
    out: set[str] = set()
    page_token = None
    while True:
        payload = {"parent": collection_id, "pageSize": 1000}
        if page_token:
            payload["pageToken"] = page_token
        response = ee.data.listAssets(payload)
        for asset in response.get("assets", []):
            name = asset.get("name", "")
            if name:
                out.add(name.rstrip("/").split("/")[-1])
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return out


def write_prediction_tif(
    stem: str,
    pred: torch.Tensor,
    *,
    stack_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    with rasterio.open(stack_path) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            compress="ZSTD",
            ZSTD_LEVEL=1,
            predictor=3,
            tiled=True,
            bigtiff="IF_SAFER",
        )
        nodata = profile.get("nodata")
        if nodata is not None and not torch.isfinite(torch.tensor(float(nodata))):
            profile.pop("nodata", None)
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("interleave", None)
        tags = src.tags()

    arr = pred.detach().cpu().numpy().astype("float32", copy=False)
    try:
        with rasterio.open(part_path, "w", **profile) as dst:
            dst.write(arr, 1)
            dst.set_band_description(1, "elevation")
            dst.update_tags(**tags)
            dst.update_tags(source="dem_film_unet_prediction", patch_stem=stem)
        part_path.replace(output_path)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise


def build_manifest(asset_id: str, gcs_uri: str, tif_path: Path) -> dict[str, object]:
    manifest = {
        "name": manifest_name_for_asset(asset_id),
        "tilesets": [{"sources": [{"uris": [gcs_uri]}]}],
        "bands": [
            {
                "id": "elevation",
                "tilesetBandIndex": 0,
                "pyramidingPolicy": "MEAN",
            }
        ],
        "properties": {
            "source": "dem_film_unet_prediction",
            "local_path": str(tif_path),
        },
    }
    with rasterio.open(tif_path) as src:
        if src.nodata is not None and math.isfinite(float(src.nodata)):
            manifest["missingData"] = {"values": [float(src.nodata)]}
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("customer_example_chips_manifest.txt"),
        help="One selected patch stem per line",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("dem_film_unet.pt"))
    parser.add_argument("--data-root", default=None, help="Override training root from checkpoint")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("customer_example_predictions"),
        help="Local directory for generated TIFFs and ingestion manifests",
    )
    parser.add_argument(
        "--gcs-uri",
        default="gs://ee-gorelick-upload/customer_example_predictions/tifs",
        help="GCS prefix for uploaded prediction TIFFs",
    )
    parser.add_argument(
        "--ee-collection",
        default="users/ngorelick/DTM/tmp/customer_example_predictions",
        help="Destination Earth Engine ImageCollection",
    )
    parser.add_argument("--overwrite-local", action="store_true", help="Regenerate local TIFFs if they exist")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    stems = load_patch_stems_manifest(args.manifest)
    log.info("Manifest %s: %d stems", args.manifest, len(stems))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    data_root = args.data_root or (ckpt.get("data_root") if isinstance(ckpt, dict) else None)
    if data_root is None:
        raise RuntimeError("Set --data-root or use a checkpoint that includes data_root.")

    ds = LocalDemPatchDataset(
        data_root,
        patch_stems=stems,
        load_ae=True,
        use_precomputed_weight=False,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_dem_batch,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    model = DemFilmUNet().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    tif_dir = args.workspace_dir / "tifs"
    manifest_dir = args.workspace_dir / "manifests"
    tif_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}
    with torch.no_grad():
        for batch in loader:
            stems_batch = list(batch["stem"])
            x_dem = batch["x_dem"].to(device, non_blocking=True)
            x_ae = batch["x_ae"].to(device, non_blocking=True)
            z_lr = batch["z_lr"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x_dem, x_ae, z_lr).squeeze(1)

            for idx, stem in enumerate(stems_batch):
                out_path = tif_dir / f"{stem}.tif"
                stack_path = Path(data_root) / "stack" / f"{stem}.tif"
                write_prediction_tif(
                    stem,
                    pred[idx],
                    stack_path=stack_path,
                    output_path=out_path,
                    overwrite=args.overwrite_local,
                )
                generated[stem] = out_path

    log.info("Generated %d prediction TIFFs in %s", len(generated), tif_dir)

    gcs_target = parse_gcs_uri(args.gcs_uri)
    ee.Initialize(**EE_INIT_KWARGS)
    ensure_ee_collection(ee, args.ee_collection)
    existing_gcs = list_gcs_tile_ids(gcs_target)
    existing_assets = list_ee_asset_ids(args.ee_collection)

    submitted = []
    for stem in stems:
        tif_path = generated[stem]
        object_name = join_gcs_object(gcs_target.prefix, tif_path.name)
        gcs_uri = f"gs://{gcs_target.bucket}/{object_name}"
        asset_id = f"{args.ee_collection.rstrip('/')}/{stem}"

        if stem not in existing_gcs:
            run_command(["gsutil", "-q", "cp", "-n", str(tif_path), gcs_uri])
        else:
            log.info("GCS object already present for %s", stem)

        if stem in existing_assets:
            log.info("EE asset already present for %s", stem)
            continue

        manifest = build_manifest(asset_id, gcs_uri, tif_path)
        manifest_path = manifest_dir / f"{stem}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        task_id = ee.data.newTaskId()[0]
        response = ee.data.startIngestion(task_id, manifest)
        submitted.append(
            {
                "stem": stem,
                "asset_id": asset_id,
                "gcs_uri": gcs_uri,
                "task_id": task_id,
                "response": response,
            }
        )
        log.info("Started ingestion for %s -> %s", stem, asset_id)

    summary_path = args.workspace_dir / "submitted_ingestions.json"
    summary_path.write_text(json.dumps(submitted, indent=2), encoding="utf-8")
    log.info("Submitted %d new ingestion(s); wrote %s", len(submitted), summary_path)


if __name__ == "__main__":
    main()
