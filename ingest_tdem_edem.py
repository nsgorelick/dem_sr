"""Download, normalize, upload, and ingest TDM30 EDEM tiles.

This CLI consumes a text file of authenticated DLR download URLs, then processes
each tile through the following restart-safe stages:

1. Download the source ZIP.
2. Extract the primary DEM GeoTIFF.
3. Rewrite the GeoTIFF with simple Earth Engine friendly CRS metadata.
4. Upload the normalized GeoTIFF to Google Cloud Storage.
5. Start or resume Earth Engine ingestion into an ImageCollection.

The script is designed to resume cleanly after interruption by consulting:

- local files under ``--workspace-dir``
- the configured GCS prefix
- the configured Earth Engine collection
- a local JSON state ledger for in-flight ingestion operations
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptive_export_runner import AdaptiveThreadExportRunner
from export_progress import ExportProgressLine

log = logging.getLogger("ingest_tdem_edem")

DEFAULT_URL_LIST = Path("TDEM_list.txt")
DEFAULT_WORKSPACE_DIR = Path("tdem_ingest")
DEFAULT_GCS_URI = "gs://ee-gorelick-upload/tdem_edem/tifs"
DEFAULT_EE_COLLECTION = "users/ngorelick/DTM/TDEM_EDEM"
DEFAULT_OUTPUT_CRS = "EPSG:4326"
DEFAULT_WORKERS = 16
DEFAULT_MAX_TRIES = 5
DEFAULT_URL_TIMEOUT_SEC = 120.0
DEFAULT_POLL_INTERVAL_SEC = 15.0
DEFAULT_PROGRESS_RATE_WINDOW_SEC = 60.0
# Fill these in before running the downloader.
DLR_USERNAME = "ngorelick"
DLR_PASSWORD = "etkBa26n5@LDjKy"
ZIP_EXT = ".zip"
TIF_EXT = ".tif"
STATE_FILE_NAME = "state.json"
STATE_VERSION = 1
RAW_SUBDIR = "raw"
FIXED_SUBDIR = "fixed"
ZIP_SUBDIR = "zip"
MANIFEST_SUBDIR = "manifests"
PREFERRED_DEM_SUFFIX = "_EDEM_EGM.tif"
OUTPUT_PRODUCT_SUFFIX = "_EDEM_EGM"

DONE_STAGE = "done"
INGEST_PENDING_STAGE = "ingest_pending"
UPLOAD_PENDING_STAGE = "upload_pending"
NORMALIZE_PENDING_STAGE = "normalize_pending"
EXTRACT_PENDING_STAGE = "extract_pending"
DOWNLOAD_PENDING_STAGE = "download_pending"
FAILED_STAGE = "failed"
ACTIVE_INGEST_STAGES = {"ingest_started", "ingesting"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_tile_stage(tile_id: str, stage: str, detail: str | None = None) -> None:
    msg = f"[tile {tile_id}] {stage}"
    if detail:
        msg += f": {detail}"
    log.info(msg)


def _require_ee() -> Any:
    try:
        import ee
    except ImportError as exc:  # pragma: no cover - exercised only in live envs
        raise RuntimeError(
            "Earth Engine support requires the 'earthengine-api' package."
        ) from exc
    return ee


def _require_rasterio() -> Any:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - exercised only in live envs
        raise RuntimeError(
            "GeoTIFF normalization requires the 'rasterio' package."
        ) from exc
    return rasterio


def _require_remotezip() -> Any:
    try:
        from remotezip import RemoteZip
    except ImportError as exc:  # pragma: no cover - exercised only in live envs
        raise RuntimeError(
            "Remote ZIP extraction requires the 'remotezip' package."
        ) from exc
    return RemoteZip


@dataclass(frozen=True)
class GcsTarget:
    bucket: str
    prefix: str


@dataclass(frozen=True)
class TilePaths:
    zip_path: Path
    raw_tif_path: Path
    fixed_tif_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class TileItem:
    tile_id: str
    output_stem: str
    url: str
    gcs_uri: str
    gcs_object_name: str
    asset_id: str
    paths: TilePaths


class StateLedger:
    """Thread-safe JSON ledger for local tile state."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._payload = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": STATE_VERSION, "tiles": {}}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if "tiles" not in raw:
            raw = {"version": STATE_VERSION, "tiles": raw}
        raw.setdefault("version", STATE_VERSION)
        return raw

    def _write_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        part_path = self.path.with_suffix(self.path.suffix + ".part")
        part_path.write_text(json.dumps(self._payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(part_path, self.path)

    def get(self, tile_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._payload["tiles"].get(tile_id, {}))

    def items(self) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            return [(k, dict(v)) for k, v in self._payload["tiles"].items()]

    def update(self, tile_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            record = dict(self._payload["tiles"].get(tile_id, {}))
            record.update(fields)
            record["updated_at"] = utc_now_iso()
            self._payload["tiles"][tile_id] = record
            self._write_locked()
            return dict(record)

    def mark_error(self, tile_id: str, *, stage: str, error: str) -> dict[str, Any]:
        return self.update(tile_id, stage=stage, last_error=error)


class SharedInventory:
    """In-memory view of remote completion state updated during the run."""

    def __init__(self, existing_gcs: set[str], existing_assets: set[str]) -> None:
        self._gcs = set(existing_gcs)
        self._assets = set(existing_assets)
        self._lock = threading.Lock()

    def has_gcs(self, tile_id: str) -> bool:
        with self._lock:
            return tile_id in self._gcs

    def has_asset(self, tile_id: str) -> bool:
        with self._lock:
            return tile_id in self._assets

    def add_gcs(self, tile_id: str) -> None:
        with self._lock:
            self._gcs.add(tile_id)

    def add_asset(self, tile_id: str) -> None:
        with self._lock:
            self._assets.add(tile_id)


class DlrAuthenticator:
    """Open authenticated URLs against the DLR download host."""

    def __init__(self, username: str, password: str) -> None:
        self._username = username
        self._password = password
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, "https://download.geoservice.dlr.de/", username, password)
        basic = urllib.request.HTTPBasicAuthHandler(password_mgr)
        digest = urllib.request.HTTPDigestAuthHandler(password_mgr)
        self._opener = urllib.request.build_opener(basic, digest)
        credentials = f"{self._username}:{self._password}".encode("utf-8")
        self._basic_auth_header = base64.b64encode(credentials).decode("ascii")

    def open(self, url: str, *, timeout: float) -> Any:
        request = urllib.request.Request(url)
        request.add_header("Authorization", f"Basic {self._basic_auth_header}")
        return self._opener.open(request, timeout=timeout)

    def request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Basic {self._basic_auth_header}",
            "User-Agent": "Mozilla/5.0",
        }


def read_url_list(path: Path, *, prefix: str | None, limit: int | None) -> list[str]:
    urls: list[str] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        tile_id = tile_id_from_url(url)
        if prefix and not tile_id.startswith(prefix):
            continue
        urls.append(url)
    return urls


def apply_work_limit(items: list[TileItem], stage_by_tile: dict[str, str], limit: int | None) -> list[TileItem]:
    pending_items = [item for item in items if stage_by_tile[item.tile_id] != DONE_STAGE]
    if limit is None:
        return pending_items
    return pending_items[:limit]


def tile_id_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    name = Path(path).name
    if not name.endswith(ZIP_EXT):
        raise ValueError(f"Expected .zip URL, got {url}")
    return name[: -len(ZIP_EXT)]


def parse_gcs_uri(uri: str) -> GcsTarget:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Expected gs://bucket/prefix, got {uri}")
    prefix = parsed.path.lstrip("/").rstrip("/")
    return GcsTarget(bucket=parsed.netloc, prefix=prefix)


def join_gcs_object(prefix: str, name: str) -> str:
    if not prefix:
        return name
    return f"{prefix}/{name}"


def asset_basename(asset_id: str) -> str:
    return asset_id.rstrip("/").split("/")[-1]


def output_stem_for_tile_id(tile_id: str) -> str:
    return f"{tile_id}{OUTPUT_PRODUCT_SUFFIX}"


def tile_id_from_output_stem(name: str) -> str:
    if name.endswith(OUTPUT_PRODUCT_SUFFIX):
        return name[: -len(OUTPUT_PRODUCT_SUFFIX)]
    return name


def manifest_name_for_asset(asset_id: str) -> str:
    if asset_id.startswith("projects/earthengine-legacy/assets/"):
        return asset_id
    if asset_id.startswith("users/"):
        return f"projects/earthengine-legacy/assets/{asset_id}"
    return asset_id


def tile_paths(workspace_dir: Path, tile_id: str, output_stem: str | None = None) -> TilePaths:
    output_stem = output_stem or output_stem_for_tile_id(tile_id)
    return TilePaths(
        zip_path=workspace_dir / ZIP_SUBDIR / f"{tile_id}{ZIP_EXT}",
        raw_tif_path=workspace_dir / RAW_SUBDIR / f"{output_stem}{TIF_EXT}",
        fixed_tif_path=workspace_dir / FIXED_SUBDIR / f"{output_stem}{TIF_EXT}",
        manifest_path=workspace_dir / MANIFEST_SUBDIR / f"{output_stem}.json",
    )


def build_tile_items(urls: list[str], gcs_target: GcsTarget, ee_collection: str, workspace_dir: Path) -> list[TileItem]:
    items: list[TileItem] = []
    for url in urls:
        tile_id = tile_id_from_url(url)
        output_stem = output_stem_for_tile_id(tile_id)
        object_name = join_gcs_object(gcs_target.prefix, f"{output_stem}{TIF_EXT}")
        asset_id = f"{ee_collection.rstrip('/')}/{output_stem}"
        items.append(
            TileItem(
                tile_id=tile_id,
                output_stem=output_stem,
                url=url,
                gcs_uri=f"gs://{gcs_target.bucket}/{object_name}",
                gcs_object_name=object_name,
                asset_id=asset_id,
                paths=tile_paths(workspace_dir, tile_id, output_stem),
            )
        )
    return items


def primary_dem_score(tile_id: str, member_name: str) -> tuple[int, str]:
    base = Path(member_name).name
    stem = Path(base).stem.upper()
    tile_upper = tile_id.upper()
    upper = base.upper()
    bad_needles = (
        "_EDM",
        "_HEM",
        "_WAM",
        "_COM",
        "_LCM",
        "_AMP",
        "_MASK",
        "_ERR",
        "_ERROR",
        "_SHADE",
        "_HILL",
        "_PREVIEW",
    )
    score = 0
    if upper == f"{tile_upper}{TIF_EXT.upper()}":
        score += 100
    if stem == tile_upper:
        score += 60
    if "EDEM" in upper:
        score += 20
    if "DEM" in upper:
        score += 10
    if any(needle in upper for needle in bad_needles):
        score -= 80
    return score, upper


def select_primary_dem_member(tile_id: str, names: list[str]) -> str:
    tif_names = [name for name in names if name.lower().endswith((".tif", ".tiff"))]
    if not tif_names:
        raise FileNotFoundError(f"No GeoTIFF found in zip for {tile_id}")
    preferred = [
        name
        for name in tif_names
        if Path(name).name.upper().endswith(PREFERRED_DEM_SUFFIX.upper())
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        raise RuntimeError(f"Multiple {PREFERRED_DEM_SUFFIX} files found in zip for {tile_id}")
    ranked = sorted(tif_names, key=lambda name: primary_dem_score(tile_id, name), reverse=True)
    available = ", ".join(Path(name).name for name in ranked)
    raise FileNotFoundError(
        f"Preferred DEM {PREFERRED_DEM_SUFFIX} not found in zip for {tile_id}. "
        f"Available TIFFs: {available}"
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def cleanup_local_artifacts(paths: TilePaths) -> None:
    for path in (
        paths.zip_path,
        paths.raw_tif_path,
        paths.fixed_tif_path,
        paths.manifest_path,
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        part_path = path.with_suffix(path.suffix + ".part")
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_copy_stream(src: Any, dst_path: Path) -> None:
    ensure_parent(dst_path)
    part_path = dst_path.with_suffix(dst_path.suffix + ".part")
    try:
        with part_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(part_path, dst_path)
    except Exception:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def extract_primary_tif(zip_path: Path, tile_id: str, output_path: Path) -> None:
    ensure_parent(output_path)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            member = select_primary_dem_member(tile_id, zf.namelist())
            with zf.open(member) as src, part_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        os.replace(part_path, output_path)
    except Exception:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def extract_primary_tif_from_remote_zip(
    url: str,
    tile_id: str,
    output_path: Path,
    *,
    auth: DlrAuthenticator,
    timeout: float,
    remotezip_cls: Any | None = None,
) -> str:
    """Extract the preferred DEM TIFF directly from a remote ZIP via HTTP range requests."""
    RemoteZip = remotezip_cls or _require_remotezip()
    ensure_parent(output_path)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    try:
        with RemoteZip(
            url,
            headers=auth.request_headers(),
            timeout=timeout,
            initial_buffer_size=65536,
        ) as zf:
            member = select_primary_dem_member(tile_id, zf.namelist())
            with zf.open(member) as src, part_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        os.replace(part_path, output_path)
        return member
    except Exception:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def normalize_crs_for_ee(input_path: Path, output_path: Path, *, output_crs: str) -> None:
    """Rewrite the GeoTIFF with a simple CRS and atomic output."""
    rasterio = _require_rasterio()
    ensure_parent(output_path)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    _rio_env = logging.getLogger("rasterio._env")
    saved = _rio_env.level
    _rio_env.setLevel(logging.ERROR)
    try:
        try:
            with rasterio.open(input_path) as src:
                data = src.read()
                profile = src.profile.copy()
                profile.pop("gcps", None)
                profile.pop("rpcs", None)
                profile.update(
                    driver="GTiff",
                    crs=output_crs,
                    transform=src.transform,
                    compress="ZSTD",
                    ZSTD_LEVEL=1,
                    tiled=True,
                    bigtiff="IF_SAFER",
                )
                profile.setdefault("nodata", src.nodata)
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
            os.replace(part_path, output_path)
        except Exception:
            try:
                part_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    finally:
        _rio_env.setLevel(saved)


def classify_tile(item: TileItem, ledger_record: dict[str, Any], inventory: SharedInventory) -> str:
    if inventory.has_asset(item.tile_id):
        return DONE_STAGE
    if inventory.has_gcs(item.tile_id):
        return INGEST_PENDING_STAGE
    if item.paths.fixed_tif_path.is_file():
        return UPLOAD_PENDING_STAGE
    if item.paths.raw_tif_path.is_file():
        return NORMALIZE_PENDING_STAGE
    if item.paths.zip_path.is_file():
        return EXTRACT_PENDING_STAGE
    if ledger_record.get("stage") == FAILED_STAGE:
        return FAILED_STAGE
    return DOWNLOAD_PENDING_STAGE


def ee_state_from_operation(operation: dict[str, Any]) -> str:
    if not operation:
        return "UNKNOWN"
    metadata = operation.get("metadata") or {}
    state = metadata.get("state")
    if state:
        return str(state)
    if operation.get("done"):
        if "error" in operation:
            return "FAILED"
        return "SUCCEEDED"
    return "RUNNING"


def parse_ingestion_response(task_id: str, response: Any) -> tuple[str | None, str | None]:
    operation_name = None
    returned_task_id = task_id
    if isinstance(response, dict):
        operation_name = response.get("name")
        metadata = response.get("metadata") or {}
        returned_task_id = metadata.get("id") or response.get("id") or task_id
    elif isinstance(response, str):
        operation_name = response
    return returned_task_id, operation_name


def list_gcs_tile_ids(target: GcsTarget) -> set[str]:
    prefix = target.prefix.rstrip("/")
    pattern = f"gs://{target.bucket}/{prefix}/*{TIF_EXT}" if prefix else f"gs://{target.bucket}/*{TIF_EXT}"
    stdout = run_command(
        ["gsutil", "ls", pattern],
        ok_empty_match=("One or more URLs matched no objects", "matched no objects"),
    )
    return parse_gsutil_ls_output(stdout)


def list_ee_tile_ids(ee: Any, collection_id: str) -> set[str]:
    del ee
    stdout = run_command(["earthengine", "ls", collection_id])
    return parse_earthengine_ls_output(stdout)


def ensure_ee_collection(ee: Any, collection_id: str) -> None:
    try:
        ee.data.getAsset(collection_id)
        return
    except Exception as exc:
        message = str(exc).lower()
        if "not found" not in message and "asset not found" not in message:
            raise
    ee.data.createAsset({"type": "IMAGE_COLLECTION"}, collection_id)


def build_ingestion_manifest(item: TileItem) -> dict[str, Any]:
    rasterio = _require_rasterio()
    manifest = {
        "name": manifest_name_for_asset(item.asset_id),
        "tilesets": [
            {
                "sources": [{"uris": [item.gcs_uri]}],
            }
        ],
        "bands": [
            {
                "id": "elevation",
                "tilesetBandIndex": 0,
                "pyramidingPolicy": "MEAN",
            }
        ],
        "properties": {
            "source": "DLR_TDM30_EDEM",
            "tile_id": item.tile_id,
            "source_url": item.url,
        },
    }
    with rasterio.open(item.paths.fixed_tif_path) as src:
        if src.nodata is not None:
            manifest["missingData"] = {"values": [float(src.nodata)]}
    return manifest


def operation_error_message(operation: dict[str, Any]) -> str:
    error = operation.get("error") or {}
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return str(message)
    metadata = operation.get("metadata") or {}
    if isinstance(metadata, dict):
        message = metadata.get("error_message") or metadata.get("errorMessage")
        if message:
            return str(message)
    return json.dumps(operation, sort_keys=True)


def download_zip(item: TileItem, auth: DlrAuthenticator, *, timeout: float) -> None:
    if item.paths.zip_path.is_file():
        return
    with auth.open(item.url, timeout=timeout) as response:
        atomic_copy_stream(response, item.paths.zip_path)


def upload_fixed_tif(item: TileItem, inventory: SharedInventory) -> None:
    if inventory.has_gcs(item.tile_id):
        return
    run_command(["gsutil", "-q", "cp", "-n", str(item.paths.fixed_tif_path), item.gcs_uri])
    inventory.add_gcs(item.tile_id)


def poll_ingestion_operation(ee: Any, operation_name: str, *, poll_interval_sec: float) -> dict[str, Any]:
    while True:
        operation = ee.data.getOperation(operation_name)
        if operation.get("done"):
            return operation
        time.sleep(poll_interval_sec)


def poll_ingestion_task(ee: Any, task_id: str, *, poll_interval_sec: float) -> dict[str, Any]:
    get_status = getattr(ee.data, "getTaskStatus", None)
    if get_status is None:
        raise RuntimeError("Earth Engine client did not return an operation name and getTaskStatus() is unavailable.")
    while True:
        statuses = get_status([task_id])
        status = statuses[0] if statuses else {}
        state = str(status.get("state", "")).upper()
        if state in {"COMPLETED", "SUCCEEDED"}:
            return status
        if state in {"FAILED", "CANCELLED", "CANCEL_REQUESTED"}:
            raise RuntimeError(str(status.get("error_message") or status))
        time.sleep(poll_interval_sec)


def start_or_resume_ingestion(
    ee: Any,
    item: TileItem,
    *,
    ledger: StateLedger,
    poll_interval_sec: float,
) -> str:
    record = ledger.get(item.tile_id)
    operation_name = record.get("operation_name")
    if operation_name:
        operation = ee.data.getOperation(operation_name)
        state = ee_state_from_operation(operation)
        if state in {"PENDING", "RUNNING"} or not operation.get("done"):
            return operation_name
        if state == "SUCCEEDED":
            return operation_name
        if state in {"FAILED", "CANCELLED"}:
            raise RuntimeError(operation_error_message(operation))
    task_id = ee.data.newTaskId()[0]
    manifest = build_ingestion_manifest(item)
    ensure_parent(item.paths.manifest_path)
    item.paths.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ledger.update(
        item.tile_id,
        stage="ingest_started",
        task_id=task_id,
        operation_name=None,
        asset_id=item.asset_id,
        gcs_uri=item.gcs_uri,
        source_url=item.url,
    )
    response = ee.data.startIngestion(task_id, manifest)
    task_id, operation_name = parse_ingestion_response(task_id, response)
    ledger.update(
        item.tile_id,
        stage="ingesting",
        task_id=task_id,
        operation_name=operation_name,
        asset_id=item.asset_id,
        gcs_uri=item.gcs_uri,
    )
    return operation_name or str(task_id)


def run_command(command: list[str], *, ok_empty_match: tuple[str, ...] = ()) -> str:
    """Run a CLI command and return stdout.

    Some listing CLIs return a non-zero exit code for an empty result set; callers
    can pass known stderr fragments via ``ok_empty_match`` to treat that as success.
    """
    log.info("Running command: %s", " ".join(str(part) for part in command))
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        return proc.stdout
    stderr = (proc.stderr or "").strip()
    if ok_empty_match and any(fragment in stderr for fragment in ok_empty_match):
        return ""
    quoted = " ".join(str(part) for part in command)
    raise RuntimeError(f"Command failed ({quoted}): {stderr or proc.stdout.strip()}")


def parse_gsutil_ls_output(stdout: str) -> set[str]:
    return {
        tile_id_from_output_stem(Path(line.strip()).stem)
        for line in stdout.splitlines()
        if line.strip().startswith("gs://") and line.strip().endswith(TIF_EXT)
    }


def parse_earthengine_ls_output(stdout: str) -> set[str]:
    return {
        tile_id_from_output_stem(asset_basename(line.strip()))
        for line in stdout.splitlines()
        if line.strip() and not line.strip().startswith("Running command")
    }


def refresh_active_ingestions(ee: Any, items_by_tile: dict[str, TileItem], ledger: StateLedger) -> None:
    active_count = sum(1 for _, record in ledger.items() if record.get("stage") in ACTIVE_INGEST_STAGES)
    if active_count:
        log.info("Refreshing %d in-flight ingestion(s) from local state ledger", active_count)
    for tile_id, record in ledger.items():
        if record.get("stage") not in ACTIVE_INGEST_STAGES:
            continue
        item = items_by_tile.get(tile_id)
        if item is None:
            continue
        try:
            ee.data.getAsset(item.asset_id)
            ledger.update(tile_id, stage=DONE_STAGE)
            log_tile_stage(tile_id, "resume", "asset already present in Earth Engine")
            continue
        except Exception:
            pass
        operation_name = record.get("operation_name")
        if not operation_name:
            continue
        try:
            operation = ee.data.getOperation(operation_name)
        except Exception as exc:
            ledger.mark_error(tile_id, stage=FAILED_STAGE, error=f"operation lookup failed: {exc}")
            continue
        state = ee_state_from_operation(operation)
        if state == "SUCCEEDED":
            ledger.update(tile_id, stage=DONE_STAGE)
            log_tile_stage(tile_id, "resume", "ingestion already succeeded")
        elif state in {"FAILED", "CANCELLED"}:
            ledger.mark_error(tile_id, stage=FAILED_STAGE, error=operation_error_message(operation))
            log_tile_stage(tile_id, "resume", f"ingestion ended with {state}")
        else:
            ledger.update(tile_id, stage="ingesting")
            log_tile_stage(tile_id, "resume", f"still {state}")


def initialize_earth_engine(project: str | None, *, high_volume: bool) -> Any:
    ee = _require_ee()
    kwargs: dict[str, Any] = {}
    if project:
        kwargs["project"] = project
    if high_volume:
        kwargs["opt_url"] = "https://earthengine-highvolume.googleapis.com"
    ee.Initialize(**kwargs)
    return ee


def summarize_stages(stage_by_tile: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for stage in stage_by_tile.values():
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def process_tile(
    item: TileItem,
    *,
    auth: DlrAuthenticator,
    ee: Any,
    inventory: SharedInventory,
    ledger: StateLedger,
    output_crs: str,
    url_timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, str]:
    record = ledger.get(item.tile_id)
    stage = classify_tile(item, record, inventory)
    ledger.update(
        item.tile_id,
        stage=stage,
        asset_id=item.asset_id,
        gcs_uri=item.gcs_uri,
        source_url=item.url,
    )
    if stage == DONE_STAGE:
        cleanup_local_artifacts(item.paths)
        log_tile_stage(item.tile_id, "skip", "already present in Earth Engine")
        return {"tile_id": item.tile_id, "stage": DONE_STAGE}
    try:
        if stage in {DOWNLOAD_PENDING_STAGE, FAILED_STAGE}:
            ledger.update(item.tile_id, stage="extracting")
            log_tile_stage(item.tile_id, "remote-extract", item.url)
            try:
                member = extract_primary_tif_from_remote_zip(
                    item.url,
                    item.tile_id,
                    item.paths.raw_tif_path,
                    auth=auth,
                    timeout=url_timeout_sec,
                )
                log_tile_stage(item.tile_id, "remote-extract-done", member)
                stage = NORMALIZE_PENDING_STAGE
            except Exception as exc:
                log.warning(
                    "Remote ZIP extraction failed for %s; falling back to full ZIP download: %s",
                    item.tile_id,
                    exc,
                )
                ledger.update(item.tile_id, stage="downloading")
                log_tile_stage(item.tile_id, "download", item.url)
                download_zip(item, auth, timeout=url_timeout_sec)
                stage = EXTRACT_PENDING_STAGE
        if stage == EXTRACT_PENDING_STAGE:
            ledger.update(item.tile_id, stage="extracting")
            log_tile_stage(item.tile_id, "extract", str(item.paths.zip_path))
            extract_primary_tif(item.paths.zip_path, item.tile_id, item.paths.raw_tif_path)
            stage = NORMALIZE_PENDING_STAGE
        if stage == NORMALIZE_PENDING_STAGE:
            ledger.update(item.tile_id, stage="normalizing")
            log_tile_stage(item.tile_id, "normalize", str(item.paths.raw_tif_path))
            normalize_crs_for_ee(item.paths.raw_tif_path, item.paths.fixed_tif_path, output_crs=output_crs)
            stage = UPLOAD_PENDING_STAGE
        if stage == UPLOAD_PENDING_STAGE:
            ledger.update(item.tile_id, stage="uploading")
            log_tile_stage(item.tile_id, "upload", item.gcs_uri)
            upload_fixed_tif(item, inventory)
            stage = INGEST_PENDING_STAGE
        if stage == INGEST_PENDING_STAGE:
            ledger.update(item.tile_id, stage=INGEST_PENDING_STAGE)
            log_tile_stage(item.tile_id, "ingest", item.asset_id)
            op_ref = start_or_resume_ingestion(
                ee,
                item,
                ledger=ledger,
                poll_interval_sec=poll_interval_sec,
            )
            ledger.update(item.tile_id, stage="ingesting", operation_name=op_ref, last_error=None)
            cleanup_local_artifacts(item.paths)
            log_tile_stage(item.tile_id, "cleanup", "removed local ZIP/TIFF artifacts")
            log_tile_stage(item.tile_id, "queued", op_ref)
            return {"tile_id": item.tile_id, "stage": "ingesting"}
        ledger.update(item.tile_id, stage=DONE_STAGE, last_error=None)
        cleanup_local_artifacts(item.paths)
        log_tile_stage(item.tile_id, "done")
        return {"tile_id": item.tile_id, "stage": DONE_STAGE}
    except Exception as exc:
        ledger.mark_error(item.tile_id, stage=FAILED_STAGE, error=str(exc))
        log_tile_stage(item.tile_id, "failed", str(exc))
        raise RuntimeError(f"[tile {item.tile_id}] {exc}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-list", type=Path, default=DEFAULT_URL_LIST, help="Text file containing one authenticated DLR ZIP URL per line.")
    parser.add_argument("--workspace-dir", type=Path, default=DEFAULT_WORKSPACE_DIR, help="Local workspace used for ZIPs, extracted TIFFs, manifests, and state.")
    parser.add_argument("--gcs-uri", default=DEFAULT_GCS_URI, help="Destination GCS prefix, e.g. gs://bucket/path/to/tdem.")
    parser.add_argument("--ee-collection", default=DEFAULT_EE_COLLECTION, help="Destination Earth Engine ImageCollection path.")
    parser.add_argument("--ee-project", default=None, help="Optional Earth Engine Cloud project for ee.Initialize().")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Initial worker count for adaptive concurrency.")
    parser.add_argument("--max-tries", type=int, default=DEFAULT_MAX_TRIES, help="Retries per tile for transient failures.")
    parser.add_argument("--prefix", default=None, help="Optional tile-id prefix filter applied to the URL list.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit after filtering the URL list.")
    parser.add_argument("--output-crs", default=DEFAULT_OUTPUT_CRS, help="CRS written into normalized GeoTIFFs before upload.")
    parser.add_argument("--url-timeout-sec", type=float, default=DEFAULT_URL_TIMEOUT_SEC, help="HTTP timeout used when downloading DLR ZIPs.")
    parser.add_argument("--poll-interval-sec", type=float, default=DEFAULT_POLL_INTERVAL_SEC, help="Polling interval for Earth Engine ingestion operations.")
    parser.add_argument("--dry-run", action="store_true", help="Only classify tiles and print stage counts; do not process tiles.")
    parser.add_argument("--no-high-volume", action="store_true", help="Disable the Earth Engine high-volume endpoint.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level.",
    )
    return parser


def ensure_workspace_dirs(workspace_dir: Path) -> None:
    for name in (ZIP_SUBDIR, RAW_SUBDIR, FIXED_SUBDIR, MANIFEST_SUBDIR):
        (workspace_dir / name).mkdir(parents=True, exist_ok=True)


def load_credentials() -> tuple[str, str]:
    if not DLR_USERNAME or not DLR_PASSWORD:
        raise RuntimeError(
            "Set DLR_USERNAME and DLR_PASSWORD constants at the top of ingest_tdem_edem.py."
        )
    return DLR_USERNAME, DLR_PASSWORD


def run_ingest(args: argparse.Namespace) -> int:
    ensure_workspace_dirs(args.workspace_dir)
    log.info("Loading URL list from %s", args.url_list)
    urls = read_url_list(args.url_list, prefix=args.prefix, limit=args.limit)
    if not urls:
        print("No URLs matched the requested filters.", flush=True)
        return 0

    gcs_target = parse_gcs_uri(args.gcs_uri)
    log.info("Using workspace directory %s", args.workspace_dir)
    log.info("Target GCS prefix: %s", args.gcs_uri)
    log.info("Target EE collection: %s", args.ee_collection)
    log.info("Building tile catalog for %d URL(s)", len(urls))
    items = build_tile_items(urls, gcs_target, args.ee_collection, args.workspace_dir)
    items_by_tile = {item.tile_id: item for item in items}
    ledger = StateLedger(args.workspace_dir / STATE_FILE_NAME)

    log.info("Initializing Earth Engine")
    ee = initialize_earth_engine(args.ee_project, high_volume=not args.no_high_volume)
    ensure_ee_collection(ee, args.ee_collection)
    refresh_active_ingestions(ee, items_by_tile, ledger)

    log.info("Listing existing GCS objects")
    existing_gcs = list_gcs_tile_ids(gcs_target)
    log.info("Found %d existing GCS object(s)", len(existing_gcs))
    log.info("Listing existing Earth Engine assets")
    existing_assets = list_ee_tile_ids(ee, args.ee_collection)
    log.info("Found %d existing Earth Engine asset(s)", len(existing_assets))
    inventory = SharedInventory(existing_gcs, existing_assets)

    stage_by_tile = {
        item.tile_id: classify_tile(item, ledger.get(item.tile_id), inventory)
        for item in items
    }
    counts = summarize_stages(stage_by_tile)
    total = len(items)
    already_done = counts.get(DONE_STAGE, 0)
    print(
        f"catalog {total} tiles, {already_done} already in EE, "
        f"{counts.get(INGEST_PENDING_STAGE, 0)} ready to ingest, "
        f"{counts.get(UPLOAD_PENDING_STAGE, 0)} ready to upload, "
        f"{counts.get(NORMALIZE_PENDING_STAGE, 0)} ready to normalize, "
        f"{counts.get(EXTRACT_PENDING_STAGE, 0)} ready to extract, "
        f"{counts.get(DOWNLOAD_PENDING_STAGE, 0)} ready to download",
        flush=True,
    )
    if args.dry_run:
        return 0

    pending_items = apply_work_limit(items, stage_by_tile, args.limit)
    if args.limit is not None:
        log.info("Applying work limit %d after skip classification", args.limit)
    if not pending_items:
        print("done (nothing to run).", flush=True)
        return 0

    log.info("Starting work on %d pending tile(s) with %d worker(s)", len(pending_items), args.workers)
    username, password = load_credentials()
    auth = DlrAuthenticator(username, password)
    runner = AdaptiveThreadExportRunner(
        min_concurrent=1,
        max_concurrent=max(args.workers * 2, 1),
        initial_concurrent=max(args.workers, 1),
        quiet_before_scale_up_sec=10.0,
        scale_down_step=1,
        scale_up_step=1,
        max_tries=args.max_tries,
        retry_base_delay_sec=1.0,
        retry_backoff=2.0,
    )
    progress = ExportProgressLine(rate_window_sec=DEFAULT_PROGRESS_RATE_WINDOW_SEC)
    errors: list[str] = []
    error_lock = threading.Lock()

    def on_task_error(exc: Exception) -> None:
        with error_lock:
            errors.append(str(exc))
        print(f"\n[tile failed] {exc}", file=sys.stderr, flush=True)

    worker = lambda item: process_tile(  # noqa: E731
        item,
        auth=auth,
        ee=ee,
        inventory=inventory,
        ledger=ledger,
        output_crs=args.output_crs,
        url_timeout_sec=args.url_timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )

    for _ in runner.run_unordered(
        worker,
        pending_items,
        on_each_done=progress.emit,
        rate_window_sec=DEFAULT_PROGRESS_RATE_WINDOW_SEC,
        continue_on_errors=True,
        on_task_error=on_task_error,
    ):
        pass
    progress.end_line()

    if errors:
        print(f"completed with {len(errors)} failed tile(s). See {ledger.path}.", flush=True)
        return 1
    active_after_run = sum(
        1 for _, record in ledger.items() if record.get("stage") in ACTIVE_INGEST_STAGES
    )
    if active_after_run:
        print(
            f"submitted work for all pending tiles; {active_after_run} ingestion(s) still running in Earth Engine.",
            flush=True,
        )
    else:
        print("done.", flush=True)
    return 0


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)
    raise SystemExit(run_ingest(args))


if __name__ == "__main__":
    main()
