"""Export DTM patch stacks and Alpha Earth annual embeddings from Earth Engine to GeoTIFF.

Writes under ``stack/`` and ``ae/`` with ZSTD compression; uses ``.part`` + atomic rename.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import ee
import rasterio
from rasterio.io import MemoryFile

from adaptive_export_runner import AdaptiveThreadExportRunner
from elvis_au import elvis_au_image_ensure_year
from export_progress import ExportProgressLine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PATCH_EXPORT_DIR = Path(".")
STACK_EXPORT_DIR = PATCH_EXPORT_DIR / "stack"
AE_EXPORT_DIR = PATCH_EXPORT_DIR / "ae"

EE_INIT_KWARGS: dict[str, str] = {
    "opt_url": "https://earthengine-highvolume.googleapis.com",
}

SATELLITE_EMBEDDING_ANNUAL = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
SATELLITE_EMBEDDING_MIN_YEAR = 2017
SATELLITE_EMBEDDING_MAX_YEAR = 2025
# Match EE catalog coverage for ``SATELLITE_EMBEDDING_ANNUAL``; FC is filtered server-side to these years.

# Backward-compatible alias (older scripts used ``HIGHVOL_URL``).
HIGHVOL_URL = EE_INIT_KWARGS["opt_url"]

EE_COMPUTEPIXELS_PARALLEL_PER_PATCH = 2
DEFAULT_EXPORT_POOL_WORKERS = 50

_TIF_EXT = ".tif"
_AE_SUFFIX = "_aef_uint8.tif"

# Rolling window for progress-line "recent patches/min" (completions in this span).
_PROGRESS_RATE_WINDOW_SEC = 60.0

ee.Initialize(**EE_INIT_KWARGS)


# ---------------------------------------------------------------------------
# Filesystem & GeoTIFF I/O
# ---------------------------------------------------------------------------


def patch_id_from_properties(props: dict[str, Any]) -> str:
    """Filename stem shared by stack + AE outputs."""
    return f"{int(props['x'])}_{int(props['y'])}_{props['zone']}_{props['country']}_{props['year']}"


def completed_patch_ids_on_disk() -> set[str]:
    """Return patch IDs that have both final stack and AE filenames (``os.listdir`` only)."""

    stack_names = os.listdir(STACK_EXPORT_DIR)
    ae_names = frozenset(os.listdir(AE_EXPORT_DIR))

    done: set[str] = set()
    for name in stack_names:
        if ".part" in name or not name.endswith(_TIF_EXT):
            continue
        oid = name[: -len(_TIF_EXT)]
        if f"{oid}{_AE_SUFFIX}" in ae_names:
            done.add(oid)
    return done


def filter_pending_patch_items(items: list[dict], completed_ids: set[str]) -> list[dict]:
    """Keep items whose patch id is not already on disk."""
    return [it for it in items if patch_id_from_properties(it["properties"]) not in completed_ids]


def write_geotiff_zstd(path: Path | str, tif_bytes: bytes, predictor: int) -> None:
    """Write GeoTIFF with ZSTD level 1; *predictor* 2 = integer bands, 3 = float.

    Writes ``*.tif.part`` then ``os.replace`` to the final path.
    EE GeoTIFFs often tag multi-band stacks with inconsistent Photometric/ExtraSamples;
    we rewrite as MINISBLACK and drop ``extra_samples`` so GDAL stops warning and output is sane.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_suffix(path.suffix + ".part")
    # Opening EE bytes still triggers a CPLE_AppDefined on mis-tagged sources; rasterio logs it here.
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
                        for b in range(1, src.count + 1):
                            desc = src.descriptions[b - 1]
                            if desc:
                                dst.set_band_description(b, desc)
                            btags = src.tags(b)
                            if btags:
                                dst.update_tags(b, **btags)
            os.replace(part_path, path)
        except Exception:
            try:
                part_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    finally:
        _rio_env.setLevel(_saved_rio)


# ---------------------------------------------------------------------------
# Earth Engine — collections & stack (naming follows existing DEM scripts)
# ---------------------------------------------------------------------------


def binaryMerge(collections):
    """Binary-merge feature collections client-side (fewer merge operations)."""
    while len(collections) > 1:
        next_collections = []
        for i in range(0, len(collections), 2):
            if i + 1 < len(collections):
                next_collections.append(
                    ee.FeatureCollection(collections[i]).merge(
                        ee.FeatureCollection(collections[i + 1])
                    )
                )
            else:
                next_collections.append(collections[i])
        collections = next_collections
    return collections[0]


def makeUTMZone(z):
    """Return the geometry footprint for a UTM zone."""
    centralMeridian = (abs(z) * 6) - 183
    return ee.Geometry.Rectangle(
        [centralMeridian - 3, -80 if z < 0 else 0, centralMeridian + 3, 0 if z < 0 else 80]
    )


def makeUTMProj(z):
    """Return the UTM projection for a signed zone id."""
    zone = abs(z)
    return ee.Projection("EPSG:327" + str(zone)) if z < 0 else ee.Projection("EPSG:326" + str(zone))


def makeCollectionDE(year, zone):
    """Combine all the DE collections and return a collection of images for year + UTMzoneNumber"""
    date = ee.Date.fromYMD(year, 1, 1)
    # Note: BW is trash; don't include it.
    cs = ["BB", "BE", "BY", "HB", "HE", "HH", "MV", "NI", "NRW", "RP", "SA", "SH", "SN", "TH"]
    collection = (
        binaryMerge([ee.ImageCollection("users/ngorelick/DTM/DE/" + s) for s in cs])
        .filterDate(date, date.advance(1, "year"))
        .filterBounds(makeUTMZone(zone))
    )
    return ee.ImageCollection(collection)


def makeCollectionAT(year, zone):
    """Return Austria HR collection for a year and zone."""
    zone = makeUTMZone(zone)
    dates = (
        ee.FeatureCollection("users/ngorelick/DTM/dates/AT")
        .filterBounds(zone)
        .filter(ee.Filter.eq("Flugjahr", str(year)))
    )
    at = ee.ImageCollection("users/ngorelick/DTM/AT").mosaic()

    return dates.map(lambda f: at.clip(zone).clip(f.geometry()))


def makeCollectionFR(year, zone):
    """Return France HR collection for a year and zone."""
    zone = makeUTMZone(zone)
    dates = (
        ee.FeatureCollection("users/ngorelick/DTM/dates/FR")
        .filterBounds(zone)
        .filter(ee.Filter.stringStartsWith("DATE_FIN", str(year)))
    )
    fr = ee.ImageCollection("users/ngorelick/DTM/FR").mosaic()

    return dates.map(lambda f: fr.clip(f.geometry()))


def makeCollectionPT(year, zone):
    """Return Portugal HR collection for a year and zone."""
    date = ee.Date.fromYMD(year, 1, 1)
    zone = makeUTMZone(zone)
    pt = (
        ee.ImageCollection("users/ngorelick/DTM/PT")
        .filterBounds(zone)
        .filterDate(date, date.advance(1, "year"))
    )
    return pt


def makeCollectionJP(year, zone):
    """Return Japan HR collection for a year and zone."""
    zone = makeUTMZone(zone)
    date = ee.Date.fromYMD(year, 1, 1)
    jp = (
        ee.ImageCollection("users/ngorelick/DTM/JP")
        .filterBounds(zone)
        .filterDate(date, date.advance(1, "year"))
    )
    return jp


def makeCollectionAU(year, zone):
    """Return Australia HR collection for a year and zone."""
    zone = makeUTMZone(zone)
    cs = (
        ee.ImageCollection("AU/ELVIS/ELVIS_5m")
        .merge(ee.ImageCollection("AU/ELVIS/ELVIS_2m"))
        .merge(ee.ImageCollection("AU/ELVIS/ELVIS_1m")
          .filter(ee.Filter.stringContains(
              "system:index", 
              "UpperNamoiNorth202304_BATCH_ELVIS_1m_PROJCSGDA2_103").not())
        .select([0], ["elevation"])
        .cast({"elevation": "float"}, ["elevation"])
        .map(elvis_au_image_ensure_year)
        .filter(ee.Filter.eq("year", int(year)))
        .filterBounds(zone)
    )
    return ee.ImageCollection(cs)


def makeCollectionNordic(year, zone):
    """Return Nordic HR collection for a year and zone."""
    zone = makeUTMZone(zone)
    dtm = ee.ImageCollection(
        [
            ee.Image("users/ngorelick/DTM/FI_2m"),
            ee.Image("users/ngorelick/DTM/SE_1m"),
            ee.Image("users/ngorelick/DTM/DK_040"),
            # ee.Image("users/ngorelick/DTM/NO"),  # Skipping; all 2025.
        ]
    )
    dates = (
        ee.FeatureCollection("users/ngorelick/DTM/dates/DK")
        .merge(ee.FeatureCollection("users/ngorelick/DTM/dates/SE"))
        .merge(ee.FeatureCollection("users/ngorelick/DTM/dates/FI"))
        .filterDate(str(year), str(year + 1))
        .filterBounds(zone)
        .map(lambda f: dtm.mosaic().clip(f.geometry()))
    )
    return dates


def makeCollectionUS(year, zone):
    """Return US HR collection for a year and zone."""
    z = makeUTMZone(zone)
    # Which projects can we identify as belonging to this year?

    date = ee.Date.fromYMD(year, 1, 1)
    wesm = (
        ee.FeatureCollection("users/ngorelick/DTM/dates/WESM")
        .filterBounds(z)
        .filter(ee.Filter.gt("collect_en", date.millis()))  # 1 year of end
        .filter(ee.Filter.lt("collect_en", date.advance(1, "year").millis()))
        .filter(ee.Filter.gt("collect_st", date.advance(-1, "year").millis()))
    )
    projects = ee.Dictionary(
        wesm.reduceColumns(ee.Reducer.frequencyHistogram(), ["project"]).get("histogram")
    ).keys()
    # Mask by year with WESM_cover, which contains YYYYMM per 100m pixel
    mask = ee.Image("users/ngorelick/DTM/dates/WESM_cover")
    mask_lo = year * 100
    mask_hi = (year + 1) * 100
    year_mask = mask.gt(mask_lo).And(mask.lt(mask_hi))
    # Get the images that are within: zone, projects[] and mask with yearMask.
    dtm = (
        ee.ImageCollection("USGS/3DEP/1m")
        .filterBounds(z)
        .map(lambda img: img.set("key", img.id().replace(".*x[0-9]+y[0-9]+_", "")))
        .filter(ee.Filter.inList("key", projects))
        .map(lambda img: img.clip(z).updateMask(year_mask))
    )
    return dtm


def makeCollection(country, year, zone):
    """Dispatch to country-specific HR collection builder."""
    builders = {
        "DE": makeCollectionDE,
        "AT": makeCollectionAT,
        "FR": makeCollectionFR,
        "PT": makeCollectionPT,
        "JP": makeCollectionJP,
        "AU": makeCollectionAU,
        "NORDIC": makeCollectionNordic,
        "US": makeCollectionUS,
    }
    if country not in builders:
        raise ValueError(f"Unsupported country: {country}")
    return ee.ImageCollection(builders[country](year, zone))


def make_z_lr10(zone):
    """Return GEDTM low-res elevation upsampled to 10 m."""
    proj = makeUTMProj(zone)
    gedtm30 = ee.ImageCollection("users/ngorelick/DTM/GEDTM30")
    z_lr = gedtm30.mosaic().divide(10).reproject(gedtm30.first().projection())
    z_lr10 = z_lr.resample("bilinear").reproject(proj.atScale(10))
    return z_lr10


def make_u_lr10(zone):
    """Return GEDTM uncertainty upsampled to 10 m."""
    proj = makeUTMProj(zone)
    gedtm30_std = ee.Image("users/ngorelick/DTM/GEDTM30_std_v20250611")
    u_lr10 = gedtm30_std.divide(100).resample("bilinear").reproject(proj.atScale(10))
    return u_lr10


def fastGaussianBlur(img, radius=1, sigma=1):
    """Apply a lightweight Gaussian blur in pixel units."""
    kernel = ee.Kernel.gaussian(radius=radius, sigma=sigma, units="pixels", normalize=True)
    return img.convolve(kernel)


def make_z_gt10(country, year, zone):
    """Return HR target DTM downsampled to 10 m."""
    proj = makeUTMProj(zone)
    hr = makeCollection(country, year, zone)
    # Force projection 2m, then blur a 5x5 window, then downsample to 10 m.
    z_gt = hr.mosaic().reproject(proj.atScale(2))
    z_blur = fastGaussianBlur(z_gt, radius=2, sigma=1)
    return z_blur.reduceResolution(ee.Reducer.mean(), False, 256).reproject(proj.atScale(10))


def make_M_bld10(year, zone):
    """Return building mask at 10 m for a year and zone."""
    proj = makeUTMProj(zone).atScale(10)
    date = ee.Date.fromYMD(year, 1, 1)
    wsf_col = ee.ImageCollection("projects/sat-io/open-datasets/WSF/WSF_EVO")
    wsf_year = wsf_col.filterDate(date, date.advance(1, "year"))
    wsf_ref = ee.Image(ee.Algorithms.If(wsf_year.size().gt(0), wsf_year.first(), wsf_col.first()))
    return wsf_col.mosaic().reproject(wsf_ref.projection()).reproject(proj).lte(year).unmask(0)


def make_M_wp10(year, zone):
    """Return persistent water mask at 10 m."""
    proj = makeUTMProj(zone).atScale(10)
    date = ee.Date.fromYMD(year, 1, 1)

    # Primary, GSW.  Mean through months = occurance.
    gsw_col = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory").filterDate(date, date.advance(1, "year"))
    gsw = gsw_col.map(lambda img: img.selfMask().subtract(1).rename("water")).mean()

    # Fallback to GLAD annual, which is percentange, 0-100.
    glad_col = ee.ImageCollection("projects/glad/water/C2/annual").filterDate(date, date.advance(1, "year"))
    glad = glad_col.map(lambda i: i.divide(100).float().rename("water")).first()

    # GSW on top.
    gsw_ref = ee.Image(
        ee.Algorithms.If(gsw_col.size().gt(0), gsw_col.first(), ee.ImageCollection("JRC/GSW1_4/MonthlyHistory").first())
    )
    return ee.ImageCollection([glad, gsw]).mosaic().unmask(0).reproject(gsw_ref.projection()).reproject(proj)


def erode(img, distance):
    """Erode a binary mask by distance in pixels."""
    d = img.Not().unmask(1).fastDistanceTransform(10).sqrt()
    return img.updateMask(d.gt(distance))


def dilate(img, distance):
    """Dilate a binary mask by distance in pixels."""
    d = img.fastDistanceTransform(10).sqrt()
    return d.lt(distance)


def make_M_ws10(year, zone):
    """Return shoreline band mask derived from water edges."""
    p10 = makeUTMProj(zone).atScale(10)

    # Shoreline: 30m buffer on the water band, but not water.
    # Using a 70% threshold to indicate persistent water.
    water = make_M_wp10(year, zone).gt(0.7)
    return dilate(water, 4).And(erode(water, 4).unmask(0).Not()).reproject(p10)


def makeStack(country, year, zone):
    """Build per-pixel feature stack used for patch stats.

    Returns:
        (stack, band_ids): Catenated multi-band image and output band names in order.
    """
    z_gt10 = make_z_gt10(country, year, zone)
    z_lr10 = make_z_lr10(zone)
    u_lr10 = make_u_lr10(zone)
    M_bld10 = make_M_bld10(year, zone)
    M_wp10 = make_M_wp10(year, zone).unmask(0)
    M_ws10 = make_M_ws10(year, zone).unmask(0)

    # Encoded uncertainty.
    u_enc = u_lr10.add(1).log().clamp(0, 1)

    # Overall weight.
    weights = (
        ee.Image(1)
        .multiply(ee.Image(1).subtract(M_bld10))
        .multiply(ee.Image(1).subtract(M_wp10))
        .multiply(ee.Image(1).subtract(M_ws10.multiply(0.8)))
        .multiply(ee.Image(1).subtract(u_enc.multiply(u_enc).multiply(0.5)))
    )

    residAbs = z_gt10.subtract(z_lr10).abs()
    slope = ee.Terrain.slope(z_gt10)

    # Single source of truth for band names and stack order (also used as computePixels bandIds).
    layers = (
        ("z_gt10", z_gt10),
        ("z_gtMask", z_gt10.mask()),
        ("z_lr10", z_lr10),
        ("u_enc", u_enc),
        ("slope", slope),
        ("residAbs", residAbs),
        ("M_bld10", M_bld10),
        ("M_wp10", M_wp10),
        ("M_ws10", M_ws10),
        ("weight", weights),
    )
    band_ids = tuple(name for name, _ in layers)
    stack = ee.Image.cat([img.rename(name) for name, img in layers]).float()
    return stack, band_ids


def grid_128x128_snap10m(x, y, scale_m=10, size=128, coarse_m=1280):
    """Pixel grid for Earth Engine getPixels: *size*×*size* at ``scale_m`` in UTM meters.

    ``x`` and ``y`` are the southwest patch-corner coordinates on the UTM grid with
    spacing ``coarse_m`` (e.g. 1280 m cells). The returned grid is exactly one
    physical coarse cell wide/tall, anchored at ``(x * coarse_m, y * coarse_m)``.

    Returns ``dimensions`` and ``affineTransform`` for the REST ``grid`` object;
    the caller should set ``crsCode`` from the zone’s UTM EPSG code.

    Affine uses scaleY = -scale_m and translateY = top (north edge), matching EE
    getPixels examples.
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


def _utm_crs_code(signed_zone):
    z = int(signed_zone)
    if z < 0:
        return "EPSG:%d" % (32700 + abs(z))
    return "EPSG:%d" % (32600 + abs(z))


def makeAnnualSatelliteEmbeddingByte(year, zone, px, py, coarse_m=1280):
    """Alpha Earth / Satellite Embedding for *year*, 10 m UTM *zone*, byte-encoded on [-1,1] → uint8.

    Uses the patch center on the coarse grid (same as DTM grid) to select the embedding mosaic tile,
    reprojects to ``makeUTMProj(zone).atScale(10)``, then scales with 127.5 * x + 127.5.
    """
    crs = _utm_crs_code(zone)
    proj = ee.Projection(crs)
    center_e = ee.Number(px).multiply(coarse_m)
    center_n = ee.Number(py).multiply(coarse_m)
    pt = ee.Geometry.Point([center_e, center_n], proj)

    start = ee.Date.fromYMD(int(year), 1, 1)
    end = start.advance(1, "year")
    col = (
        ee.ImageCollection(SATELLITE_EMBEDDING_ANNUAL)
        .filterDate(start, end)
        .filterBounds(pt)
    )
    # first() is null when no annual tile matches (ocean-only, etc.); years are filtered in fetch_patch_items.
    # Placeholder: 64 floats (catalog dimensionality), encodes to uint8 128.
    img = ee.Image(
        ee.Algorithms.If(
            col.size().gt(0),
            col.first(),
            ee.Image.constant([0.0] * 64),
        )
    )
    proj10 = makeUTMProj(zone).atScale(10)
    img = img.reproject(proj10, None, 10)
    return img.multiply(127.5).add(127.5).clamp(0, 255).uint8()


def fetch_patch_items(collection_ids: Sequence[str]) -> list[dict]:
    """Download GeoJSON rows from each Earth Engine feature collection ID (minimal properties).

    Results are concatenated in order; each feature's ``properties.path`` is
    ``{collection_id}/{feature_id}``.
    """
    selectors = ["system:index", "x", "y", "zone", "year", "country"]
    items: list[dict] = []
    for collection_path in collection_ids:
        fc = (
            ee.FeatureCollection(collection_path)
            # Only downloading the AU patches now.
            .filter(ee.Filter.equals("country", "AU"))
            .filter(
                ee.Filter.And(
                    ee.Filter.gte("year", SATELLITE_EMBEDDING_MIN_YEAR),
                    ee.Filter.lte("year", SATELLITE_EMBEDDING_MAX_YEAR),
                )
            )
        )
        url = fc.getDownloadURL(filetype="geojson", selectors=selectors)
        with urllib.request.urlopen(url) as response:
            raw = response.read()
        payload = json.loads(raw.decode("utf-8"))
        for feat in payload["features"]:
            fid = collection_path + "/" + str(feat["id"])
            feat["properties"]["path"] = fid
            items.append(feat)
    return items


def export_one_patch(item: dict) -> dict:
    """Run ``computePixels`` for DTM stack + annual embedding; write ZSTD GeoTIFFs.

    Retries and adaptive concurrency live in ``AdaptiveThreadExportRunner`` (see ``run_export``).
    ``ee.EEException`` is re-raised with the patch output id (and EE feature path) in the message.
    """

    props = item["properties"]
    zone = props["zone"]
    country = props["country"]
    year = props["year"]
    px = props["x"]
    py = props["y"]
    out_id = patch_id_from_properties(props)
    ee_path = props.get("path", "")

    try:
        grid = grid_128x128_snap10m(px, py)
        grid["crsCode"] = _utm_crs_code(zone)
        stack, band_ids = makeStack(country, year, zone)
        payload = {
            "expression": stack,
            "fileFormat": "GEO_TIFF",
            "bandIds": list(band_ids),
            "grid": grid,
        }
        aef_image = makeAnnualSatelliteEmbeddingByte(year, zone, px, py)
        aef_payload = {
            "expression": aef_image,
            "fileFormat": "GEO_TIFF",
            "grid": grid,
        }
        out_path = STACK_EXPORT_DIR / f"{out_id}.tif"
        with ThreadPoolExecutor(max_workers=EE_COMPUTEPIXELS_PARALLEL_PER_PATCH) as ex:
            fut_stack = ex.submit(ee.data.computePixels, payload)
            fut_ae = ex.submit(ee.data.computePixels, aef_payload)
            tif_bytes = fut_stack.result()
            aef_bytes = fut_ae.result()

        write_geotiff_zstd(out_path, tif_bytes, predictor=3)

        aef_path = AE_EXPORT_DIR / f"{out_id}_aef_uint8.tif"
        write_geotiff_zstd(aef_path, aef_bytes, predictor=2)
    except ee.EEException as exc:
        prefix = f"[patch {out_id}]"
        if ee_path:
            prefix += f" [ee_feature {ee_path}]"
        raise ee.EEException(f"{prefix} {exc}") from exc

    return {
        "id": out_id,
        "path": str(out_path.resolve()),
        "aef_uint8_path": str(aef_path.resolve()),
    }


def run_export(pool_workers: int | None = None) -> None:
    collection_ids = [
        # "users/ngorelick/DTM/tmp/sample_us_100k",
        "users/ngorelick/DTM/tmp/sample_100k"
    ]
    items = fetch_patch_items(collection_ids)
    completed = completed_patch_ids_on_disk()
    n_all = len(items)
    items = filter_pending_patch_items(items, completed)
    skipped = n_all - len(items)
    total = len(items)
    print(
        f"catalog {n_all} patches, {skipped} already on disk, {total} to export",
        flush=True,
    )
    if not items:
        print("done (nothing to run).", flush=True)
        return
    if pool_workers is None:
        pool_workers = DEFAULT_EXPORT_POOL_WORKERS

    runner = AdaptiveThreadExportRunner(
        min_concurrent=1,
        max_concurrent=pool_workers * 2,
        initial_concurrent=pool_workers,
        quiet_before_scale_up_sec=10.0,
        scale_down_step=1,
        scale_up_step=1,
        max_tries=10,
        retry_base_delay_sec=1.0,
        retry_backoff=2.0,
    )
    progress = ExportProgressLine(rate_window_sec=_PROGRESS_RATE_WINDOW_SEC)

    def on_export_error(exc: Exception) -> None:
        # Newline clears the live ``\r`` progress line; stderr avoids mixing with bar redraws.
        print(f"\n[export skipped] {exc}", file=sys.stderr, flush=True)

    for _ in runner.run_unordered(
        export_one_patch,
        items,
        on_each_done=progress.emit,
        rate_window_sec=_PROGRESS_RATE_WINDOW_SEC,
        continue_on_errors=True,
        on_task_error=on_export_error,
    ):
        pass
    progress.end_line()
    print("done.", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)
    run_export()
