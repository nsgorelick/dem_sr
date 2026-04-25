#!/usr/bin/env python3
"""Build normalized stratified train/val draws from patch-table GeoJSONs.

Outputs:
- training GeoJSON (US + non-US)
- validation GeoJSON (AU)
- combined manifest for export (all IDs)
- per-pool normalization summaries
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BIN_LABELS = ("A", "B", "C", "D", "E")
BIN_TARGET_FRAC = {"A": 0.20, "B": 0.25, "C": 0.25, "D": 0.20, "E": 0.10}
SLOPE_EDGES = (2.0, 5.0, 10.0, 20.0)
PREFIX_MAP = {
    "de": "DE",
    "fr": "FR",
    "nordic": "NORDIC",
    "pt": "PT",
    "us": "US",
    "at": "AT",
    "jp": "JP",
    "au": "AU",
    "es": "ES",
}


@dataclass(frozen=True)
class DrawRecord:
    props: dict[str, Any]
    geometry: dict[str, Any] | None
    country: str
    slope_bin: str
    score: float
    stem: str


def _log(msg: str) -> None:
    print(msg, flush=True)


def _get_num(p: dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        v = p.get(k)
        if v is None or isinstance(v, bool):
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            return x
    return None


def _slope_bin(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return "missing"
    if v <= SLOPE_EDGES[0]:
        return "A"
    if v <= SLOPE_EDGES[1]:
        return "B"
    if v <= SLOPE_EDGES[2]:
        return "C"
    if v <= SLOPE_EDGES[3]:
        return "D"
    return "E"


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = p * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    t = pos - lo
    return sorted_vals[lo] * (1.0 - t) + sorted_vals[hi] * t


def _norm_clip(x: float | None, lo: float, hi: float) -> float:
    if x is None or not math.isfinite(x):
        return 0.0
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return 0.0
    t = (float(x) - lo) / (hi - lo)
    return max(0.0, min(1.0, t))


def _country_from_table_stem(table_stem: str) -> str:
    m = re.match(r"^(?P<prefix>[a-z]+)_sites(?:\d+)?_", table_stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"unexpected table stem: {table_stem!r}")
    pre = str(m.group("prefix")).lower()
    return PREFIX_MAP.get(pre, pre.upper())


def _passes_hard_filters(p: dict[str, Any]) -> bool:
    mean_w = _get_num(p, "mean_W", "weight_mean")
    valid_frac = _get_num(p, "valid_frac", "weight_valid_mean")
    frac_water = _get_num(p, "frac_water", "water_mean")
    relief = _get_num(p, "relief")
    if mean_w is None or mean_w < 0.3:
        return False
    if valid_frac is None or valid_frac < 0.7:
        return False
    if frac_water is None:
        frac_water = 0.0
    if frac_water > 0.5:
        return False
    if relief is None or relief < 0.5:
        return False
    return True


def _stem_from_props(p: dict[str, Any], country: str) -> str:
    return (
        f"{int(float(p['x']))}_{int(float(p['y']))}_{int(float(p['zone']))}"
        f"_{country}_{int(float(p['year']))}"
    )


def _compute_quotas(total: int) -> dict[str, int]:
    exact = [total * BIN_TARGET_FRAC[b] for b in BIN_LABELS]
    base = [int(math.floor(x)) for x in exact]
    rem = total - sum(base)
    frac = sorted(((exact[i] - base[i], i) for i in range(len(BIN_LABELS))), reverse=True)
    for i in range(rem):
        base[frac[i][1]] += 1
    return {b: base[i] for i, b in enumerate(BIN_LABELS)}


def _draw_within_bin(records: list[DrawRecord], target: int, rng: random.Random) -> list[DrawRecord]:
    if target <= 0 or not records:
        return []
    groups: dict[tuple[int, int], list[DrawRecord]] = defaultdict(list)
    for rec in records:
        z = int(float(rec.props["zone"]))
        y = int(float(rec.props["year"]))
        groups[(z, y)].append(rec)

    picked: list[DrawRecord] = []
    while len(picked) < target:
        keys = [k for k, v in groups.items() if v]
        if not keys:
            break
        w_groups = [1.0 / max(1, len(groups[k])) for k in keys]
        gk = rng.choices(keys, weights=w_groups, k=1)[0]
        chunk = groups[gk]
        w_patch = [0.05 + math.sqrt(max(0.0, r.score)) for r in chunk]
        j = rng.choices(range(len(chunk)), weights=w_patch, k=1)[0]
        chosen = chunk.pop(j)
        picked.append(chosen)
        if not chunk:
            del groups[gk]
    return picked


def _score_and_dedupe(records: list[dict[str, Any]], *, tau_shore: float, tau_bld: float) -> tuple[list[DrawRecord], dict[str, Any]]:
    slope_vals = sorted(
        float(_get_num(r["props"], "p90_slope", "slope_p90"))
        for r in records
        if _get_num(r["props"], "p90_slope", "slope_p90") is not None
    )
    resid_vals = sorted(
        float(_get_num(r["props"], "resid_scale", "residAbs_p95"))
        for r in records
        if _get_num(r["props"], "resid_scale", "residAbs_p95") is not None
    )
    unc_vals = sorted(
        float(_get_num(r["props"], "mean_uncert", "U_lr10_mean"))
        for r in records
        if _get_num(r["props"], "mean_uncert", "U_lr10_mean") is not None
    )
    rel_vals = sorted(
        float(_get_num(r["props"], "relief"))
        for r in records
        if _get_num(r["props"], "relief") is not None
    )

    s_lo, s_hi = _percentile(slope_vals, 0.05), _percentile(slope_vals, 0.95)
    r_lo, r_hi = _percentile(resid_vals, 0.05), _percentile(resid_vals, 0.95)
    u_lo, u_hi = _percentile(unc_vals, 0.05), _percentile(unc_vals, 0.95)
    h_lo, h_hi = _percentile(rel_vals, 0.05), _percentile(rel_vals, 0.95)

    by_stem: dict[str, DrawRecord] = {}
    for row in records:
        p = row["props"]
        country = row["country"]
        s = _norm_clip(_get_num(p, "p90_slope", "slope_p90"), s_lo, s_hi)
        rv = _norm_clip(_get_num(p, "resid_scale", "residAbs_p95"), r_lo, r_hi)
        u = _norm_clip(_get_num(p, "mean_uncert", "U_lr10_mean"), u_lo, u_hi)
        h = _norm_clip(_get_num(p, "relief"), h_lo, h_hi)
        fs = _get_num(p, "frac_shore", "shore_mean") or 0.0
        fb = _get_num(p, "frac_building", "bld_mean") or 0.0
        e = 1.0 if (fs > tau_shore or fb > tau_bld) else 0.0
        score = 0.40 * s + 0.30 * rv + 0.15 * u + 0.10 * h + 0.05 * e
        stem = _stem_from_props(p, country)
        rec = DrawRecord(
            props=p,
            geometry=row["geometry"],
            country=country,
            slope_bin=row["slope_bin"],
            score=float(score),
            stem=stem,
        )
        prev = by_stem.get(stem)
        if prev is None or rec.score > prev.score:
            by_stem[stem] = rec

    stats = {
        "n_input_records": len(records),
        "n_unique_stems": len(by_stem),
        "p05_p95": {
            "slope": [s_lo, s_hi],
            "resid": [r_lo, r_hi],
            "unc": [u_lo, u_hi],
            "relief": [h_lo, h_hi],
        },
    }
    return list(by_stem.values()), stats


def _draw(records: list[DrawRecord], *, total: int, seed: int) -> tuple[list[DrawRecord], dict[str, int], dict[str, int]]:
    quotas = _compute_quotas(total)
    by_bin: dict[str, list[DrawRecord]] = defaultdict(list)
    for r in records:
        by_bin[r.slope_bin].append(r)

    rng = random.Random(seed)
    for v in by_bin.values():
        rng.shuffle(v)

    selected: list[DrawRecord] = []
    for b in BIN_LABELS:
        picked = _draw_within_bin(by_bin.get(b, []), quotas[b], rng)
        selected.extend(picked)

    # Safety dedupe.
    out: list[DrawRecord] = []
    seen: set[str] = set()
    for r in selected:
        if r.stem in seen:
            continue
        seen.add(r.stem)
        out.append(r)

    if len(out) != total:
        raise SystemExit(f"Requested {total}, drew {len(out)} (shortfall {total-len(out)})")
    counts = Counter(r.slope_bin for r in out)
    return out, quotas, dict(counts)


def _to_feature(rec: DrawRecord) -> dict[str, Any]:
    p = dict(rec.props)
    p.setdefault("country", rec.country)
    p["stratum_slope_bin"] = rec.slope_bin
    p["stratified_score"] = rec.score
    feat: dict[str, Any] = {"type": "Feature", "properties": p}
    if isinstance(rec.geometry, dict):
        feat["geometry"] = rec.geometry
    return feat


def build(args: argparse.Namespace) -> None:
    patch_dir = args.patch_dir.resolve()
    files = sorted(patch_dir.glob(args.glob))
    _log(f"[1/6] scanning files in {patch_dir} with glob {args.glob!r}: {len(files)} files")

    pools: dict[str, list[dict[str, Any]]] = {"US": [], "NON_US": [], "AU": []}
    total_seen = 0
    total_kept = 0
    for idx, fp in enumerate(files, start=1):
        payload = json.loads(fp.read_text(encoding="utf-8"))
        rows = payload.get("features") or []
        for feat in rows:
            total_seen += 1
            p = feat.get("properties")
            if not isinstance(p, dict):
                continue
            year = _get_num(p, "year")
            if year is None or int(year) < args.min_year:
                continue
            if not _passes_hard_filters(p):
                continue
            sb = _slope_bin(_get_num(p, "p90_slope", "slope_p90"))
            if sb == "missing":
                continue
            c = p.get("country")
            country = (
                str(c).strip()
                if c is not None and str(c).strip()
                else _country_from_table_stem(fp.stem)
            )
            row = {"props": dict(p), "geometry": feat.get("geometry"), "country": country, "slope_bin": sb}
            if country == "US":
                pools["US"].append(row)
            elif country == "AU":
                pools["AU"].append(row)
            else:
                pools["NON_US"].append(row)
            total_kept += 1
        if idx % args.log_every_files == 0 or idx == len(files):
            _log(f"  processed {idx}/{len(files)} files; seen={total_seen} kept_after_hard={total_kept}")

    _log(
        "[2/6] pool sizes after hard filters: "
        f"US={len(pools['US'])} NON_US={len(pools['NON_US'])} AU={len(pools['AU'])}"
    )

    scored: dict[str, list[DrawRecord]] = {}
    norm_stats: dict[str, Any] = {}
    for key in ("US", "NON_US", "AU"):
        _log(f"[3/6] scoring + deduping {key} pool")
        scored[key], norm_stats[key] = _score_and_dedupe(
            pools[key], tau_shore=args.tau_shore, tau_bld=args.tau_bld
        )
        _log(
            f"  {key}: input={norm_stats[key]['n_input_records']} "
            f"unique_stems={norm_stats[key]['n_unique_stems']}"
        )

    _log("[4/6] drawing stratified samples")
    us_draw, us_q, us_counts = _draw(scored["US"], total=args.n_us_train, seed=args.seed)
    non_us_draw, nu_q, nu_counts = _draw(scored["NON_US"], total=args.n_non_us_train, seed=args.seed)
    au_draw, au_q, au_counts = _draw(scored["AU"], total=args.n_au_val, seed=args.seed)
    _log(f"  US counts={us_counts} quotas={us_q}")
    _log(f"  NON_US counts={nu_counts} quotas={nu_q}")
    _log(f"  AU counts={au_counts} quotas={au_q}")

    _log("[5/6] writing normalization summaries")
    (patch_dir / args.norm_us).write_text(json.dumps(norm_stats["US"], indent=2), encoding="utf-8")
    (patch_dir / args.norm_non_us).write_text(
        json.dumps(norm_stats["NON_US"], indent=2), encoding="utf-8"
    )
    (patch_dir / args.norm_au).write_text(json.dumps(norm_stats["AU"], indent=2), encoding="utf-8")

    _log("[6/6] writing train/val geojson + manifests")
    training = us_draw + non_us_draw
    validation = au_draw
    training_fc = {"type": "FeatureCollection", "features": [_to_feature(r) for r in training]}
    validation_fc = {"type": "FeatureCollection", "features": [_to_feature(r) for r in validation]}
    (patch_dir / args.training_geojson).write_text(json.dumps(training_fc, indent=2), encoding="utf-8")
    (patch_dir / args.validation_geojson).write_text(json.dumps(validation_fc, indent=2), encoding="utf-8")

    train_ids = [r.stem for r in training]
    val_ids = [r.stem for r in validation]
    all_ids = train_ids + val_ids
    (patch_dir / args.training_manifest).write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (patch_dir / args.validation_manifest).write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    (patch_dir / args.export_manifest).write_text("\n".join(all_ids) + "\n", encoding="utf-8")

    _log(f"done: training={len(train_ids)} validation={len(val_ids)} total_manifest={len(all_ids)}")
    _log(f"wrote {(patch_dir / args.training_geojson)}")
    _log(f"wrote {(patch_dir / args.validation_geojson)}")
    _log(f"wrote {(patch_dir / args.export_manifest)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patch-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument(
        "--glob",
        default="*_sites*_????_*.geojson",
        help="Patch-table file glob within --patch-dir",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-year", type=int, default=2017, help="Exclude patches with year < min-year")
    p.add_argument("--tau-shore", type=float, default=0.02)
    p.add_argument("--tau-bld", type=float, default=0.02)
    p.add_argument("--n-us-train", type=int, default=100_000)
    p.add_argument("--n-non-us-train", type=int, default=100_000)
    p.add_argument("--n-au-val", type=int, default=50_000)
    p.add_argument("--training-geojson", default="training.geojson")
    p.add_argument("--validation-geojson", default="validation.geojson")
    p.add_argument("--training-manifest", default="training_manifest.txt")
    p.add_argument("--validation-manifest", default="validation_manifest.txt")
    p.add_argument("--export-manifest", default="export_manifest_250k.txt")
    p.add_argument("--norm-us", default="patch_table_normalization_stats_us.json")
    p.add_argument("--norm-non-us", default="patch_table_normalization_stats_non_us.json")
    p.add_argument("--norm-au", default="patch_table_normalization_stats_au.json")
    p.add_argument("--log-every-files", type=int, default=10)
    return p


if __name__ == "__main__":
    build(build_parser().parse_args())

