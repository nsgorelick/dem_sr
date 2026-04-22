#!/usr/bin/env python3
"""Create manifests from local stems or a patch-summary table."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

from local_patch_dataset import list_patch_stems
from core.patch_table import load_patch_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("make_manifest")


def write_manifest(path: Path, stems: list[str]) -> None:
    text = "".join(f"{stem}\n" for stem in stems)
    path.write_text(text, encoding="utf-8")


def load_exclusion_patterns(path: Path) -> list[tuple[str, ...]]:
    patterns: list[tuple[str, ...]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = tuple(part for part in line.split("_") if part)
        if parts:
            patterns.append(parts)
    return patterns


def stem_matches_pattern(
    stem: str,
    pattern: tuple[str, ...],
    *,
    locked_country: str | None = None,
    default_zone: int | None = None,
) -> bool:
    stem_parts = stem.split("_")
    if len(pattern) == 2 and locked_country is not None and default_zone is not None:
        if len(stem_parts) != 5:
            return False
        x, y, zone, country, _year = stem_parts
        return (
            x == pattern[0]
            and y == pattern[1]
            and zone == str(default_zone)
            and country.upper() == locked_country.upper()
        )
    if len(pattern) > len(stem_parts):
        return False
    return tuple(stem_parts[: len(pattern)]) == pattern


def filter_excluded_stems(
    stems: list[str],
    patterns: list[tuple[str, ...]],
    *,
    locked_country: str | None = None,
    default_zone: int | None = None,
) -> tuple[list[str], list[str]]:
    if not patterns:
        return sorted(stems), []
    kept: list[str] = []
    excluded: list[str] = []
    for stem in stems:
        if any(
            stem_matches_pattern(
                stem,
                pattern,
                locked_country=locked_country,
                default_zone=default_zone,
            )
            for pattern in patterns
        ):
            excluded.append(stem)
        else:
            kept.append(stem)
    return sorted(kept), sorted(excluded)


def parse_patch_stem(stem: str) -> dict[str, str | int]:
    """Parse ``x_y_zone_country_year`` stem metadata."""
    parts = stem.split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected patch stem format: {stem!r}")
    x, y, zone, country, year = parts
    return {
        "x": int(x),
        "y": int(y),
        "zone": int(zone),
        "country": country,
        "year": int(year),
    }


def get_numeric(row: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def row_passes_eval_filters(
    row: dict[str, object],
    *,
    min_mean_w: float,
    min_valid_frac: float,
    min_gt_coverage: float,
    min_relief: float,
    max_frac_water: float,
) -> bool:
    mean_w = get_numeric(row, "mean_W", "weight_mean")
    valid_frac = get_numeric(row, "valid_frac", "weight_valid_mean")
    gt_coverage = get_numeric(row, "gt_coverage_mean")
    relief = get_numeric(row, "relief")
    frac_water = get_numeric(row, "frac_water", "water_mean")
    if mean_w is None or mean_w < min_mean_w:
        return False
    if valid_frac is None or valid_frac < min_valid_frac:
        return False
    if gt_coverage is None or gt_coverage < min_gt_coverage:
        return False
    if relief is None or relief < min_relief:
        return False
    if frac_water is None:
        frac_water = 0.0
    return frac_water <= max_frac_water


def split_stems_randomly(stems: list[str], *, fraction: float, seed: int) -> tuple[list[str], list[str]]:
    frac = max(0.0, min(0.5, fraction))
    ordered = list(stems)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    if frac <= 0 or len(ordered) < 2:
        return sorted(ordered), []
    n_val = max(1, int(round(len(ordered) * frac)))
    n_val = min(n_val, len(ordered) - 1)
    val = sorted(ordered[:n_val])
    train = sorted(ordered[n_val:])
    return train, val


def stems_for_country(stems: list[str], country: str) -> list[str]:
    target = country.strip().upper()
    out: list[str] = []
    for stem in stems:
        try:
            meta = parse_patch_stem(stem)
        except ValueError:
            continue
        if str(meta["country"]).upper() == target:
            out.append(stem)
    return out


def summarize_stems(stems: list[str]) -> dict[str, object]:
    countries: Counter[str] = Counter()
    years: Counter[int] = Counter()
    zones: Counter[int] = Counter()
    invalid = 0

    for stem in stems:
        try:
            meta = parse_patch_stem(stem)
        except ValueError:
            invalid += 1
            continue
        countries[str(meta["country"])] += 1
        years[int(meta["year"])] += 1
        zones[int(meta["zone"])] += 1

    return {
        "count": len(stems),
        "invalid_stems": invalid,
        "countries": dict(sorted(countries.items())),
        "years": dict(sorted(years.items())),
        "zones": dict(sorted(zones.items())),
    }


def log_summary(label: str, summary: dict[str, object]) -> None:
    log.info("%s count: %d", label, int(summary["count"]))
    if int(summary["invalid_stems"]):
        log.warning("%s invalid stems: %d", label, int(summary["invalid_stems"]))
    log.info("%s countries: %s", label, summary["countries"])
    log.info("%s years: %s", label, summary["years"])
    log.info("%s zones: %s", label, summary["zones"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        default="/data/training",
        help="Parent root with stack/ and ae/ subdirs",
    )
    p.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.1,
        help="Fraction of stems assigned to the holdout manifest in random-split mode",
    )
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Optional validation fraction for the development pool in table-driven mode",
    )
    p.add_argument(
        "--holdout-out",
        type=Path,
        default=Path("holdout_manifest.txt"),
        help="Output file for holdout stems (or locked final-test stems in table-driven mode)",
    )
    p.add_argument(
        "--train-out",
        type=Path,
        default=Path("train_manifest.txt"),
        help="Output file for training stems",
    )
    p.add_argument(
        "--val-out",
        type=Path,
        default=None,
        help="Optional output file for validation stems in table-driven mode",
    )
    p.add_argument(
        "--no-train-manifest",
        action="store_true",
        help="Do not write the complementary training manifest",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON file with full/train/holdout summary counts",
    )
    p.add_argument(
        "--patch-table",
        type=Path,
        default=None,
        help="Optional CSV/JSON/GeoJSON patch-summary table for table-driven manifest generation",
    )
    p.add_argument(
        "--locked-country",
        default=None,
        help="Country code reserved as a locked final test set when using --patch-table",
    )
    p.add_argument("--min-mean-w", type=float, default=0.3, help="Minimum acceptable mean_W / weight_mean")
    p.add_argument("--min-valid-frac", type=float, default=0.7, help="Minimum acceptable valid_frac")
    p.add_argument(
        "--min-gt-coverage",
        type=float,
        default=0.95,
        help="Minimum acceptable gt_coverage_mean for locked final-test patches",
    )
    p.add_argument("--min-relief", type=float, default=0.5, help="Minimum acceptable relief")
    p.add_argument("--max-frac-water", type=float, default=0.5, help="Maximum acceptable frac_water")
    p.add_argument(
        "--exclude-list",
        type=Path,
        default=None,
        help="Optional newline-delimited full stems or stem prefixes to exclude from manifests",
    )
    args = p.parse_args()

    stems = list_patch_stems(args.data_root)
    if not stems:
        log.error("No patch stems found under %s", args.data_root)
        sys.exit(1)

    stems = sorted(stems)
    exclusion_patterns = load_exclusion_patterns(args.exclude_list) if args.exclude_list else []
    full_summary = summarize_stems(stems)
    log_summary("all", full_summary)

    if args.patch_table is not None and args.locked_country:
        stem_set = set(stems)
        table = load_patch_table(args.patch_table, allowed_stems=stem_set)
        locked_country = args.locked_country.strip().upper()
        locked_all_unfiltered = sorted(stems_for_country(stems, locked_country))
        locked_all, excluded_locked = filter_excluded_stems(
            locked_all_unfiltered,
            exclusion_patterns,
            locked_country=locked_country,
            default_zone=-55,
        )
        locked_all_unfiltered_set = set(locked_all_unfiltered)
        dev_pool = sorted(stem for stem in stems if stem not in locked_all_unfiltered_set)
        locked_missing_table = sorted(stem for stem in locked_all if stem not in table)
        holdout = sorted(
            stem
            for stem in locked_all
            if stem in table
            and row_passes_eval_filters(
                table[stem],
                min_mean_w=args.min_mean_w,
                min_valid_frac=args.min_valid_frac,
                min_gt_coverage=args.min_gt_coverage,
                min_relief=args.min_relief,
                max_frac_water=args.max_frac_water,
            )
        )
        train, val = split_stems_randomly(dev_pool, fraction=args.val_fraction, seed=args.seed)
        holdout_summary = summarize_stems(holdout)
        train_summary = summarize_stems(train)
        val_summary = summarize_stems(val)
        dev_summary = summarize_stems(dev_pool)
        locked_all_summary = summarize_stems(locked_all_unfiltered)
        locked_eligible_summary = summarize_stems(locked_all)
        excluded_locked_summary = summarize_stems(excluded_locked)

        write_manifest(args.holdout_out, holdout)
        log.info(
            "Wrote %d locked final-test stems for %s to %s",
            len(holdout),
            locked_country,
            args.holdout_out,
        )
        log_summary("locked_country_all", locked_all_summary)
        if excluded_locked:
            log.info("Excluded %d locked-country stems via %s", len(excluded_locked), args.exclude_list)
            log_summary("locked_country_excluded", excluded_locked_summary)
        log_summary("locked_country_eligible", locked_eligible_summary)
        log_summary("locked_final_test", holdout_summary)
        log_summary("development_pool", dev_summary)

        if not args.no_train_manifest:
            write_manifest(args.train_out, train)
            log.info("Wrote %d development-train stems to %s", len(train), args.train_out)
            log_summary("train", train_summary)
        if args.val_out is not None:
            write_manifest(args.val_out, val)
            log.info("Wrote %d development-val stems to %s", len(val), args.val_out)
            log_summary("val", val_summary)

        if args.summary_json is not None:
            payload = {
                "mode": "locked_country_table",
                "data_root": args.data_root,
                "patch_table": str(args.patch_table),
                "locked_country": locked_country,
                "seed": args.seed,
                "val_fraction": max(0.0, min(0.5, args.val_fraction)),
                "exclude_list": str(args.exclude_list) if args.exclude_list else None,
                "filters": {
                    "min_mean_w": args.min_mean_w,
                    "min_valid_frac": args.min_valid_frac,
                    "min_gt_coverage": args.min_gt_coverage,
                    "min_relief": args.min_relief,
                    "max_frac_water": args.max_frac_water,
                },
                "all": full_summary,
                "locked_country_all": locked_all_summary,
                "locked_country_excluded": excluded_locked_summary,
                "locked_country_eligible": locked_eligible_summary,
                "locked_final_test": holdout_summary,
                "locked_country_missing_table_rows": len(locked_missing_table),
                "development_pool": dev_summary,
                "train": train_summary,
                "val": val_summary,
            }
            args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.info("Wrote summary JSON to %s", args.summary_json)
        return

    stems, excluded_stems = filter_excluded_stems(stems, exclusion_patterns)
    frac = max(0.0, min(0.5, args.holdout_fraction))
    train, holdout = split_stems_randomly(stems, fraction=frac, seed=args.seed)
    holdout_summary = summarize_stems(holdout)
    train_summary = summarize_stems(train)
    excluded_summary = summarize_stems(excluded_stems)

    write_manifest(args.holdout_out, holdout)
    log.info("Wrote %d holdout stems to %s", len(holdout), args.holdout_out)
    log_summary("holdout", holdout_summary)
    if excluded_stems:
        log.info("Excluded %d stems via %s", len(excluded_stems), args.exclude_list)
        log_summary("excluded", excluded_summary)

    if not args.no_train_manifest:
        write_manifest(args.train_out, train)
        log.info("Wrote %d train stems to %s", len(train), args.train_out)
        log_summary("train", train_summary)

    if args.summary_json is not None:
        payload = {
            "mode": "random_split",
            "data_root": args.data_root,
            "seed": args.seed,
            "holdout_fraction": frac,
            "exclude_list": str(args.exclude_list) if args.exclude_list else None,
            "all": full_summary,
            "excluded": excluded_summary,
            "holdout": holdout_summary,
            "train": train_summary,
        }
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Wrote summary JSON to %s", args.summary_json)


if __name__ == "__main__":
    main()
