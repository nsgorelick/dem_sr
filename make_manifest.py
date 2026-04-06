#!/usr/bin/env python3
"""Create deterministic train/holdout manifests from a local training root."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

from local_patch_dataset import list_patch_stems

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("make_manifest")


def write_manifest(path: Path, stems: list[str]) -> None:
    text = "".join(f"{stem}\n" for stem in stems)
    path.write_text(text, encoding="utf-8")


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
        help="Fraction of stems assigned to the holdout manifest",
    )
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    p.add_argument(
        "--holdout-out",
        type=Path,
        default=Path("holdout_manifest.txt"),
        help="Output file for holdout stems",
    )
    p.add_argument(
        "--train-out",
        type=Path,
        default=Path("train_manifest.txt"),
        help="Optional output file for remaining train stems",
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
    args = p.parse_args()

    frac = max(0.0, min(0.5, args.holdout_fraction))
    stems = list_patch_stems(args.data_root)
    if not stems:
        log.error("No patch stems found under %s", args.data_root)
        sys.exit(1)

    rng = random.Random(args.seed)
    stems = list(stems)
    rng.shuffle(stems)

    n_total = len(stems)
    n_holdout = max(1, int(round(n_total * frac)))
    n_holdout = min(n_holdout, n_total - 1) if n_total > 1 else 1

    holdout = sorted(stems[:n_holdout])
    train = sorted(stems[n_holdout:])
    full_summary = summarize_stems(stems)
    holdout_summary = summarize_stems(holdout)
    train_summary = summarize_stems(train)

    write_manifest(args.holdout_out, holdout)
    log.info("Wrote %d holdout stems to %s", len(holdout), args.holdout_out)
    log_summary("all", full_summary)
    log_summary("holdout", holdout_summary)

    if not args.no_train_manifest:
        write_manifest(args.train_out, train)
        log.info("Wrote %d train stems to %s", len(train), args.train_out)
        log_summary("train", train_summary)

    if args.summary_json is not None:
        payload = {
            "data_root": args.data_root,
            "seed": args.seed,
            "holdout_fraction": frac,
            "all": full_summary,
            "holdout": holdout_summary,
            "train": train_summary,
        }
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Wrote summary JSON to %s", args.summary_json)


if __name__ == "__main__":
    main()
