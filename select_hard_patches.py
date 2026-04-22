#!/usr/bin/env python3
"""Subsample a training manifest to the hardest K% of patches.

"Hard" is scored by the patch-summary residual magnitude (``residAbs_p95`` /
``resid_scale``) by default: the larger this is, the more the bicubic baseline
``z_lr`` disagrees with the 10 m ground-truth DTM, and therefore the more the
model has room to help. Low-quality patches (dominated by water, poor GT
coverage, thin weights) are filtered out before ranking so the "hard" pool is
still trainable.

Example::

    python3 select_hard_patches.py \\
        --manifest runs/.../manifests/train_non_au_manifest.txt \\
        --patch-table 200k.geojson \\
        --fraction 0.10 \\
        --out runs/.../manifests/train_non_au_hard_p90_manifest.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

from make_manifest import get_numeric, row_passes_eval_filters
from core.patch_table import load_patch_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("select_hard_patches")


SCORE_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "resid_scale": ("resid_scale", "residAbs_p95"),
    "p90_slope": ("p90_slope", "slope_p90"),
    "relief": ("relief",),
    "mean_uncert": ("mean_uncert", "U_lr10_mean"),
}


def load_manifest(path: Path) -> list[str]:
    stems: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            stems.append(s)
    return stems


def score_row(row: dict[str, object], field: str) -> float | None:
    aliases = SCORE_FIELD_ALIASES.get(field, (field,))
    value = get_numeric(row, *aliases)
    if value is None or not math.isfinite(value):
        return None
    return value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Input training manifest")
    p.add_argument(
        "--patch-table",
        type=Path,
        required=True,
        help="Patch summary CSV/JSON/GeoJSON keyed by patch stem",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="Keep this top-fraction of eligible stems (0 < fraction <= 1)",
    )
    p.add_argument(
        "--score-field",
        default="resid_scale",
        help="Patch-table field to rank by (alias-aware): resid_scale | p90_slope | relief | mean_uncert",
    )
    p.add_argument("--out", type=Path, required=True, help="Output manifest path")
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON with selection summary (sizes, thresholds, score quantiles)",
    )
    p.add_argument("--min-mean-w", type=float, default=0.4)
    p.add_argument("--min-valid-frac", type=float, default=0.8)
    p.add_argument("--min-gt-coverage", type=float, default=0.8)
    p.add_argument("--min-relief", type=float, default=0.5)
    p.add_argument("--max-frac-water", type=float, default=0.25)
    p.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Optional cap on how many stems to keep after ranking (top scores first); "
        "use with --fraction 1.0 to take the N hardest eligible patches",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=64,
        help="Minimum number of selected stems; exit if eligible pool is smaller",
    )
    p.add_argument(
        "--tie-break-seed",
        type=int,
        default=42,
        help="Seed for deterministic tie-breaking of equal-score rows",
    )
    args = p.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        log.error("--fraction must be in (0, 1], got %.6f", args.fraction)
        sys.exit(1)

    manifest_stems = load_manifest(args.manifest)
    if not manifest_stems:
        log.error("Empty manifest: %s", args.manifest)
        sys.exit(1)
    log.info("Loaded %d stems from %s", len(manifest_stems), args.manifest)

    table = load_patch_table(args.patch_table, allowed_stems=set(manifest_stems))
    log.info("Patch table matched %d/%d stems", len(table), len(manifest_stems))

    rng = random.Random(args.tie_break_seed)
    eligible: list[tuple[float, float, str]] = []
    missing_row = 0
    failed_quality = 0
    missing_score = 0
    for stem in manifest_stems:
        row = table.get(stem)
        if row is None:
            missing_row += 1
            continue
        if not row_passes_eval_filters(
            row,
            min_mean_w=args.min_mean_w,
            min_valid_frac=args.min_valid_frac,
            min_gt_coverage=args.min_gt_coverage,
            min_relief=args.min_relief,
            max_frac_water=args.max_frac_water,
        ):
            failed_quality += 1
            continue
        score = score_row(row, args.score_field)
        if score is None:
            missing_score += 1
            continue
        eligible.append((score, rng.random(), stem))

    eligible_count = len(eligible)
    log.info(
        "Eligibility: %d eligible / %d missing_patch_table_row / %d failed_quality / %d missing_score",
        eligible_count,
        missing_row,
        failed_quality,
        missing_score,
    )
    if eligible_count < args.min_count:
        log.error(
            "Only %d eligible stems (< --min-count=%d). Loosen filters or pick a different score-field.",
            eligible_count,
            args.min_count,
        )
        sys.exit(1)

    eligible.sort(key=lambda t: (-t[0], t[1]))
    keep = max(args.min_count, int(round(eligible_count * args.fraction)))
    keep = min(keep, eligible_count)
    if args.max_count is not None:
        keep = min(keep, max(0, int(args.max_count)))
    selected = eligible[:keep]
    cutoff_score = selected[-1][0] if selected else float("nan")
    top_score = selected[0][0] if selected else float("nan")
    median_score = selected[len(selected) // 2][0] if selected else float("nan")

    selected_stems = sorted(stem for _, _, stem in selected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(selected_stems) + "\n", encoding="utf-8")
    log.info(
        "Wrote %d stems to %s (score=%s: top=%.6g median=%.6g cutoff=%.6g)",
        len(selected_stems),
        args.out,
        args.score_field,
        top_score,
        median_score,
        cutoff_score,
    )

    if args.summary_json is not None:
        payload = {
            "manifest_in": str(args.manifest),
            "patch_table": str(args.patch_table),
            "score_field": args.score_field,
            "fraction": args.fraction,
            "max_count": args.max_count,
            "input_stems": len(manifest_stems),
            "matched_table_rows": len(table),
            "missing_table_rows": missing_row,
            "failed_quality_filter": failed_quality,
            "missing_score": missing_score,
            "eligible_stems": eligible_count,
            "selected_stems": len(selected_stems),
            "quality_thresholds": {
                "min_mean_w": args.min_mean_w,
                "min_valid_frac": args.min_valid_frac,
                "min_gt_coverage": args.min_gt_coverage,
                "min_relief": args.min_relief,
                "max_frac_water": args.max_frac_water,
            },
            "score_top": top_score,
            "score_median": median_score,
            "score_cutoff": cutoff_score,
            "out": str(args.out),
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Wrote %s", args.summary_json)


if __name__ == "__main__":
    main()
