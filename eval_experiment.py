#!/usr/bin/env python3
"""Experimental evaluation entrypoint (additive, leaves eval_dem unchanged)."""

from __future__ import annotations

import argparse
import logging
import sys

import eval_dem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_experiment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="baseline",
        choices=("baseline",),
        help="Experiment key (baseline currently delegates to eval_dem).",
    )
    args, passthrough = parser.parse_known_args()
    if args.experiment != "baseline":
        raise ValueError(f"unsupported experiment: {args.experiment}")
    log.info("Experiment 'baseline' delegates to legacy eval_dem entrypoint.")
    sys.argv = [sys.argv[0], *passthrough]
    eval_dem.main()


if __name__ == "__main__":
    main()

