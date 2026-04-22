"""Shared run-config loading and standardized output naming."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_run_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"run config must be a JSON object: {path}")
    return payload


def section_defaults(config: dict[str, Any], section: str) -> dict[str, Any]:
    shared = config.get("shared", {})
    specific = config.get(section, {})
    out: dict[str, Any] = {}
    if isinstance(shared, dict):
        out.update(shared)
    if isinstance(specific, dict):
        out.update(specific)
    return out


def resolve_description(config: dict[str, Any], config_path: Path, cli_description: str | None) -> str:
    if cli_description:
        return cli_description.strip()
    raw = config.get("description")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return config_path.stem


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "run"


def standardized_eval_output_path(*, config_path: Path | None, description: str) -> Path:
    if config_path is None:
        base = Path("runs/results")
        stem = "eval_results"
    else:
        base = config_path.parent / "results"
        stem = f"{config_path.stem}_eval_results"
    return base / f"{stem}_{_slugify(description)}.json"

