"""Helpers for loading and normalizing patch-summary tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"none", "null", "nan"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _coerce_scalar(value) for key, value in row.items()}


def _rows_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [_normalize_row(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object, array, or GeoJSON FeatureCollection")
    if isinstance(payload.get("features"), list):
        rows: list[dict[str, Any]] = []
        for feature in payload["features"]:
            if not isinstance(feature, dict):
                continue
            props = feature.get("properties")
            if isinstance(props, dict):
                rows.append(_normalize_row(props))
        return rows
    if isinstance(payload.get("rows"), list):
        return [_normalize_row(row) for row in payload["rows"] if isinstance(row, dict)]
    return [_normalize_row(payload)]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return [_normalize_row(row) for row in csv.DictReader(fh)]
    if suffix in {".json", ".geojson"}:
        with path.open("r", encoding="utf-8") as fh:
            return _rows_from_json_payload(json.load(fh))
    raise ValueError(f"Unsupported patch table format: {path}")


def _coerce_int(value: Any) -> int:
    if value is None:
        raise ValueError("missing required integer field")
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid integer field")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float is not a valid integer field")
        return int(value)
    return int(float(str(value)))


def stem_from_table_row(row: dict[str, Any]) -> str:
    for key in ("stem", "patch_id", "patchId", "patch"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    system_index = row.get("system:index")
    if system_index not in (None, ""):
        return str(system_index)
    x = _coerce_int(row.get("x"))
    y = _coerce_int(row.get("y"))
    zone = _coerce_int(row.get("zone"))
    country = row.get("country")
    year = _coerce_int(row.get("year"))
    if country in (None, ""):
        raise ValueError("missing country in patch table row")
    return f"{x}_{y}_{zone}_{country}_{year}"


def load_patch_table(path: str | Path, *, allowed_stems: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Load a CSV/JSON/GeoJSON patch table keyed by patch stem."""
    table_path = Path(path)
    rows = _load_rows(table_path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            stem = stem_from_table_row(row)
        except ValueError:
            continue
        if allowed_stems is not None and stem not in allowed_stems:
            continue
        normalized = dict(row)
        normalized["stem"] = stem
        out[stem] = normalized
    return out
