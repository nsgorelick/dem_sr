"""Normalize optional path fields on argparse namespaces."""

from __future__ import annotations

import argparse
from pathlib import Path


def as_optional_path(value: str | Path | None) -> Path | None:
    if value is None or isinstance(value, Path):
        return value
    return Path(value)


def set_attr_path(args: argparse.Namespace, name: str) -> None:
    setattr(args, name, as_optional_path(getattr(args, name, None)))
