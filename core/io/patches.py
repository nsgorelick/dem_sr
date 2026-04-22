"""Shared patch manifest/listing helpers."""

from __future__ import annotations

from pathlib import Path


def list_patch_stems(training_root: str | Path) -> list[str]:
    from local_patch_dataset import list_patch_stems as _list_patch_stems

    return _list_patch_stems(training_root)


def load_patch_stems_manifest(path: Path | str) -> list[str]:
    from local_patch_dataset import load_patch_stems_manifest as _load_patch_stems_manifest

    return _load_patch_stems_manifest(path)

