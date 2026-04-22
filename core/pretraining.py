"""Utilities for SSL encoder pretraining and supervised handoff."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from core.checkpoints import load_checkpoint

ENCODER_PREFIXES = (
    "dem_b0.",
    "down_d1.",
    "dem_b1.",
    "down_d2.",
    "dem_b2.",
    "down_d3.",
    "dem_b3.",
    "ae_b0.",
    "down_a1.",
    "ae_b1.",
    "down_a2.",
    "ae_b2.",
    "down_a3.",
    "ae_b3.",
)


def extract_encoder_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    """Extract DEM/AE encoder weights using shared module prefixes."""
    state = model.state_dict()
    return {
        key: value
        for key, value in state.items()
        if any(key.startswith(prefix) for prefix in ENCODER_PREFIXES)
    }


def load_pretrained_encoder(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> tuple[int, int]:
    """Load pretrained encoder weights into model by key+shape match.

    Returns:
        (loaded_key_count, skipped_key_count)
    """
    checkpoint = load_checkpoint(checkpoint_path)
    if "encoder_state" in checkpoint and isinstance(checkpoint["encoder_state"], dict):
        source_state = checkpoint["encoder_state"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        source_state = {
            key: value
            for key, value in checkpoint["model"].items()
            if any(key.startswith(prefix) for prefix in ENCODER_PREFIXES)
        }
    else:
        raise TypeError(f"checkpoint does not contain encoder state: {checkpoint_path}")

    model_state = model.state_dict()
    merged = dict(model_state)
    loaded = 0
    skipped = 0
    for key, value in source_state.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            merged[key] = value
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(merged, strict=False)
    return loaded, skipped

