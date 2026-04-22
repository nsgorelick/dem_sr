"""Checkpoint helpers shared by new training/eval entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def make_training_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    data_root: str,
    epoch: int,
    args: dict[str, Any],
    history: dict[str, list[float | None]],
    train_size: int,
    val_size: int,
) -> dict[str, Any]:
    """Build the canonical training-checkpoint payload."""
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "data_root": data_root,
        "epoch": epoch,
        "args": args,
        "history": history,
        "train_loss_curve": history.get("train_loss", []),
        "val_loss_curve": history.get("val_loss", []),
        "train_size": train_size,
        "val_size": val_size,
    }


def save_training_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Save training checkpoint payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoint with compatibility across torch versions."""
    load_kw: dict[str, Any] = {"map_location": "cpu"}
    try:
        checkpoint = torch.load(path, weights_only=False, **load_kw)
    except TypeError:
        checkpoint = torch.load(path, **load_kw)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"unsupported checkpoint format at {path}")
    return checkpoint


def extract_model_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return model state dict from either wrapped or raw checkpoints."""
    model_state = checkpoint.get("model", checkpoint)
    if not isinstance(model_state, dict):
        raise TypeError("checkpoint does not contain a valid model state dict")
    return model_state

