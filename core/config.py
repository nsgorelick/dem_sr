"""Shared experiment CLI/config helpers."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Normalized config used by experiment entrypoints."""

    experiment: str = "baseline"
    data_root: str | None = None
    manifest: Path | None = None
    list_from_root: bool = False
    max_patches: int | None = None
    batch_size: int = 4
    workers: int = 2
    amp: bool = False
    precomputed_weight: bool = False
    contour_interval: float = 10.0
    arch: str = "film_unet"
    loss_preset: str = "baseline"

_DEFAULT_CONFIG = ExperimentConfig()


def add_shared_experiment_args(parser: ArgumentParser) -> None:
    """Add common arguments used by train/eval experiment entrypoints."""
    parser.add_argument("--data-root", default="/data/training", help="Parent root with stack/ and ae/ subdirs")
    parser.add_argument("--manifest", type=Path, default=None, help="One patch stem per line")
    parser.add_argument("--max-patches", type=int, default=None, help="Cap dataset size")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="torch.cuda.amp")
    parser.add_argument("--precomputed-weight", action="store_true", help="Use stack weight band")
    parser.add_argument(
        "--contour-interval",
        type=float,
        default=10.0,
        help="Contour interval in meters for SDF metric",
    )
    parser.add_argument("--arch", default="film_unet")
    parser.add_argument("--loss-preset", default="baseline")


def resolve_config(args: Namespace, *, default_data_root: str | None = None) -> ExperimentConfig:
    """Create normalized experiment config from argparse namespace."""
    data_root = args.data_root if getattr(args, "data_root", None) is not None else default_data_root
    return ExperimentConfig(
        experiment=str(getattr(args, "experiment", "baseline")),
        data_root=data_root,
        manifest=getattr(args, "manifest", None),
        list_from_root=bool(getattr(args, "list_from_root", False)),
        max_patches=getattr(args, "max_patches", None),
        batch_size=int(getattr(args, "batch_size", 4)),
        workers=int(getattr(args, "workers", 2)),
        amp=bool(getattr(args, "amp", False)),
        precomputed_weight=bool(getattr(args, "precomputed_weight", False)),
        contour_interval=float(getattr(args, "contour_interval", 10.0)),
        arch=str(getattr(args, "arch", "film_unet")),
        loss_preset=str(getattr(args, "loss_preset", "baseline")),
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "experiment": config.experiment,
        "data_root": config.data_root,
        "manifest": config.manifest,
        "list_from_root": config.list_from_root,
        "max_patches": config.max_patches,
        "batch_size": config.batch_size,
        "workers": config.workers,
        "amp": config.amp,
        "precomputed_weight": config.precomputed_weight,
        "contour_interval": config.contour_interval,
        "arch": config.arch,
        "loss_preset": config.loss_preset,
    }


def apply_preset_defaults(config: ExperimentConfig, preset: dict[str, Any]) -> ExperimentConfig:
    """Apply preset values only for fields still set to shared defaults."""
    current = asdict(config)
    defaults = asdict(_DEFAULT_CONFIG)
    updates: dict[str, Any] = {}
    for key, value in preset.items():
        if key not in current:
            continue
        if current[key] == defaults[key]:
            updates[key] = value
    return replace(config, **updates)


def apply_namespace_preset_defaults(args: Namespace, parser: ArgumentParser, preset: dict[str, Any]) -> None:
    """Apply preset values to args when corresponding flags are still defaults."""
    for key, value in preset.items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        try:
            default = parser.get_default(key)
        except Exception:  # noqa: BLE001
            continue
        if current == default:
            setattr(args, key, value)

