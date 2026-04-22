"""Canonical stack band ordering and semantics."""

from __future__ import annotations

STACK_BAND_NAMES: tuple[str, ...] = (
    "z_gt10",
    "z_gtMask",
    "z_lr10",
    "u_enc",
    "slope",
    "residAbs",
    "M_bld10",
    "M_wp10",
    "M_ws10",
    "weight",
)

STACK_BAND_TO_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(STACK_BAND_NAMES)}

