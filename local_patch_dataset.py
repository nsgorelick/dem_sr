"""PyTorch Dataset: DEM training chips as local GeoTIFFs on disk.

Expected layout:

  {root}/stack/{patch_id}.tif
  {root}/ae/{patch_id}_aef_uint8.tif

Stack bands (1..10): z_gt10, z_gtMask, z_lr10, u_enc, slope, residAbs,
M_bld10, M_wp10, M_ws10, weight.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_AE_SUFFIX = "_aef_uint8.tif"


def list_patch_stems(training_root: str | Path) -> list[str]:
    """List patch ids that have both stack and AE files under a local training root."""
    root = Path(training_root)
    stack_dir = root / "stack"
    ae_dir = root / "ae"
    if not stack_dir.is_dir() or not ae_dir.is_dir():
        logger.warning("Expected local training dirs %s and %s", stack_dir, ae_dir)
        return []

    stack_stems = {
        p.stem
        for p in stack_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".tif" and ".part" not in p.name
    }
    ae_stems = {
        p.name[: -len(_AE_SUFFIX)]
        for p in ae_dir.iterdir()
        if p.is_file() and p.name.endswith(_AE_SUFFIX) and ".part" not in p.name
    }
    both = sorted(stack_stems & ae_stems)
    if not both:
        logger.warning("No patch stems found with both stack and AE files under %s", root)
    return both


def load_patch_stems_manifest(path: Path | str) -> list[str]:
    """One patch id per line (filename stem, no extension)."""
    text = Path(path).read_text(encoding="utf-8")
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def build_loss_weight(
    M_bld: np.ndarray,
    M_wp: np.ndarray,
    M_ws: np.ndarray,
    U_enc: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Match export ``weight`` construction: ``(1-M_bld)*(1-M_wp)*...``.

    ``M_bld10`` in the stack is already dilated/buffered in Earth Engine; no extra dilation here.
    """
    W = np.ones_like(U_enc, dtype=np.float32)
    W *= 1.0 - M_bld
    W *= 1.0 - M_wp
    W *= 1.0 - 0.8 * M_ws
    W *= 1.0 - 0.5 * (U_enc * U_enc)
    if valid_mask is not None:
        W *= (valid_mask > 0.5).astype(np.float32)
    return W


def decode_ae_uint8(ae_u8: np.ndarray) -> np.ndarray:
    """(64, H, W) uint8 -> float32 in [-1, 1]."""
    x = ae_u8.astype(np.float32) / 255.0
    return x * 2.0 - 1.0


def sanitize_float32(x: np.ndarray) -> np.ndarray:
    """Replace non-finite raster values with zeros for masked training."""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


class LocalDemPatchDataset(Dataset):
    """Random-access reads of stack + AE GeoTIFFs from local disk."""

    def __init__(
        self,
        training_root: str,
        patch_stems: list[str] | None = None,
        *,
        use_precomputed_weight: bool = False,
        load_ae: bool = True,
        candidate_root: str | Path | None = None,
        candidate_product: str | None = None,
        candidate_band: int = 1,
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
        max_patches: int | None = None,
    ) -> None:
        """
        Args:
            training_root: e.g. ``/data/training``.
            patch_stems: If None, list stems from the local training root.
            use_precomputed_weight: If True, use stack band ``weight`` instead of recomputing W.
            load_ae: If False, skip reading AE GeoTIFFs and omit ``x_ae`` from samples.
            candidate_root: Optional root for per-patch comparison rasters.
            candidate_product: Optional product subdirectory under ``candidate_root``.
            candidate_band: 1-based band to read from the comparison raster.
            transform: Optional callable mutating the sample dict (e.g. augmentation).
            max_patches: Keep only the first *N* stems after listing (debug / smoke runs).
        """
        self.training_root = training_root.rstrip("/")
        self.use_precomputed_weight = use_precomputed_weight
        self.load_ae = load_ae
        self.candidate_root = str(candidate_root).rstrip("/") if candidate_root is not None else None
        self.candidate_product = candidate_product
        self.candidate_band = int(candidate_band)
        self.transform = transform

        if patch_stems is None:
            self._stems = list_patch_stems(self.training_root)
        else:
            self._stems = list(patch_stems)
        if max_patches is not None:
            self._stems = self._stems[: int(max_patches)]

        self._stack_base = f"{self.training_root}/stack"
        self._ae_base = f"{self.training_root}/ae"

    def __len__(self) -> int:
        return len(self._stems)

    def _paths_for_stem(self, stem: str) -> tuple[str, str]:
        stack_path = f"{self._stack_base}/{stem}.tif"
        ae_path = f"{self._ae_base}/{stem}{_AE_SUFFIX}"
        return stack_path, ae_path

    def _candidate_path_for_stem(self, stem: str) -> str | None:
        if self.candidate_root is None:
            return None
        root = Path(self.candidate_root)
        if self.candidate_product:
            return str(root / self.candidate_product / f"{stem}.tif")
        direct = root / f"{stem}.tif"
        if direct.is_file():
            return str(direct)
        return str(root / stem / f"{stem}.tif")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        stem = self._stems[index]
        stack_path, ae_path = self._paths_for_stem(stem)

        with rasterio.open(stack_path) as src:
            if src.count < 10:
                raise ValueError(f"{stack_path}: expected >=10 bands, got {src.count}")
            stack = src.read(out_dtype=np.float32)

        ae_u8 = None
        if self.load_ae:
            with rasterio.open(ae_path) as asrc:
                if asrc.count != 64:
                    raise ValueError(f"{ae_path}: expected 64 AE bands, got {asrc.count}")
                ae_u8 = asrc.read(out_dtype=np.uint8)

        z_gt = stack[0:1]
        z_gt_mask = stack[1:2]
        z_lr = stack[2:3]
        u_enc = stack[3:4]
        m_bld = stack[6:7]
        m_wp = stack[7:8]
        m_ws = stack[8:9]

        finite_valid = (
            np.isfinite(z_gt)
            & np.isfinite(z_lr)
            & np.isfinite(u_enc)
            & np.isfinite(m_bld)
            & np.isfinite(m_wp)
            & np.isfinite(m_ws)
        )
        valid_mask = (z_gt_mask > 0.5) & finite_valid

        z_gt = sanitize_float32(z_gt)
        z_lr = sanitize_float32(z_lr)
        u_enc = sanitize_float32(u_enc)
        m_bld = sanitize_float32(m_bld)
        m_wp = sanitize_float32(m_wp)
        m_ws = sanitize_float32(m_ws)

        if self.use_precomputed_weight:
            W = sanitize_float32(stack[9:10])
            W *= valid_mask.astype(np.float32)
        else:
            W = build_loss_weight(
                m_bld[0], m_wp[0], m_ws[0], u_enc[0], valid_mask=valid_mask[0]
            )[np.newaxis, ...]

        x_dem = np.concatenate([z_lr, u_enc, m_bld, m_wp, m_ws], axis=0).astype(np.float32)
        candidate = None
        candidate_valid = None
        candidate_path = self._candidate_path_for_stem(stem)
        if candidate_path is not None:
            with rasterio.open(candidate_path) as csrc:
                if self.candidate_band < 1 or self.candidate_band > csrc.count:
                    raise ValueError(
                        f"{candidate_path}: requested band {self.candidate_band}, available bands=1..{csrc.count}"
                    )
                cand = csrc.read(self.candidate_band, out_dtype=np.float32, masked=True)
            candidate_valid = (~np.ma.getmaskarray(cand) & np.isfinite(np.ma.getdata(cand)))[
                np.newaxis, ...
            ].astype(np.float32)
            candidate = sanitize_float32(np.ma.getdata(cand)[np.newaxis, ...])
            if candidate.shape[-2:] != z_gt.shape[-2:]:
                raise ValueError(
                    f"{candidate_path}: shape {candidate.shape[-2:]} does not match patch shape {z_gt.shape[-2:]}"
                )

        sample = {
            "stem": stem,
            "x_dem": torch.from_numpy(x_dem),
            "z_lr": torch.from_numpy(z_lr),
            "z_gt": torch.from_numpy(z_gt),
            "w": torch.from_numpy(W),
        }
        if ae_u8 is not None:
            sample["x_ae"] = torch.from_numpy(decode_ae_uint8(ae_u8))
        if candidate is not None and candidate_valid is not None:
            sample["z_candidate"] = torch.from_numpy(candidate)
            sample["z_candidate_valid"] = torch.from_numpy(candidate_valid)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def collate_dem_batch(samples: list[dict]) -> dict[str, torch.Tensor | list]:
    """Stack tensors; keep stems as list."""
    batch = {
        "stem": [s["stem"] for s in samples],
        "x_dem": torch.stack([s["x_dem"] for s in samples], dim=0),
        "z_lr": torch.stack([s["z_lr"] for s in samples], dim=0),
        "z_gt": torch.stack([s["z_gt"] for s in samples], dim=0),
        "w": torch.stack([s["w"] for s in samples], dim=0),
    }
    if "x_ae" in samples[0]:
        batch["x_ae"] = torch.stack([s["x_ae"] for s in samples], dim=0)
    if "z_candidate" in samples[0]:
        batch["z_candidate"] = torch.stack([s["z_candidate"] for s in samples], dim=0)
        batch["z_candidate_valid"] = torch.stack([s["z_candidate_valid"] for s in samples], dim=0)
    return batch
