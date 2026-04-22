# Plan: Frequency-Domain / Multi-Band DEM Learning

## Goal
Predict terrain corrections in separate spatial-frequency bands (low/mid/high), then recombine.

## Why This Is Different
- Current models learn in pixel space only.
- This enforces explicit control over smooth trends vs sharp terrain edges.

## Implementation Plan
1. Choose decomposition (wavelet pyramid or Laplacian pyramid).
2. Build branch heads for each frequency band.
3. Train with band-specific losses and optional weighting by terrain stratum.
4. Reconstruct final DEM from predicted bands.

## Required Beyond New Model
- **Input/target preparation changes**
  - Add deterministic decomposition of `z_gt` (and optionally `z_lr`) into bands.
  - Define reconstruction operator used identically in train/eval.
- **Loss and metric changes**
  - Add per-band reconstruction losses.
  - Add spectral consistency metrics (power spectrum error by band).
- **Training loop updates**
  - Track per-band losses in checkpoint history.
  - Optional curriculum: train low bands first, then high bands.
- **Evaluation updates**
  - Save and inspect per-band error summaries for debugging.

## Data and Preprocessing Needs
- No new external inputs required.
- Need stable normalization per band to avoid high-frequency instability.

## Evaluation Plan
- Standard metrics plus:
  - band-wise RMSE,
  - edge-focused score around contour crossings/high slope zones.

## Risks
- Decomposition/reconstruction mismatch can create artifacts.
- High-frequency branch may overfit noise.

## Pilot Exit Criteria
- Keep if high-slope and contour-proximal errors drop while preserving global RMSE.
