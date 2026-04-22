# Plan: Large-Context Tile Training (Patch-to-Tile)

## Goal
Train with substantially larger spatial context and compute loss on center crops to reduce border/context starvation.

## Why This Is Different
- Current training is 128x128 local chips.
- This uses broader terrain context for disambiguation.

## Implementation Plan
1. Build tile sampler (e.g., 512x512 context windows).
2. Predict full tile but compute primary loss on center crop (e.g., 128 or 256).
3. Add overlap-aware inference stitching for full-size outputs.
4. Optionally combine with multi-scale pyramid inputs.

## Required Beyond New Model
- **Input pipeline changes**
  - New dataset/indexing mode for larger tiles or grouped neighboring patches.
  - Coordinate-consistent stitching metadata.
- **Memory/compute changes**
  - Smaller batch size, gradient accumulation, checkpointing, mixed precision tuning.
- **Training loop changes**
  - Center-crop loss masking.
  - Optional edge-loss to avoid seam artifacts.
- **Inference changes**
  - Sliding-window prediction and blending at overlaps.

## Data and Preprocessing Needs
- Likely requires regenerated training samples or a loader that can assemble neighbor patches into tiles.
- Must verify valid-weight coverage when tiles include sparse/noisy regions.

## Evaluation Plan
- Standard metrics plus seam-specific diagnostics on stitched predictions.
- Stratify by terrain continuity classes where long-range context matters.

## Risks
- Engineering effort in tiling/stitching.
- Throughput drop may slow experimentation.

## Pilot Exit Criteria
- Keep if slope/curvature improve on AU with minimal seam artifacts and manageable runtime.
