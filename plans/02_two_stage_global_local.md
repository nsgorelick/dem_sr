# Plan: Two-Stage Global + Local Residual System

## Goal
Decompose error into:
- low-frequency/global bias field correction,
- high-frequency/local detail refinement.

## Residual + Safeguard Requirements
- Both stages predict residuals (Stage A coarse residual, Stage B detail residual), never absolute DEM.
- Reconstruct via `z_mid = z_lr + r_coarse_up` and `z_hat = z_mid + r_detail`.
- Apply residual caps to both residual heads.
- Keep weighted masking (`W`), finite-value sanitization, and uncertainty/mask channels in both stages.
- Preserve evaluation parity with baseline metrics and the non-AU train / AU val protocol.

## Why This Is Different
- Current models learn one residual in a single forward pass.
- This explicitly separates scale-dependent error modes.

## Implementation Plan
1. Stage A (global): predict coarse correction at low resolution.
2. Upsample Stage A output and apply to `z_lr` to form intermediate DEM.
3. Stage B (local): predict detail residual conditioned on intermediate DEM plus guidance.
4. Train sequentially (A then B), then optional joint finetune.

## Required Beyond New Model
- **Input/output pipeline changes**
  - Create low-resolution target path for Stage A.
  - Add intermediate DEM tensor handoff from Stage A to Stage B.
- **Training orchestration**
  - Support multi-stage training schedule and separate checkpoints.
  - Add option to freeze Stage A during Stage B training.
- **Loss stack changes**
  - Stage A weighted toward elevation bias and broad slope.
  - Stage B weighted toward gradient/curvature/contour detail.
- **Inference path**
  - Update eval to run two checkpoints in sequence.
  - Add fallback path if only Stage A is available.

## Data and Preprocessing Needs
- Existing data works, but add:
  - downsample policy for Stage A targets,
  - anti-aliasing consistency between train and eval.

## Evaluation Plan
- Report per-stage and combined performance:
  - Stage A only vs Stage A+B.
- Stratify by relief/slope bins to validate local-detail gains in steep terrain.

## Risks
- Complexity in checkpoint management and reproducibility.
- Error propagation if Stage A is poor.

## Pilot Exit Criteria
- Keep if combined model improves slope/curvature metrics without worsening elevation RMSE.
