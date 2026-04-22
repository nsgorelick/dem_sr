# Plan: Diffusion-Based Conditional DEM Refinement

## Goal
Replace one-shot regression with iterative denoising that models a conditional distribution:
`p(z_gt | z_lr, x_dem, x_ae)`.

## Why This Is Different
- Current methods are deterministic feed-forward residual predictors.
- Diffusion performs many denoising steps and can model multi-modal plausible terrain shapes.

## Implementation Plan
1. Build a conditional UNet diffusion backbone operating on DEM residual space.
2. Define noise schedule and training objective (epsilon or v-prediction).
3. Condition on `z_lr`, `x_dem`, `x_ae`, and optionally uncertainty channels.
4. Add fast sampler settings for screening and high-quality sampler for final eval.
5. Train short pilot on smaller chips or reduced timesteps, then scale.

## Required Beyond New Model
- **Input construction changes**
  - Add a diffusion timestep embedding input.
  - Decide whether to predict full DEM, residual to `z_lr`, or residual to bicubic baseline.
  - Potentially add extra channels for masks/uncertainty explicitly to conditioning.
- **Training loop changes**
  - Sample random timestep `t` per batch.
  - Add forward noising step and denoising loss path.
  - Add EMA weights for stable sampling.
- **Inference changes**
  - Add iterative denoising sampler path to `eval_experiment.py`.
  - Add configurable sampler params (steps, guidance strength, deterministic/stochastic).
- **Checkpoint format**
  - Store diffusion config (schedule, parameterization, sampler defaults).
- **Compute requirements**
  - Higher memory and wall-clock than current runs; need a reduced-cost pilot mode.

## Data and Preprocessing Needs
- No required patch regeneration, but may need:
  - normalized residual targets for diffusion stability,
  - clipping policy for extreme elevations/slope artifacts.

## Evaluation Plan
- Compare against best current model on:
  - elevation RMSE/MAE,
  - slope RMSE deg,
  - curvature/laplacian RMSE,
  - SDF RMSE.
- Add stochastic consistency metric: mean/std across multiple samples per patch.

## Risks
- Expensive and slower to iterate.
- Can underperform if sampling steps are too small.

## Pilot Exit Criteria
- Keep if it beats current best on slope RMSE deg and at least ties on elevation RMSE.
- Drop if compute cost is much higher without measurable geometry gain.
