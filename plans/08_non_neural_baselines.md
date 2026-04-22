# Plan: Strong Non-Neural Baselines (GBDT / Shallow MLP)

## Goal
Test robust tabular/feature-based predictors as competitive baselines under distribution shift.

## Residual + Safeguard Requirements
- Predict correction residuals relative to `z_lr` (per-patch or per-pixel), not absolute DEM directly.
- Reconstruct only with `z_hat = z_lr + r_pred`.
- Apply residual range constraints (clip by terrain-plausible limits).
- Keep masking/sanitization safeguards in feature extraction (valid pixels only, finite values only, uncertainty-aware features).
- Evaluate with the same baseline metric suite and non-AU/AU split discipline.

## Why This Is Different
- Current effort is deep convolution/attention models only.
- Tree models can perform strongly with engineered geomorphology features and are easier to interpret.

## Implementation Plan
1. Build feature extraction pipeline from each chip:
  - local stats on `z_lr`,
  - gradients/curvature summaries,
  - uncertainty and mask features,
  - AE summary descriptors.
2. Train models for:
  - per-patch aggregate correction,
  - optional per-pixel correction (sampled points) if feasible.
3. Evaluate standalone and as hybrid post-corrector on neural outputs.

## Required Beyond New Model
- **Input representation changes**
  - Feature engineering and serialization pipeline (Parquet/NumPy tables).
  - Optional neighborhood context feature windows.
- **Training infra changes**
  - New training/eval scripts using LightGBM/XGBoost/sklearn.
  - Hyperparameter search workflow distinct from PyTorch stack.
- **Inference integration**
  - Add prediction-source support for tabular models in eval.
  - Optional stacking path: neural output + tabular correction.
- **Metrics and analysis**
  - Feature importance / SHAP tracking for error diagnosis.

## Data and Preprocessing Needs
- No new raw inputs required.
- Need careful train/val split integrity to avoid leakage via engineered aggregates.

## Evaluation Plan
- Compare against `z_lr` and neural models on same manifests.
- Use as a diagnostic: if non-neural matches neural on AU, representation/pipeline may be bottleneck.

## Risks
- Feature engineering effort can grow quickly.
- Per-pixel tabular inference may be heavy without sampling/compression.

## Pilot Exit Criteria
- Keep if it meaningfully beats `z_lr` and narrows gap to neural baselines with better calibration.
