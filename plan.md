# DEM Architecture Exploration Plan

This document tracks architecture experiments while keeping the same training stack and `128x128` chip size.

## Goals

- Test at least 4 alternative model architectures against the current FiLM U-Net baseline.
- Use short, comparable runs to identify promising directions quickly.
- Avoid losing context across runs by recording commands, checkpoints, and metrics in one place.

## Ground Rules

- Keep data pipeline unchanged (`/data/training`, existing manifests).
- Keep loss stack unchanged unless explicitly noted.
- Keep patch/chip size unchanged (`128x128`).
- Compare models using the same evaluation pipeline and metrics.

## Baseline

- Model: `DemFilmUNet` (`dem_film_unet.py`)
- Training script: `train_experiment.py`
- Evaluation script: `eval_experiment.py`
- Key reference metrics (from `status.md`): compare new models to current best on
  - elevation RMSE
  - slope RMSE deg
  - elevation MAE

## Candidate Architectures

1. Spatial Gated Fusion U-Net — `--arch gated_unet` (local gated AE fusion at S1–S3)
2. Windowed Cross-Attention Fusion U-Net — `--arch xattn_unet` (FiLM at S1; windowed DEM→AE cross-attention at S2–S3, 8×8 windows)
3. Hybrid Conv U-Net + Lightweight Transformer bottleneck — `--arch hybrid_tf_unet` (FiLM S1–S3; two windowed self-attn + FFN blocks at S3 before decoder)
4. RCAN-style residual channel-attention backbone with AE conditioning — `--arch rcan_ae_unet` (RCAB residual groups + AE channel-gated conditioning)
5. Optional: Mixture-of-Experts residual head

## Screening Protocol (Short Runs)

- Train each candidate for 6 epochs (or similar fixed GPU-time budget).
- Save checkpoints every epoch.
- Evaluate checkpoints at epochs 1, 3, and 6.
- Rank by:
  1. elevation RMSE
  2. slope RMSE deg
  3. elevation MAE

### Early Stop Rule

- If a candidate is clearly worse than baseline by epoch 3 on both elevation RMSE and slope RMSE deg, deprioritize it.

## Run Log

Use one section per run.

### Template

#### Run ID

- Date:
- Architecture:
- Branch/commit:
- Train command:
- Eval command:
- Checkpoints:
- Notes:

#### Results

- Epoch 1:
  - elev RMSE:
  - slope RMSE deg:
  - elev MAE:
- Epoch 3:
  - elev RMSE:
  - slope RMSE deg:
  - elev MAE:
- Epoch 6:
  - elev RMSE:
  - slope RMSE deg:
  - elev MAE:

#### Decision

- Status: keep / drop / revisit
- Why:

## Current TODO

- [x] Add architecture selection flag to `train_experiment.py` (e.g., `--arch`).
- [x] Implement candidate #1 (Spatial Gated Fusion U-Net).
- [x] Implement candidate #2 (Windowed Cross-Attention U-Net).
- [x] Implement candidate #3 (Hybrid Conv+Transformer bottleneck).
- [x] Implement candidate #4 (RCAN-style backbone).
- [ ] Add consistent output naming for checkpoints by architecture.
- [ ] Run 6-epoch screening for each candidate.
- [ ] Record results and shortlist top 1-2 for longer runs.

## Next Run Commands

Use these to compare baseline FiLM vs spatial gated fusion with identical settings.

### Train baseline (FiLM)

```bash
python3 train_experiment.py \
  --arch film_unet \
  --manifest train_manifest_seed42.txt \
  --epochs 6 \
  --batch-size 32 \
  --workers 3 \
  --amp \
  --checkpoint-out dem_film_unet_arch_film_6ep.pt
```

### Train candidate #1 (Gated)

```bash
python3 train_experiment.py \
  --arch gated_unet \
  --manifest train_manifest_seed42.txt \
  --epochs 6 \
  --batch-size 32 \
  --workers 3 \
  --amp \
  --checkpoint-out dem_film_unet_arch_gated_6ep.pt
```

### Train candidate #2 (Cross-attn)

```bash
python3 train_experiment.py \
  --arch xattn_unet \
  --manifest train_manifest_seed42.txt \
  --epochs 6 \
  --batch-size 32 \
  --workers 8 \
  --amp \
  --checkpoint-out dem_film_unet_arch_xattn_6ep.pt
```

### Train candidate #3 (Hybrid transformer bottleneck)

```bash
python3 train_experiment.py \
  --arch hybrid_tf_unet \
  --manifest train_manifest_seed42.txt \
  --epochs 6 \
  --batch-size 32 \
  --workers 8 \
  --amp \
  --checkpoint-out dem_film_unet_arch_hybrid_tf_6ep.pt
```

### Train candidate #4 (RCAN+AE)

```bash
python3 train_experiment.py \
  --arch rcan_ae_unet \
  --manifest train_manifest_seed42.txt \
  --epochs 6 \
  --batch-size 32 \
  --workers 8 \
  --amp \
  --checkpoint-out dem_film_unet_arch_rcan_ae_6ep.pt
```

### Evaluate checkpoints

```bash
python3 eval_experiment.py \
  --prediction-source model \
  --checkpoint dem_film_unet_arch_film_6ep_epoch_003.pt \
  --manifest holdout_manifest_seed42.txt \
  --batch-size 32 \
  --workers 3 \
  --amp \
  --output-json eval_arch_film_epoch003.json
```

```bash
python3 eval_experiment.py \
  --prediction-source model \
  --checkpoint dem_film_unet_arch_gated_6ep_epoch_003.pt \
  --manifest holdout_manifest_seed42.txt \
  --batch-size 32 \
  --workers 3 \
  --amp \
  --output-json eval_arch_gated_epoch003.json
```

```bash
python3 eval_experiment.py \
  --prediction-source model \
  --arch xattn_unet \
  --checkpoint dem_film_unet_arch_xattn_6ep_epoch_003.pt \
  --manifest holdout_manifest_seed42.txt \
  --batch-size 32 \
  --workers 3 \
  --amp \
  --output-json eval_arch_xattn_epoch003.json
```

```bash
python3 eval_experiment.py \
  --prediction-source model \
  --arch hybrid_tf_unet \
  --checkpoint dem_film_unet_arch_hybrid_tf_6ep_epoch_003.pt \
  --manifest holdout_manifest_seed42.txt \
  --batch-size 32 \
  --workers 3 \
  --amp \
  --output-json eval_arch_hybrid_tf_epoch003.json
```

```bash
python3 eval_experiment.py \
  --prediction-source model \
  --arch rcan_ae_unet \
  --checkpoint dem_film_unet_arch_rcan_ae_6ep_epoch_003.pt \
  --manifest holdout_manifest_seed42.txt \
  --batch-size 32 \
  --workers 3 \
  --amp \
  --output-json eval_arch_rcan_ae_epoch003.json
```

## Loss Preset Sweep (contour-aware supervision)

Orthogonal to `--arch`, `train_experiment.py` and `eval_experiment.py` now expose a
`--loss-preset` flag that selects a contour-aware loss configuration from
`contours.md` without changing the dataset or the backbone:

- `baseline`  – elev SmoothL1 + λ·|slope diff| (matches legacy `loss_dem`).
- `geom`      – baseline + dx/dy gradient L1 + Laplacian (curvature) L1 + 2x multi-scale elev L1.
- `multitask` – `geom` + contour SDF L1 + soft contour-indicator L1.
- `contour`   – baseline + contour SDF L1 only.

All auxiliary targets are derived from `z_gt`/`z_hat` at train time, so no
patch regeneration is required. The SDF target is a signed vertical offset
to the nearest contour at `--contour-interval` meters (default `10`).

### Sweep commands

Run all four presets on the FiLM baseline under the non-AU/AU split. The
default configuration trains on the hardest **10%** of non-AU patches
and evaluates on the hardest **10%** of AU patches (both ranked by
`resid_scale` / `residAbs_p95`, with quality filtering) for **3 epochs**
per preset — a fast screening sweep rather than a full comparison. The
hard AU val slice is shared across the `z_lr` baseline and all presets,
so within-sweep rank-ordering is valid; re-evaluate the winner on the
full AU manifest (`VAL_HARD_FRACTION=1.0`) for honest absolute numbers.

```bash
PATCH_TABLE=200k.geojson \
  ./run_loss_presets_non_au_vs_au.sh
```

Override knobs via env vars (`EPOCHS`, `BATCH_SIZE`, `WORKERS`, `ARCH`,
`PRESETS`, `CONTOUR_INTERVAL`, `HARD_FRACTION`, `VAL_HARD_FRACTION`,
`HARD_SCORE_FIELD`, etc.). Set `HARD_FRACTION=1.0` or
`VAL_HARD_FRACTION=1.0` to fall back to the full train or val manifest.
`select_hard_patches.py` is the underlying subsampler:

```bash
python3 select_hard_patches.py \
  --manifest runs/<run>/manifests/train_non_au_manifest.txt \
  --patch-table 200k.geojson \
  --fraction 0.10 \
  --score-field resid_scale \
  --out runs/<run>/manifests/train_non_au_hard_resid_scale_p10_manifest.txt \
  --summary-json runs/<run>/manifests/train_non_au_hard_resid_scale_p10_summary.json
```

Eval JSONs land under `runs/<RUN_NAME>/eval/eval_film_unet_<preset>.json`
and now include `grad_x_rmse_w`, `grad_y_rmse_w`, `laplacian_rmse_w`, and
`sdf_rmse_w` alongside the existing elevation/slope metrics.

### Results template

#### Preset comparison (FiLM, 3 epochs, hardest 10% non-AU train, hardest 10% AU val)

- `baseline`
  - elev RMSE:
  - slope RMSE deg:
  - sdf RMSE (m):
  - grad_x RMSE:
  - elev MAE:
- `geom`
  - elev RMSE:
  - slope RMSE deg:
  - sdf RMSE (m):
  - grad_x RMSE:
  - elev MAE:
- `multitask`
  - elev RMSE:
  - slope RMSE deg:
  - sdf RMSE (m):
  - grad_x RMSE:
  - elev MAE:
- `contour`
  - elev RMSE:
  - slope RMSE deg:
  - sdf RMSE (m):
  - grad_x RMSE:
  - elev MAE:

### Decision rule

Rank by (1) elev RMSE, (2) slope RMSE deg, (3) sdf RMSE, (4) grad_x/grad_y
RMSE, (5) elev MAE. Winning preset is promoted to the current-best arch
for a longer run. If `contour` alone matches or beats `multitask`, the
follow-up is a dedicated SDF prediction head on the decoder.
