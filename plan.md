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
- Training script: `train_dem.py`
- Evaluation script: `eval_dem.py`
- Key reference metrics (from `status.md`): compare new models to current best on
  - elevation RMSE
  - slope RMSE deg
  - elevation MAE

## Candidate Architectures

1. Spatial Gated Fusion U-Net (replace global FiLM with local gated fusion)
2. Windowed Cross-Attention Fusion U-Net (coarse scales only)
3. Hybrid Conv U-Net + Lightweight Transformer bottleneck
4. RCAN-style residual channel-attention backbone with AE conditioning
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

- [x] Add architecture selection flag to `train_dem.py` (e.g., `--arch`).
- [x] Implement candidate #1 (Spatial Gated Fusion U-Net).
- [ ] Implement candidate #2 (Windowed Cross-Attention U-Net).
- [ ] Implement candidate #3 (Hybrid Conv+Transformer bottleneck).
- [ ] Implement candidate #4 (RCAN-style backbone).
- [ ] Add consistent output naming for checkpoints by architecture.
- [ ] Run 6-epoch screening for each candidate.
- [ ] Record results and shortlist top 1-2 for longer runs.

## Next Run Commands

Use these to compare baseline FiLM vs spatial gated fusion with identical settings.

### Train baseline (FiLM)

```bash
python3 train_dem.py \
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
python3 train_dem.py \
  --arch gated_unet \
  --manifest train_manifest_seed42.txt \
  --epochs 6 \
  --batch-size 32 \
  --workers 3 \
  --amp \
  --checkpoint-out dem_film_unet_arch_gated_6ep.pt
```

### Evaluate checkpoints

```bash
python3 eval_dem.py \
  --prediction-source model \
  --checkpoint dem_film_unet_arch_film_6ep_epoch_003.pt \
  --manifest holdout_manifest_seed42.txt \
  --batch-size 32 \
  --workers 3 \
  --amp \
  --output-json eval_arch_film_epoch003.json
```

```bash
python3 eval_dem.py \
  --prediction-source model \
  --checkpoint dem_film_unet_arch_gated_6ep_epoch_003.pt \
  --manifest holdout_manifest_seed42.txt \
  --batch-size 32 \
  --workers 3 \
  --amp \
  --output-json eval_arch_gated_epoch003.json
```
