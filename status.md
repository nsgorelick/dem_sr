# DEM Project Status

## Overview

This repo has been patched to support:

- local-only training/evaluation from `/data/training`
- deterministic train/holdout manifests
- per-epoch checkpointing during training
- progress bars for training and evaluation
- evaluation of multiple sources in one pass:
  - model inference
  - input baseline `z_lr`
  - external per-patch rasters such as FABDEM
- export of external comparison DTMs from Earth Engine to the same 10 m patch grid

Current environment assumptions:

- repo: `/home/gorelick/projects/DEM`
- venv: `/home/gorelick/venv-cu128`
- training data root: `/data/training`
- comparison raster root: `/data/comparison`

## Data Splits

Deterministic manifest split already exists:

- `train_manifest_seed42.txt`
- `holdout_manifest_seed42.txt`
- `manifest_summary_seed42.json`

Counts from `manifest_summary_seed42.json`:

- total patches: `150,521`
- holdout patches: `15,052`
- train patches: `135,469`
- holdout fraction: `10%`
- seed: `42`

Note: this is a random patch-level holdout, not a strict spatial holdout.

Additional data now available for future evaluation planning:

- approximately `50,000` more Australia patches are available
- plan is to reserve these for the "real" holdout once the model pipeline is working well enough to justify a stronger final evaluation

## Code Changes Made

### `local_patch_dataset.py`

Current behavior:

- local-only dataset loader
- sanitizes non-finite values with `nan_to_num`
- optional AE loading via `load_ae=False`
- optional external candidate raster loading via:
  - `candidate_root`
  - `candidate_product`
  - `candidate_band`
- collate function now supports optional:
  - `x_ae`
  - `z_candidate`
  - `z_candidate_valid`

This is what allows:

- fast `z_lr` evaluation without reading AE TIFFs
- raster-vs-truth evaluation for downloaded comparison DTMs

### `train_dem.py`

Current behavior:

- local-only training from `/data/training`
- `tqdm` progress bar
- per-epoch checkpoints like `dem_film_unet_epoch_015.pt`
- checkpoints now include loss history:
  - `history["train_loss"]`
  - `history["val_loss"]`
  - `train_loss_curve`
  - `val_loss_curve`
- final checkpoint `dem_film_unet.pt`
- modern AMP usage

### `eval_dem.py`

Current behavior:

- supports `--prediction-source` with one or more of:
  - `model`
  - `z_lr`
  - `raster`
- evaluates multiple sources in a single pass over the holdout set
- `tqdm` progress bar during eval
- optional JSON output via `--output-json`
- reports weighted:
  - elevation bias
  - elevation MAE
  - elevation RMSE
  - slope MAE
  - slope RMSE
  - slope MAE in degrees
  - slope RMSE in degrees

Important: when evaluating `raster`, invalid or masked pixels in the external raster are excluded from the weighted metrics for that source.

### `export_comparison_dtms.py`

New script created.

Current behavior:

- exports external comparison DTMs from Earth Engine
- writes separate single-band files per product and patch
- current supported products:
  - `fabdem`
  - `tdem_edem`
- outputs:
  - `comparison/fabdem/<patch_id>.tif`
  - `comparison/tdem_edem/<patch_id>.tif`
- uses the same `128x128`, `10 m` UTM patch grid as the training patches
- treats FABDEM and `TDEM_EDEM` as `ImageCollection`s
- export logic:
  - `collection.mosaic()`
  - anchored to `collection.first().projection()`
  - bilinear resampling
  - reprojected to patch UTM `10 m` grid

Current defaults:

- `--max-tries 5`
- top-level logging level back to `ERROR`

## Training Result

A 15-epoch training run completed successfully.

- runtime: `9:55:55`
- final checkpoint: `dem_film_unet.pt`
- per-epoch checkpoint example: `dem_film_unet_epoch_015.pt`

## Evaluation Commands

### Evaluate model only

```bash
source /home/gorelick/venv-cu128/bin/activate
cd /home/gorelick/projects/DEM

python eval_dem.py \
  --prediction-source model \
  --checkpoint dem_film_unet.pt \
  --manifest holdout_manifest_seed42.txt \
  --batch-size 16 \
  --workers 3 \
  --amp \
  --output-json eval_holdout_model_15ep.json
```

### Evaluate `z_lr` only

```bash
python eval_dem.py \
  --prediction-source z_lr \
  --data-root /data/training \
  --manifest holdout_manifest_seed42.txt \
  --batch-size 16 \
  --workers 3 \
  --output-json eval_holdout_zlr.json
```

### Evaluate model + `z_lr` + FABDEM in one pass

```bash
python eval_dem.py \
  --prediction-source model z_lr raster \
  --checkpoint dem_film_unet.pt \
  --manifest holdout_manifest_seed42.txt \
  --candidate-root /data/comparison \
  --candidate-product fabdem \
  --candidate-band 1 \
  --batch-size 16 \
  --workers 3 \
  --amp \
  --output-json eval_holdout_model_zlr_fabdem.json
```

## Export Commands

### Export FABDEM patches

```bash
source /home/gorelick/venv-cu128/bin/activate
cd /home/gorelick/projects/DEM

python export_comparison_dtms.py \
  --manifest holdout_manifest_seed42.txt \
  --product fabdem \
  --output-dir comparison \
  --pool-workers 50
```

### Export `TDEM_EDEM` patches

```bash
python export_comparison_dtms.py \
  --manifest holdout_manifest_seed42.txt \
  --product tdem_edem \
  --output-dir comparison \
  --pool-workers 50
```

### Debug export command

Useful when Earth Engine export appears stuck:

```bash
python export_comparison_dtms.py \
  --manifest holdout_manifest_seed42.txt \
  --product fabdem \
  --output-dir comparison \
  --pool-workers 1 \
  --max-tries 1
```

## Recorded Metrics

From `eval_holdout_model_zlr_fabdem.json`:

### Model

- patches: `15,052`
- sum(W): `151,686,976.4171`
- elev bias: `-0.4435 m`
- elev MAE: `1.3380 m`
- elev RMSE: `3.2045 m`
- slope MAE: `0.031955`
- slope RMSE: `0.558762`
- slope MAE deg: `1.4098 deg`
- slope RMSE deg: `2.9654 deg`

### `z_lr`

- patches: `15,052`
- sum(W): `151,686,976.4171`
- elev bias: `-0.4154 m`
- elev MAE: `1.8445 m`
- elev RMSE: `3.7796 m`
- slope MAE: `0.043852`
- slope RMSE: `0.560333`
- slope MAE deg: `2.0694 deg`
- slope RMSE deg: `3.7344 deg`

### FABDEM raster

- patches: `15,052`
- sum(W): `151,686,976.4171`
- elev bias: `-0.1195 m`
- elev MAE: `1.7228 m`
- elev RMSE: `3.8222 m`
- slope MAE: `0.037092`
- slope RMSE: `0.559958`
- slope MAE deg: `1.6927 deg`
- slope RMSE deg: `3.5172 deg`

## What We Learned

### Model vs `z_lr`

The model clearly improves on the input DEM baseline.

Improvements over `z_lr`:

- elevation MAE: `1.8445 -> 1.3380 m`
  - gain: `0.5065 m`
  - improvement: `27.5%`
- elevation RMSE: `3.7796 -> 3.2045 m`
  - gain: `0.5750 m`
  - improvement: `15.2%`
- slope MAE deg: `2.0694 -> 1.4098 deg`
  - gain: `0.6595 deg`
  - improvement: `31.9%`
- slope RMSE deg: `3.7344 -> 2.9654 deg`
  - gain: `0.7690 deg`
  - improvement: `20.6%`

### FABDEM diagnosis

The original FABDEM result was invalid because the comparison exporter was using the wrong patch-grid anchor.

What was found:

- the exported FABDEM rasters were shifted by half a patch (`640 m` in both `x` and `y`) relative to the training stack grid
- the root cause was a center-anchored grid helper in the exporter, while the training data and Earth Engine patch collections use corner-anchored patch coordinates
- both `export_comparison_dtms.py` and the current checked-in `export_patches_gcs.py` were updated to use the correct corner anchoring
- a one-patch re-export was verified to match existing training-stack bounds exactly

After fixing the exporter and re-running holdout evaluation, FABDEM became a credible baseline:

- elevation MAE: `1.7228 m`
- elevation RMSE: `3.8222 m`
- slope MAE deg: `1.6927 deg`
- slope RMSE deg: `3.5172 deg`

Interpretation:

- the previous FABDEM failure was caused by grid misalignment, not by a bad vertical datum or unusable source product
- FABDEM is now clearly competitive with `z_lr`
- the trained model still outperforms both `z_lr` and corrected FABDEM on this holdout

Corrected FABDEM comparison:

- vs `z_lr`
  - elevation MAE: `1.8445 -> 1.7228 m` (`6.6%` better)
  - slope MAE deg: `2.0694 -> 1.6927 deg` (`18.2%` better)
  - slope RMSE deg: `3.7344 -> 3.5172 deg` (`5.8%` better)
- vs model
  - elevation MAE: `1.7228 -> 1.3380 m` (`22.3%` better for model)
  - elevation RMSE: `3.8222 -> 3.2045 m` (`16.2%` better for model)
  - slope MAE deg: `1.6927 -> 1.4098 deg` (`16.7%` better for model)
  - slope RMSE deg: `3.5172 -> 2.9654 deg` (`15.7%` better for model)

### Checkpoint / training-curve diagnosis

The old checkpoints did **not** contain saved loss curves, so direct inspection of training flattening had to be done by re-evaluating checkpoints on a fixed holdout subset.

Checkpoint evaluation used:

- sample: first `512` patches from `holdout_manifest_seed42.txt`
- metrics: weighted elevation RMSE and slope RMSE in degrees from `eval_dem.py`

Observed checkpoint trend on that fixed sample:

- baseline `z_lr`: elev RMSE `2.7988`, slope RMSE deg `3.5065`
- epoch `1`: elev RMSE `2.6829`, slope RMSE deg `3.4772`
- epoch `2`: elev RMSE `2.6322`, slope RMSE deg `3.1568`
- epoch `3`: elev RMSE `2.2202`, slope RMSE deg `2.9994`
- epoch `10`: elev RMSE `2.2093`, slope RMSE deg `2.9099`
- epoch `12`: elev RMSE `2.1395`, slope RMSE deg `2.9388`
- epoch `15`: elev RMSE `2.1412`, slope RMSE deg `2.8697`

What this suggests:

- most of the gain happens early, especially by epoch `3`
- later epochs still help a bit, but returns are much smaller and somewhat noisy
- from epoch `3` to epoch `15`, the sample only improved about:
  - `3.6%` more in elevation RMSE
  - `4.3%` more in slope RMSE deg
- practical takeaway: training appears to flatten fairly early, roughly around epochs `3-6`

Follow-up change made:

- `train_dem.py` now writes loss history into every new checkpoint so future runs can be inspected without re-evaluating model files

## Current File Outputs

Known outputs in repo:

- `dem_film_unet.pt`
- `dem_film_unet_epoch_015.pt` and other per-epoch checkpoints
- `checkpoint_eval_subset_targeted_512.json`
- `eval_holdout_model_15ep.json`
- `eval_holdout_zlr.json`
- `eval_holdout_model_zlr_fabdem.json`
- `smoke_eval_seed42.json`

## Recommended Next Steps

1. Run full-holdout evaluation for a few key checkpoints such as epochs `3`, `10`, and `15` to confirm whether early stopping is justified.
2. Add or run per-zone / per-country summaries for corrected FABDEM to see where it helps most relative to `z_lr`.
3. Evaluate `tdem_edem` once enough patches are downloaded.
4. Once the model pipeline is behaving well, reserve the additional ~`50,000` Australia patches as the "real" holdout and use that set for the stronger final evaluation.
5. Build a final comparison table across:
   - model
   - `z_lr`
   - FABDEM
   - `TDEM_EDEM`

## Notes For Next Chat

If restarting in a fresh chat, mention:

- `status.md` exists and summarizes repo status
- the FABDEM benchmark was fixed by correcting a half-patch grid offset in the comparison exporter
- corrected FABDEM is now a credible baseline and is better than `z_lr` on most holdout metrics, but still worse than the model
- checkpoint analysis suggests training flattens fairly early, around epochs `3-6`
- new training checkpoints now store train/val loss curves
- `eval_dem.py` already supports multi-source evaluation in one pass
- `export_comparison_dtms.py` already exports separate per-product patch rasters from Earth Engine
