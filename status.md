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
- selectable model architectures (`--arch`) and training resume (`--resume`) for architecture experiments

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
- `--arch` selects the model: `film_unet` (default, FiLM dual-encoder U-Net), `gated_unet` (spatial gated AE fusion at S1–S3), `xattn_unet` (FiLM at S1; windowed DEM→AE cross-attention at S2–S3), `hybrid_tf_unet` (FiLM at S1–S3; windowed self-attention + FFN bottleneck at S3), or `rcan_ae_unet` (RCAN-style residual channel-attention trunk with AE conditioning)
- `--resume PATH` continues from a checkpoint written by this script (restores model, optimizer, scaler, history; next epoch is `saved_epoch + 1`). Architecture must match the checkpoint (`--arch`).
- `tqdm` progress bar
- per-epoch checkpoints like `dem_film_unet_epoch_015.pt`
- optional random `90/180/270` rotations and horizontal/vertical flips for training samples via `--augment-rotflip` (default enabled)
- checkpoints now include loss history:
  - `history["train_loss"]`
  - `history["train_elev_loss"]`
  - `history["train_slope_loss"]`
  - `history["val_loss"]`
  - `history["val_elev_loss"]`
  - `history["val_slope_loss"]`
  - `history["epoch_seconds"]`
  - `history["samples_per_second"]`
  - `train_loss_curve`
  - `val_loss_curve`
- checkpoints also include:
  - optimizer state
  - AMP scaler state
  - `train_size`
  - `val_size`
- final checkpoint `dem_film_unet.pt`
- modern AMP usage

### `dem_film_unet.py`

- `DemFilmUNet`: residual dual-encoder U-Net with global FiLM on AE features (design doc v1).
- `DemGatedFusionUNet`: same backbone; replaces FiLM with **spatial gated fusion** at S1–S3: AE features projected to DEM width, combined with DEM features and a **trust stack** (uncertainty + masks from `x_dem`) for per-pixel gating; output remains `z_lr + clamped residual`.
- `DemCrossAttnFusionUNet`: FiLM at S1; **windowed cross-attention** (DEM queries, AE keys/values) in 8×8 windows at S2 and S3, then gated residual with the trust stack (Plan B2-style).
- `DemHybridTransformerUNet`: same FiLM fusion as baseline; after S3 FiLM, **two** `BottleneckTransformerBlock` stacks (8×8 windowed self-attention + conv FFN, residual), then U-Net decode (DSRT-style global context at coarse scale).
- `DemRCANAE`: RCAN-style model with channel-attention residual groups at full resolution, AE channel-gated conditioning, and residual output head (`z_lr + clamped residual`).
- `create_model(arch)` factory; `ARCH_FILM` / `ARCH_GATED` / `ARCH_XATTN` / `ARCH_HYBRID_TF` / `ARCH_RCAN_AE` constants.

### `plan.md`

- Tracks architecture exploration runs (screening protocol, example train/eval commands, TODO list) without changing training data or chip size.

### `eval_dem.py`

Current behavior:

- `--arch` overrides model class when loading `--checkpoint` (optional; defaults to `arch` stored in checkpoint `args`, else `film_unet`)
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
- optional per-patch JSON output via `--per-patch-json`
- computes per-patch customer-example fields including:
  - model vs `z_lr` deltas / percentage improvements
  - model vs FABDEM deltas / percentage improvements
  - `customer_example_score` for ranking
- when `--prediction-source raster` is used, checks candidate raster availability on disk up front and skips stems that do not have the requested candidate file present

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

### `upload_customer_example_predictions.py`

New script created.

Current behavior:

- reads a manifest of selected patch stems
- runs model inference and writes prediction GeoTIFFs for those stems
- uploads TIFFs to GCS
- submits Earth Engine ingestions into:
  - `users/ngorelick/DTM/tmp/customer_example_predictions`
- uses the Earth Engine Python API for asset listing / ingestion submission rather than relying on the `earthengine` CLI
- strips non-finite `nodata` values from the ingest path so EE manifest submission succeeds

### `make_customer_example_panels.py`

New script created.

Current behavior:

- generates local `2x4` panel PNGs for selected chips
- panels include:
  - input DEM
  - FABDEM
  - model prediction
  - ground truth
  - hillshade of each of the above
- uses shared robust stretches across the top-row elevation panels and separately across the bottom-row hillshade panels so each chip is visually comparable within a row

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

### Evaluate augmented model + `z_lr` + FABDEM in one pass

```bash
python eval_dem.py \
  --prediction-source model z_lr raster \
  --checkpoint dem_film_unet_aug15.pt \
  --manifest holdout_manifest_seed42.txt \
  --candidate-root /data/comparison \
  --candidate-product fabdem \
  --candidate-band 1 \
  --batch-size 32 \
  --workers 3 \
  --amp \
  --output-json eval_holdout_model_zlr_fabdem_aug15.json
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

From `eval_holdout_model_zlr_fabdem_aug15.json`:

### Model (`dem_film_unet_aug15.pt`)

- patches: `15,052`
- sum(W): `151,686,976.4171`
- elev bias: `+0.1072 m`
- elev MAE: `1.2488 m`
- elev RMSE: `3.2199 m`
- slope MAE: `0.033840`
- slope RMSE: `0.559041`
- slope MAE deg: `1.5131 deg`
- slope RMSE deg: `3.0998 deg`

## What We Learned

### TDEM_EDEM status

The earlier TDEM bias-offset / vertical-datum detour is no longer the active issue after switching to a newer TDEM product.

What matters for current status:

- the old inline datum-correction attempt was backed out
- the newer product removed the previously observed gross bias problem
- `tdem_edem` remains available as an external comparison product in `export_comparison_dtms.py`
- future TDEM evaluation should be treated as a straightforward comparison run against the current downloaded product, not as a current export-pipeline blocker

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

### Customer example workflow

Customer-example selection and packaging is now implemented end to end.

What was added:

- `eval_dem.py` can emit full per-patch metrics and ranking fields
- a selected customer-example manifest was created in `customer_example_chips_manifest.txt`
- the current manifest contains `23` chips:
  - original top set
  - one Portugal example
  - ten additional chips added for broader coverage
- model prediction TIFFs can be generated and uploaded to Earth Engine for any selected manifest via `upload_customer_example_predictions.py`
- local `2x4` panel PNGs can be rendered via `make_customer_example_panels.py`

Current status:

- prediction TIFFs exist in `customer_example_predictions/tifs`
- panel PNGs exist in `customer_example_predictions/panels`
- the added `10` chips were submitted to:
  - `users/ngorelick/DTM/tmp/customer_example_predictions`
- submission metadata for that latest batch is recorded in:
  - `customer_example_predictions/submitted_ingestions.json`

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

### Augmented 15-epoch run (`dem_film_unet_aug15.pt`)

Training run settings (latest run):

- batch size: `32`
- workers: `3`
- epochs: `15`
- augmentation: `--augment-rotflip` enabled
- checkpoint prefix: `dem_film_unet_aug15*.pt`

Checkpoint-history summary:

- train size: `135,469`
- val size: `0` (no validation split in this run)
- epoch 1 train loss: `1.4631`
- epoch 15 train loss: `1.0485`
- train loss dropped by about `28.3%` across 15 epochs

Holdout comparison vs prior model (`dem_film_unet.pt`):

- improved:
  - elevation MAE: `1.3380 -> 1.2488 m`
- regressed:
  - elevation RMSE: `3.2045 -> 3.2199 m`
  - slope MAE deg: `1.4098 -> 1.5131 deg`
  - slope RMSE deg: `2.9654 -> 3.0998 deg`
- bias shifted from `-0.4435 m` to `+0.1072 m`

Interpretation:

- simple rot/flip augmentation improved elevation MAE but hurt slope-sensitive metrics and slightly hurt elevation RMSE on the full holdout
- this run is useful evidence, but it does not replace the current best checkpoint for overall multi-metric performance

## Current File Outputs

Known outputs in repo:

- `dem_film_unet.pt`
- `dem_film_unet_epoch_015.pt` and other per-epoch checkpoints
- `dem_film_unet_aug15.pt`
- `dem_film_unet_aug15_epoch_015.pt` and other per-epoch checkpoints
- `checkpoint_eval_subset_targeted_512.json`
- `eval_holdout_model_15ep.json`
- `eval_holdout_zlr.json`
- `eval_holdout_model_zlr_fabdem.json`
- `eval_holdout_model_zlr_fabdem_aug15.json`
- `eval_holdout_model_zlr_fabdem_per_patch.json`
- `smoke_eval_seed42.json`
- `customer_example_chips_manifest.txt`
- `customer_example_added10_manifest.txt`
- `customer_example_predictions/tifs`
- `customer_example_predictions/panels`
- `customer_example_predictions/submitted_ingestions.json`

## Recommended Next Steps

1. Screen alternative architectures (`gated_unet` and later candidates) with short runs and holdout `eval_dem.py`; use `--resume` to continue after worker or environment tweaks.
2. Check Earth Engine task completion for the customer-example asset submissions and verify that all selected chips are available in `users/ngorelick/DTM/tmp/customer_example_predictions`.
3. Run full-holdout evaluation for a few key checkpoints such as epochs `3`, `10`, and `15` to confirm whether early stopping is justified.
4. Run the same augmented training recipe with a small validation split (or equivalent holdout checkpoint sweep) so augmentation effects can be selected by validation/holdout metrics rather than train loss alone.
5. Add or run per-zone / per-country summaries for corrected FABDEM and the model so we can see where gains are concentrated.
6. Evaluate the current `tdem_edem` product as a clean comparison baseline now that the earlier bias-offset issue is no longer the active blocker.
7. Once the model pipeline is behaving well, reserve the additional ~`50,000` Australia patches as the "real" holdout and use that set for the stronger final evaluation.

## Notes For Next Chat

If restarting in a fresh chat, mention:

- `status.md` exists and summarizes repo status
- `train_dem.py` supports `--arch film_unet|gated_unet|xattn_unet|hybrid_tf_unet|rcan_ae_unet` and `--resume` for checkpoint continuation; `eval_dem.py` supports `--arch` when loading checkpoints
- `plan.md` tracks architecture experiments and example commands
- the FABDEM benchmark was fixed by correcting a half-patch grid offset in the comparison exporter
- corrected FABDEM is now a credible baseline and is better than `z_lr` on most holdout metrics, but still worse than the model
- checkpoint analysis suggests training flattens fairly early, around epochs `3-6`
- new training checkpoints now store train/val loss curves
- `eval_dem.py` already supports multi-source evaluation in one pass plus per-patch ranking output for customer-example selection
- `export_comparison_dtms.py` already exports separate per-product patch rasters from Earth Engine
- the current customer-example package includes a `23`-chip manifest, local prediction TIFFs, local panel PNGs, and EE uploads for the selected examples
