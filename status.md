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

Split policy has changed from the old seed-42 random holdout.

Current target protocol:

- training set: all non-AU patches
- validation set: AU patches

Implications:

- old random train/holdout metrics are deprecated for model selection
- manifests should be regenerated to reflect the non-AU/AU split before new training runs
- architecture screening and benchmark comparisons should use this split consistently

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

### `make_manifest.py`

Current behavior:

- keeps the old random split mode for simple local manifest generation
- now also supports a table-driven mode via a patch-summary CSV / JSON / GeoJSON
- `--patch-table` + `--locked-country AU` can generate:
  - a locked final-test manifest from eligible Australia rows
  - a development-train manifest from the remaining local pool
  - an optional development-val manifest via `--val-out`
- locked final-test eligibility is filtered by patch-table quality fields:
  - `mean_W` / `weight_mean`
  - `valid_frac` / `weight_valid_mean`
  - `gt_coverage_mean`
  - `relief`
  - `frac_water`
- summary JSON now records the locked-country pool, filtered final test, and development pool separately when table-driven mode is used

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
- optional patch-table join via `--patch-table`
- optional stratified summary JSON via `--stratified-json`
- computes per-patch customer-example fields including:
  - model vs `z_lr` deltas / percentage improvements
  - model vs FABDEM deltas / percentage improvements
  - `customer_example_score` for ranking
- when a patch-summary table is supplied, per-patch rows can now be enriched with canonical table fields:
  - `p90_slope`
  - `frac_shore`
  - `frac_water`
  - `has_edge`
  - `frac_building`
  - `mean_uncert`
  - `mean_W`
  - `valid_frac`
  - `gt_coverage_mean`
  - `resid_scale`
  - `relief`
  - `stratum_id`
- first-pass Australia stratified summaries are now available for:
  - `slope_bin`
  - `hydrology_bin`
  - `building_bin`
  - `uncertainty_bin`
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

The commands below are legacy examples from the previous holdout workflow and should be updated to the new non-AU/AU manifests before use.

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

## Performance Tracking Reset

Performance reporting in this file has been intentionally reset.

Reason:

- prior metrics were based on the old random patch-level holdout (`holdout_manifest_seed42.txt`)
- the project is now switching to a new split protocol and full architecture re-screening
- old numbers are no longer the authoritative comparison baseline

New protocol to use going forward:

- train split: all **non-AU** patches
- validation split: **AU** patches
- model comparison set: all supported `--arch` options under the same split and run budget

Status:

- keep old eval JSON artifacts for traceability, but do not use them as current decision metrics
- regenerate model/baseline comparisons and architecture rankings under the new split
- repopulate this section only after those reruns complete

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
- `eval_arch_gated.json`
- `eval_arch_xattn.json`
- `eval_arch_hybrid_tf.json`
- `eval_arch_rcan_ae.json`
- `eval_arch_gated_epoch003.json`
- `eval_arch_xattn_epoch003.json`
- `eval_arch_hybrid_tf_epoch003.json`
- `eval_arch_rcan_ae_epoch003.json`
- `customer_example_chips_manifest.txt`
- `customer_example_added10_manifest.txt`
- `customer_example_predictions/tifs`
- `customer_example_predictions/panels`
- `customer_example_predictions/submitted_ingestions.json`

## Recommended Next Steps

1. Run `run_arch_non_au_vs_au.sh` with `PATCH_TABLE=200k.geojson` to generate manifests and execute the full 6-epoch `--arch` sweep.
2. Confirm manifest summary output under `runs/<run_name>/manifests/manifest_summary_non_au_vs_au.json` to verify train=non-AU and val=AU counts.
3. Evaluate all checkpoints with the same eval config and regenerate comparison JSON outputs.
4. Recompute architecture ranking and, if desired, blend candidates only after fresh per-arch metrics exist.
5. Rebuild any per-patch ranking tables needed for customer examples from the new validation outputs.
6. Update this status file with the new split-specific metrics and conclusions.

## Notes For Next Chat

If restarting in a fresh chat, mention:

- `status.md` exists and summarizes repo status
- `train_dem.py` supports `--arch film_unet|gated_unet|xattn_unet|hybrid_tf_unet|rcan_ae_unet` and `--resume` for checkpoint continuation; `eval_dem.py` supports `--arch` and optional two-checkpoint residual blending (`--blend-checkpoint`, `--blend-weight`, `--blend-arch`)
- `plan.md` tracks architecture experiments and example commands
- `run_arch_non_au_vs_au.sh` is the primary sweep script and expects `PATCH_TABLE` (current table: `200k.geojson`)
- performance metrics in this file were intentionally reset and must be regenerated under the new split
- current split target is: train on non-AU patches, validate on AU patches
- new training checkpoints store train/val loss curves
- `eval_dem.py` supports multi-source evaluation in one pass plus per-patch ranking output for customer-example selection
- `export_comparison_dtms.py` already exports separate per-product patch rasters from Earth Engine
- customer-example assets/workflow exist, but any ranking claims should be refreshed from new validation outputs
