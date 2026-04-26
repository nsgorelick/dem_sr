# DEM Project Status

## Overview

Current repository state is the refactored experiment-plan architecture with config-driven train/eval/pretrain entrypoints and consolidated full-data run configs.

Key capabilities now in place:

- local-only training/evaluation from `/data/training`
- deterministic non-AU/AU split manifests
- per-epoch checkpointing and resume support
- config-driven CLI defaults via shared JSON run configs
- experiment-specific CLI/report config export cleanup (no leaked defaults from other experiment plans)
- multi-source evaluation support (`model`, `z_lr`, `stage_a`) in one eval pass

Environment assumptions:

- repo: `/home/gorelick/projects/DEM`
- training data root: `/data/training`
- canonical experiment manifests (training/eval runs):
  - `experiment-runs/manifests/fraction1/train_non_au_full.txt`
  - `experiment-runs/manifests/fraction1/val_au_full.txt`
- canonical export manifests (patch production):
  - `patches/training_manifest.txt`
  - `patches/validation_manifest.txt`
  - `patches/export_manifest_250k.txt`

## Current Experiment Layout

Experiments were split into packages under `experiments/`:

- `baseline`
- `frequency_domain`
- `hydrology`
- `mixture_specialists`
- `two_stage`

Supporting modules:

- `experiments/cli_registration.py`
- `experiments/preset_registry.py`
- `experiments/arg_paths.py`

Entry points:

- `train_experiment.py`
- `eval_experiment.py`
- `pretrain_experiment.py`

Legacy monolithic entrypoints were removed:

- `train_dem.py`
- `eval_dem.py`

## Full-Data Runs (`experiment-runs/full`)

Full-train/full-AU-val configs are consolidated into one directory with one file per run:

- `experiment-runs/full/baseline.json`
- `experiment-runs/full/frequency_domain.json`
- `experiment-runs/full/multitask.json`
- `experiment-runs/full/hydrology.json`
- `experiment-runs/full/self_supervised_pretraining.json`

Conventions:

- configs reference shared fraction-1 manifests
- run artifacts live under `experiment-runs/full/<run-name>/...`
- standardized eval output goes to `experiment-runs/full/results/<config_stem>_eval_results_<description-slug>.json` when `eval.output_json` is `null`
- only `baseline.json` evaluates `z_lr`; the other full configs evaluate `model` only

Execution helper:

- `experiment-runs/full/run.sh` runs the full sequence in order
- script is normalized to Unix line endings and includes `set -euo pipefail`
- script changes to repo root before invoking Python entrypoints

## Data Split Protocol

Active protocol for new comparisons:

- train split: 200k total = 100k US + 100k non-US
- validation split: 50k AU
- year constraint: include only patches with `year >= 2017`
- selection pipeline: `patches/build_training_validation_draws.py`
- canonical outputs:
  - `patches/training.geojson`
  - `patches/validation.geojson`
  - `patches/export_manifest_250k.txt`

Old random holdout metrics are legacy and should not be used for current model ranking.

## Testing and Stability

Test suite status at last update:

- `~/venv-cu128/bin/python -m unittest discover -s . -p 'test_*.py'`
- result: `Ran 80 tests ... OK`

## Repo Hygiene Changes

- `.gitattributes` added with `*.sh text eol=lf` to prevent CRLF shell-script argument issues
- `.gitignore` updated to:
  - keep run-config JSONs under `experiment-runs/` tracked
  - keep generated eval/train JSON artifacts ignored by default
  - ignore local `200k.geojson` and root `er` symlink

## Immediate Next Steps

1. Continue export from `patches/export_manifest_250k.txt` using manifest-driven `export_patches_gcs.py`.
2. Monitor and triage export skips (e.g., masked/empty source areas) separately from successful patch exports.
3. Run/compare full experiments using the canonical experiment manifests under `experiment-runs/manifests/fraction1/`.
4. Record final ranking and promote any tuned follow-up configs as separate named JSON files under `experiment-runs/full/`.

## Patch Table + Export Workflow (Apr 2026)

- patch table downloader now supports:
  - parallel downloads (`patches/download_patch_tables_compute_features.py --workers N`)
  - restart-safe writes via `*.part` + atomic rename
  - both `*_sites_*` and legacy `us_sites4_*` table naming
- draw/normalization pipeline is scripted in:
  - `patches/build_training_validation_draws.py`
- current canonical draw outputs (with `year >= 2017`) are:
  - `patches/training.geojson` (200k = 100k US + 100k non-US)
  - `patches/validation.geojson` (50k AU)
  - `patches/export_manifest_250k.txt` (combined IDs)
- exporter changed from EE table fetch to manifest-driven selection:
  - `export_patches_gcs.py --manifest ...`
  - `export_patches_gcs_512.py --manifest ...`
- manifest ID parsing reconstructs patch-center coordinates (`+0.5`) so exported boxes match source patch geometry.

## Exporter Fixes (Apr 2026)

- fixed 512-export georeferencing bug:
  - patch-grid stride is now fixed at `1280m` (`PATCH_GRID_STRIDE_M`) regardless of output patch size
  - this keeps `patches_512` centers aligned with base patches while expanding only extent
- added `--http-pool-size` passthrough to `export_patches_gcs_512.py` to match `export_patches_gcs.py`
- added startup request pacing to adaptive export runner:
  - `AdaptiveThreadExportRunner(..., startup_spread_sec=...)`
  - `export_patches_gcs.py` now spreads initial submissions across `5s` to avoid startup request bursts
- validated current `data/training/patches_512` outputs against base `data/training`:
  - matching CRS + centers
  - `512x512` windows are `4x` extent of `128x128` windows at same 10m resolution
  - AE files are no longer tiny constant placeholders
