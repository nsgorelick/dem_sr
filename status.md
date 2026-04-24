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
- canonical full manifests:
  - `experiment-runs/manifests/fraction1/train_non_au_full.txt`
  - `experiment-runs/manifests/fraction1/val_au_full.txt`

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

- train split: all non-AU patches
- validation split: AU patches

Old random holdout metrics are legacy and should not be used for current model ranking.

## Testing and Stability

Test suite status at last update:

- `python3 -m unittest discover -s . -p 'test_*.py'`
- result: `Ran 80 tests ... OK`

## Repo Hygiene Changes

- `.gitattributes` added with `*.sh text eol=lf` to prevent CRLF shell-script argument issues
- `.gitignore` updated to:
  - keep run-config JSONs under `experiment-runs/` tracked
  - keep generated eval/train JSON artifacts ignored by default
  - ignore local `200k.geojson` and root `er` symlink

## Immediate Next Steps

1. Verify GPU/disk budget for per-epoch checkpoints across all full runs.
2. Launch `experiment-runs/full/run.sh` in a resilient session (`tmux`/scheduler).
3. After completion, compare all model runs against the baseline eval (which includes `z_lr`) on the shared AU manifest.
4. Record final ranking and promote any tuned follow-up configs as separate named JSON files under `experiment-runs/full/`.
