# Full-data experiment configs

All full-train / full-AU-val pilots share the same fraction-1 manifests:

- `experiment-runs/manifests/fraction1/train_non_au_full.txt`
- `experiment-runs/manifests/fraction1/val_au_full.txt`

Configs live in this directory (one JSON per run). Checkpoints, train reports, and per-run eval sidecars live under a subdirectory named after the run (for example `baseline/checkpoints/`).

Eval runs `z_lr` only for **`baseline.json`**, so the low-res baseline is measured once; other configs evaluate **`model`** only on the same val manifest.

When `eval.output_json` is `null`, standardized eval summaries are written to:

- `experiment-runs/full/results/<config_stem>_eval_results_<description-slug>.json`

## baseline

```bash
python3 train_experiment.py --config experiment-runs/full/baseline.json
python3 eval_experiment.py --config experiment-runs/full/baseline.json
```

## frequency_domain (plan 03)

```bash
python3 train_experiment.py --config experiment-runs/full/frequency_domain.json
python3 eval_experiment.py --config experiment-runs/full/frequency_domain.json
```

Monitor band health: `band_share_low`, `band_share_mid`, `band_share_high`, `band_balance_penalty`, `band_high_tv`, `band_high_l2`, `decomp_recon_err`, `pred_recon_err`.

## self_supervised_pretraining (plan 07)

```bash
python3 pretrain_experiment.py --config experiment-runs/full/self_supervised_pretraining.json
python3 train_experiment.py --config experiment-runs/full/self_supervised_pretraining.json
python3 eval_experiment.py --config experiment-runs/full/self_supervised_pretraining.json
```

## multitask (plan 09)

Preset `multitask`, otherwise same manifests as baseline.

```bash
python3 train_experiment.py --config experiment-runs/full/multitask.json
python3 eval_experiment.py --config experiment-runs/full/multitask.json
```

## hydrology (plan 10)

Hydrology loss terms enabled with conservative weights. After a stable run, you can resume and ramp `lambda_hydro_flow` (for example `0.01 → 0.02 → 0.04`) and `lambda_hydro_pit_spike` (`0.005 → 0.01 → 0.02`) while keeping elevation and slope terms primary.

```bash
python3 train_experiment.py --config experiment-runs/full/hydrology.json
python3 eval_experiment.py --config experiment-runs/full/hydrology.json
```
