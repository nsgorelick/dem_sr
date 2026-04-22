# Plan 03 Frequency-Domain (Full/Full)

This run directory is the initial full-data pilot for frequency-domain multi-band residual learning.

- experiment: `frequency_domain`
- train: full non-AU set (`HARD_FRACTION=1.0` equivalent)
- eval: full AU val set (`VAL_HARD_FRACTION=1.0` equivalent)
- epochs: `6`

## 1) Use canonical fraction-1 manifests

This run uses shared reusable manifests:

- `experiment-runs/manifests/fraction1/train_non_au_full.txt`
- `experiment-runs/manifests/fraction1/val_au_full.txt`

## 2) Run training

```bash
python3 train_experiment.py --config experiment-runs/03_frequency_domain_full_full/run_config.json
```

## 3) Run evaluation

```bash
python3 eval_experiment.py --config experiment-runs/03_frequency_domain_full_full/run_config.json
```

Since `eval.output_json` is `null`, the standardized eval result JSON is written under:

- `experiment-runs/03_frequency_domain_full_full/results/`

## 4) Monitoring recommendations

Watch these train metrics for band health and collapse:

- `band_share_low`, `band_share_mid`, `band_share_high`
- `band_balance_penalty`
- `band_high_tv`, `band_high_l2`
- `decomp_recon_err`, `pred_recon_err`
