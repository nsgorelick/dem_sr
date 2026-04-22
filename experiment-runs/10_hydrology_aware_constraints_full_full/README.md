# Plan 10 Hydrology-Aware Constraints (Full/Full)

This run directory is the initial full-data pilot for hydrology-aware supervision.

- base model/loss path: existing baseline architecture
- train: full non-AU set (`HARD_FRACTION=1.0` equivalent)
- eval: full AU val set (`VAL_HARD_FRACTION=1.0` equivalent)
- epochs: `6`
- hydrology terms: enabled with conservative initial weights

## 1) Use canonical fraction-1 manifests

This run uses shared reusable manifests:

- `experiment-runs/manifests/fraction1/train_non_au_full.txt`
- `experiment-runs/manifests/fraction1/val_au_full.txt`

## 2) Run training

```bash
python3 train_experiment.py --config experiment-runs/10_hydrology_aware_constraints_full_full/run_config.json
```

## 3) Run evaluation

```bash
python3 eval_experiment.py --config experiment-runs/10_hydrology_aware_constraints_full_full/run_config.json
```

Since `eval.output_json` is `null`, the standardized eval result JSON is written under:

- `experiment-runs/10_hydrology_aware_constraints_full_full/results/`

## 4) Optional ramp-up after stable baseline

After initial stable training, increase hydrology terms gradually by resuming from checkpoint:

- increase `lambda_hydro_flow` (e.g. `0.01 -> 0.02 -> 0.04`)
- increase `lambda_hydro_pit_spike` (e.g. `0.005 -> 0.01 -> 0.02`)
- keep elevation/slope terms primary (`lambda_elev`, `lambda_slope`)
