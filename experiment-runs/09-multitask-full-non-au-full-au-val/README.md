# Plan 09 Full-Set Multitask Retest

This run directory is for the confirmation configuration:

- preset: `multitask`
- train: full non-AU set (`HARD_FRACTION=1.0` equivalent)
- eval: full AU val set (`VAL_HARD_FRACTION=1.0` equivalent)

## 1) Use canonical fraction-1 manifests

This run uses shared reusable manifests:

- `experiment-runs/manifests/fraction1/train_non_au_full.txt`
- `experiment-runs/manifests/fraction1/val_au_full.txt`

## 2) Run training

```bash
python3 train_experiment.py --config experiment-runs/09-multitask-full-non-au-full-au-val/run_config.json
```

## 3) Run evaluation

```bash
python3 eval_experiment.py --config experiment-runs/09-multitask-full-non-au-full-au-val/run_config.json
```

Since `eval.output_json` is `null`, the standardized eval result JSON is written under:

- `experiment-runs/09-multitask-full-non-au-full-au-val/results/`

