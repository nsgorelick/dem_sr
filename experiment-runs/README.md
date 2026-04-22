# Experiment Runs

This directory is the root for config-driven experiment runs.

## Full-data pilots (`full/`)

Shared fraction-1 manifests and one config file per experiment name; see [`full/README.md`](full/README.md).

```bash
python3 train_experiment.py --config experiment-runs/full/baseline.json
python3 eval_experiment.py --config experiment-runs/full/baseline.json
```

## Other layouts (for example `short/`)

You can still use one subdirectory per run intent with a `run_config.json` inside it, plus checkpoints and reports under that directory.

Use:

```bash
python3 train_experiment.py --config <run_dir>/run_config.json
python3 eval_experiment.py --config <run_dir>/run_config.json
```

When `eval.output_json` is `null`, eval writes a standardized result JSON next to the config:

- beside a `run_config.json`: `<run_dir>/results/run_config_eval_results_<description-slug>.json`
- beside a named config in `full/`: `experiment-runs/full/results/<config_stem>_eval_results_<description-slug>.json`

