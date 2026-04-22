# Experiment Runs

This directory is the root for config-driven experiment runs.

Recommended structure:

- one subdirectory per run intent
- each run subdirectory contains:
  - `run_config.json`
  - manifests used by that run
  - checkpoint(s)
  - train/eval output JSON files
  - optional notes

Use:

```bash
python3 train_experiment.py --config <run_dir>/run_config.json
python3 eval_experiment.py --config <run_dir>/run_config.json
```

When `eval.output_json` is `null`, eval writes a standardized result JSON to:

- `<run_dir>/results/<config_stem>_eval_results_<description-slug>.json`

