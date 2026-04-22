# Canonical Fraction-1 Manifests

Reusable full-manifest split for experiments:

- `train_non_au_full.txt` = full non-AU development train set
- `val_au_full.txt` = full AU validation/holdout set

Generated with:

- `data_root`: `/data/training`
- `patch_table`: `200k.geojson`
- locked country: `AU`
- `val_fraction=0` (all eligible AU goes to val)

Summary:

- `manifest_summary_non_au_vs_au_fraction1.json`

Use these manifests in experiment run configs to keep train/eval split fixed across runs.

