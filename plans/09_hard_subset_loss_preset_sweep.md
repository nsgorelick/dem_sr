# Plan: Contour-Aware Loss Sweep (Config-Driven)

## 1) Goal

Run contour-aware supervision experiments from `contours.md` using the new shared config workflow (`train_experiment.py` + `eval_experiment.py`) with outputs that are easy to identify later.

This plan supports two modes:

- **Screening mode**: hard subsets for fast ranking (`HARD_FRACTION=0.10`, `VAL_HARD_FRACTION=0.10` behavior).
- **Confirmation mode**: full manifests (`HARD_FRACTION=1.0`, `VAL_HARD_FRACTION=1.0` behavior).

Primary question:

> Which contour-aware loss preset best improves terrain structure (slope/gradient/curvature/contour alignment) without harming elevation error?

All runs should be driven by a run config file with a human-readable `description`.

---

## 2) Contour Hypotheses

### H1: Geometry terms help on hard terrain

Compared with `baseline`, `geom` should reduce:

- slope error,
- gradient error (`grad_x`, `grad_y`),
- Laplacian/curvature error.

### H2: Contour supervision helps contour alignment

`contour` and `multitask` should improve:

- `sdf_rmse_w` (distance-to-contour consistency),
- and ideally retain or improve slope/gradient behavior.

### H3: Hard subsets increase short-run sensitivity

Restricting train/val to hard patches (where `z_lr` is poor) should expose contour/geometry loss differences faster than full-manifest short runs.

---

## 3) Fixed Screening Protocol (Default)

Use:

- `EPOCHS=3`
- `PRESETS="baseline geom multitask contour"` where:
  - `baseline`: elevation + slope reference loss
  - `geom`: baseline + gradient + Laplacian + multi-scale elevation
  - `contour`: baseline + contour SDF
  - `multitask`: `geom` + contour SDF + soft contour indicator
- train manifest = hardest 10% non-AU
- val manifest = hardest 10% AU
- `HARD_SCORE_FIELD=resid_scale` (alias-aware with `residAbs_p95`)

The same hard-val manifest must be used for:

- `z_lr` baseline eval,
- all four model preset evals.

`contour_interval` must be identical between train and eval within a run.

---

## 4) Hard-Patch Selection Policy

`select_hard_patches.py` performs:

1. Read source manifest + patch table.
2. Apply quality filters:
   - `mean_W >= 0.4`
   - `valid_frac >= 0.8`
   - `gt_coverage_mean >= 0.8`
   - `relief >= 0.5`
   - `frac_water <= 0.25`
3. Rank eligible stems by score field (default `resid_scale` / `residAbs_p95`) descending.
4. Keep top `fraction` with deterministic tie-breaking.
5. Write output manifest + summary JSON (cutoff and counts).

---

## 5) Config Workflow

Use config-driven entrypoints:

1. Build full manifests via `make_manifest.py` (train=non-AU, val=AU).
2. If screening mode, produce hard train/val subsets via `select_hard_patches.py`.
3. Point config `shared.manifest` / `eval.manifest` to selected manifest files.
4. Run:
   - `python3 train_experiment.py --config <run_config.json>`
   - `python3 eval_experiment.py --config <run_config.json>`

Outputs:

- train artifacts use configured paths (checkpoint + train report JSON).
- eval writes standardized results JSON in the config directory when `eval.output_json` is null.
- optional per-patch / stratified JSONs should also be configured into the same config directory.

---

## 6) Canonical Commands

```bash
# 1) Build non-AU train and AU val manifests
python3 make_manifest.py \
  --data-root /data/training \
  --patch-table 200k.geojson \
  --locked-country AU \
  --val-fraction 0 \
  --holdout-out <val_manifest.txt> \
  --train-out <train_manifest.txt> \
  --summary-json <summary.json>
```

```bash
# 2a) Screening mode: select hard subsets (optional)
python3 select_hard_patches.py --manifest <train_manifest.txt> --patch-table 200k.geojson --fraction 0.10 --score-field resid_scale --out <train_hard.txt>
python3 select_hard_patches.py --manifest <val_manifest.txt> --patch-table 200k.geojson --fraction 0.10 --score-field resid_scale --out <val_hard.txt>
```

```bash
# 2b) Confirmation mode: use full manifests directly (equivalent to HARD_FRACTION=1.0, VAL_HARD_FRACTION=1.0)
```

```bash
# 3) Run train/eval from shared config
python3 train_experiment.py --config <run_config.json>
python3 eval_experiment.py --config <run_config.json>
```

Optional knobs:

- `HARD_FRACTION` / `VAL_HARD_FRACTION` (set either to `1.0` for full set)
- `HARD_SCORE_FIELD` (`resid_scale`, `p90_slope`, `relief`, `mean_uncert`)
- `EPOCHS`, `BATCH_SIZE`, `WORKERS`, `CONTOUR_INTERVAL`

---

## 7) Contour-Centric Ranking Metrics

Primary ranking order for this contour experiment:

1. `slope_rmse_deg_w`
2. `sdf_rmse_w`
3. `grad_x_rmse_w` + `grad_y_rmse_w`
4. `laplacian_rmse_w`
5. `elev_rmse_w`
6. `elev_mae_w` (tie-break)

Notes:

- We intentionally prioritize structural and contour metrics first.
- `elev_rmse_w` is still a hard guardrail against regressions.

---

## 8) Interpretation Guardrails

- Hard-val absolute numbers are expected to be worse than full AU numbers.
- Only **within-run preset ranking** is considered trustworthy.
- Do not compare hard-val metrics directly with historical full-val tables.
- Contour gains that come with severe elevation regression are not promotable.

---

## 9) Promotion Gate (Required)

Promote only the winning preset(s) to full validation:

1. Re-run with full AU val manifest.
2. Preferably also re-run with full non-AU train manifest and longer training (`epochs>=6`).

A preset is accepted only if full-AU validation confirms:

- improved contour/structure metrics (`sdf`, slope, gradient, Laplacian),
- and no unacceptable elevation RMSE regression.

---

## 10) Expected Cost Reduction

Relative to previous full-manifest, longer-epoch preset sweeps, this setup gives an order-of-magnitude runtime reduction by shrinking both train and val pools while retaining hard-case pressure.

Use this mode as the default **screening stage** before any expensive full-AU confirmation run.

---

## 11) Risks

### Selection Bias

Hard subsets may over-emphasize specific terrain/error modes.

Mitigation:

- final full-AU re-check is mandatory.

### Score-Field Sensitivity

`resid_scale` may favor one type of difficulty.

Mitigation:

- rerun small ablations with `HARD_SCORE_FIELD=p90_slope` if ranking looks unstable.

### Overfitting to Hard Slice

Very short runs on very hard slices can overfit idiosyncrasies.

Mitigation:

- confirm top candidate with larger fractions and/or extra epochs before conclusions.

---

## 12) Summary

This plan formalizes a config-first contour-focused workflow:

> Stress-test contour-aware losses on hard patches, rank by structure + contour fidelity, then validate the winner on full AU using standardized config-scoped outputs.

