# Plan: Contour-Aware Hard-Subset Sweep

## 1) Goal

Rapidly screen **contour-aware supervision** formulations from `contours.md` with a small, high-signal budget:

- 4 loss presets (`baseline`, `geom`, `multitask`, `contour`)
- 3 epochs each
- hardest 10% non-AU train subset
- hardest 10% AU validation subset

Primary question:

> Which contour-aware loss preset best improves terrain structure (slope/gradient/curvature/contour alignment) without harming elevation error?

This is a **ranking experiment**, not a final absolute-metric benchmark.

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
- `HARD_FRACTION=0.10`
- `VAL_HARD_FRACTION=0.10`
- `HARD_SCORE_FIELD=resid_scale` (alias-aware with `residAbs_p95`)

The same hard-val manifest must be used for:

- `z_lr` baseline eval,
- all four model preset evals.

`CONTOUR_INTERVAL` must be identical between train and eval within a run.

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

## 5) Runner Workflow

`run_loss_presets_non_au_vs_au.sh` now:

1. Builds full manifests via `make_manifest.py` (train=non-AU, val=AU).
2. Produces hard train subset (`HARD_FRACTION`).
3. Produces hard val subset (`VAL_HARD_FRACTION`).
4. Evaluates `z_lr` on selected val subset.
5. Trains/evaluates each preset on the same selected subsets.

Outputs are written under:

- `runs/<RUN_NAME>/manifests/`
- `runs/<RUN_NAME>/checkpoints/`
- `runs/<RUN_NAME>/eval/`

---

## 6) Canonical Command

```bash
PATCH_TABLE=200k.geojson \
  ./run_loss_presets_non_au_vs_au.sh
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

1. Re-run with:
   - `VAL_HARD_FRACTION=1.0`
2. Preferably also:
   - `HARD_FRACTION=1.0` and longer training (`EPOCHS>=6`) for confirmation.

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

This plan formalizes a fast contour-focused screening loop:

> Stress-test contour-aware losses on hard patches, rank by structure + contour fidelity, then validate the winner on full AU.

