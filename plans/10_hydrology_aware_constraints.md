# Plan: Hydrology-Aware Constraints for DEM Refinement

## 1) Goal (Refined)

Develop a **hydrology-aware residual correction model** that improves DEM quality not only in elevation and slope, but also in **drainage consistency and flow topology**.

This plan targets failure modes where small elevation errors create large hydrologic errors, especially in:

* flat terrain
* near-water / shoreline regions
* drainage corridors
* low-relief floodplain structure

Final prediction remains residual-based:

* `z_hat = z_lr + r`

---

## 2) Core Hypothesis

A DEM can look acceptable under elevation RMSE while still failing hydrologically.

Adding hydrology-aware constraints can:

* reduce flow-direction errors
* improve drainage continuity
* suppress hydrologically implausible pits/spikes
* improve topographic consistency in flat and near-water terrain

This is justified if current models show:

* unstable drainage in flat areas
* shoreline / riparian artifacts
* stream-network inconsistency despite acceptable geometry metrics

---

## 3) Key Design Constraints (Non-Negotiable)

* Model predicts **residuals only**, never absolute DEM
* Residual clamping remains in place
* Use the same masking / weighting (`W`) as baseline
* Preserve trust / uncertainty conditioning
* Keep identical baseline geometry metrics and splits
* Hydrology-aware constraints supplement, not replace, elevation/slope supervision

---

## 4) First Pilot Scope (Strict)

Start with **soft hydrologic consistency constraints**, not a full differentiable hydrology engine.

First pilot includes:

* standard elevation + slope supervision
* one differentiable proxy for local flow-direction consistency
* one regularizer discouraging hydrologically implausible micro-pits / spikes in trusted regions

Do NOT start with:

* full basin simulation
* complex stream routing solvers
* hard global topological constraints

---

## 5) Hydrology-Aware Objective Design

### 5.1 Base Losses (Retained)

Keep existing supervised losses:

* weighted elevation loss
* weighted slope loss

---

### 5.2 Flow-Direction Consistency Proxy (Required)

Add a differentiable local proxy encouraging predicted terrain to preserve correct downhill orientation.

Possible formulation:

* compare predicted local descent directions with target-derived local descent directions
* use soft directional weights rather than hard D8 assignments

Goal:

* penalize flow-orientation mismatches
* remain stable under small perturbations

---

### 5.3 Pit / Spike Suppression (Required)

Add a hydrology-aware plausibility penalty that discourages:

* isolated sinks
* isolated spikes
* local roughness patterns that disrupt drainage

Apply only in trusted non-masked regions.

---

### 5.4 Water / Shoreline Handling

Hydrologic constraints near water must respect masking policy:

* persistent water remains masked or weakly constrained
* shoreline regions use reduced-weight hydrology penalties

Avoid forcing incorrect behavior where supervision conventions are inconsistent.

---

## 6) Model Integration Options

This plan is primarily a **loss / constraint upgrade**, not a new architecture.

First pilot should attach to an existing strong deterministic model.

Possible later extensions:

* use with large-context model
* use with variational unrolling
* use with two-stage system

---

## 7) Training Procedure

### 7.1 Start from Existing Best Model Family

* keep architecture fixed
* add hydrology-aware terms to training objective

### 7.2 Loss Scheduling

To avoid destabilizing training:

* start with low weight on hydrology terms
* increase gradually after base geometry is stable

### 7.3 Region Weighting

Hydrology terms may be emphasized more in:

* flat terrain
  - riparian areas
* drainage-sensitive regions

Only if stable and justified by diagnostics.

---

## 8) Evaluation Plan

### 8.1 Standard Metrics (Retained)

* elevation RMSE / MAE
* slope RMSE
* curvature / Laplacian RMSE

---

### 8.2 Hydrology Diagnostics (Primary)

Track:

* local flow-direction disagreement
* stream-network overlap / agreement (if available)
* sink / pit frequency in non-water terrain
* drainage continuity errors

---

### 8.3 Stratified Evaluation (Critical)

Evaluate specifically on:

* flat / low-relief terrain
* near-water / shoreline regions
* high-uncertainty terrain
* AU validation regions

Global averages alone are insufficient.

---

## 9) Baseline Comparisons (Required)

Compare against:

* same architecture without hydrology constraints
* best current deterministic baseline

Optional:

* post-hoc hydrologic smoothing / filling baseline

Purpose:

* distinguish benefit of training-time constraints from simple post-processing

---

## 10) Pilot Success Criteria

Must satisfy ALL:

* reduce hydrologic inconsistency metrics
* reduce pits / implausible drainage artifacts
* no meaningful regression in elevation RMSE
* no new artifacts near water / shoreline masks

---

## 11) Risks

### Objective Instability

* hydrologic proxies may be noisy or non-smooth
* mitigated by soft local proxies and gradual weighting

### Supervision Mismatch

* LiDAR and source DEM may differ in hydro-flattening conventions
* mitigated by masking persistent water and downweighting shorelines

### Overconstraint

* hydrology loss may flatten legitimate terrain detail
* mitigated by keeping elevation/slope losses primary

### Limited Global Awareness

* local flow proxies may not fully capture basin-scale topology
* accepted in first pilot to preserve feasibility

---

## 12) Position in Pipeline

This is a:

**topology-aware supervision upgrade**

It complements geometry-based training by targeting hydrologic plausibility, especially where small geometric errors produce large drainage failures.

---

## 13) Summary

This plan reframes DEM refinement as:

> Predict terrain corrections that are geometrically accurate and locally hydrologically plausible.

Success depends on improving flow-related consistency without destabilizing training or imposing brittle global constraints in the first pilot.

