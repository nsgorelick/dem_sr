# Plan: Mixture-of-Specialists with Terrain Router (Rewritten)

## 1) Goal (Refined)

Develop a **mixture-of-specialists residual system** that improves performance in terrain regimes where a single global model is forced to compromise.

Target regimes may include:

* steep terrain
* hydrologic / riparian regions
* urban / built regions
* flat or low-relief terrain
* high-uncertainty regions

The objective is to improve **worst-performing strata** without harming global performance.

All outputs remain residual-based:

* expert `k` predicts `r_k`
* mixture prediction: `r_mix = Σ_k π_k * r_k`
* final output: `z_hat = z_lr + r_mix`

---

## 2) Core Hypothesis

A single model underperforms because different terrain regimes require different correction behaviors.

A specialist system can help when:

* failure modes cluster by terrain type,
* those regimes are detectable from inputs,
* experts are explicitly encouraged to differentiate.

This plan is justified only if baseline errors show clear regime-specific structure.

---

## 3) Key Design Constraints (Non-Negotiable)

* Every expert predicts a **residual**, never absolute DEM.
* Residual clamping is applied per-expert and/or after mixture aggregation.
* Use the same masking/weighting (`W`) as baseline.
* Keep uncertainty and trust conditioning available to all experts.
* Maintain evaluation parity with baseline metrics and splits.

---

## 4) First Implementation: Shared Backbone + Specialist Heads

Do NOT start with fully separate expert models.

First pilot architecture:

* shared encoder / backbone
* `K` lightweight specialist residual heads
* learned router producing mixture weights `π`

Why:

* reduces data fragmentation
* improves stability
* makes specialization easier to diagnose
* lowers operational complexity

Only move to fully separate experts if shared-backbone specialization proves insufficient.

---

## 5) Routing Strategy

### 5.1 Soft Routing First (Required)

Use **soft mixture routing** initially:

* `π = softmax(router(...))`
* `r_mix = Σ_k π_k * r_k`

Do NOT start with hard routing.

Reasons:

* avoids seams between experts
* reduces instability
* allows gradual expert differentiation

Hard routing can be tested later only if soft routing shows clear expert separation.

---

### 5.2 Router Inputs

Router inputs must come from **inference-available signals only**.

Allowed inputs:

* `Z_lr`
* uncertainty
* masks
* shallow learned features from backbone
* simple derived features available at inference

Do NOT use target-derived labels or ground-truth-only features at inference time.

---

### 5.3 Optional Hierarchical Routing

If flat routing is unstable, use a hierarchy:

* first router: coarse regime (e.g., flat vs steep)
* second router: subtype (e.g., hydrologic vs non-hydrologic)

Use only if needed for interpretability or calibration.

---

## 6) Specialist Definition

### 6.1 Initial Expert Set

Start small:

* `K = 2` or `K = 3`

Avoid 4+ experts in the first pilot.

Suggested starting regimes:

* low-relief vs high-relief
* hydrologic-sensitive vs general terrain
* uncertain / difficult vs stable terrain

---

### 6.2 Specialist Supervision

Experts should not be assumed to specialize automatically.

Use one or more of:

* biased sampling toward regime-relevant patches
* overlapping regime subsets (not strict disjoint splits)
* curriculum from global baseline initialization
* weak diversity encouragement between expert outputs

Goal: prevent expert collapse.

---

## 7) Preventing Expert Collapse

This is the main technical risk.

Mitigations:

### 7.1 Initialization

* initialize shared backbone from best global model
* initialize expert heads from a common residual head or lightly perturbed copies

### 7.2 Routing Regularization

* encourage non-degenerate router usage
* monitor entropy / expert utilization

### 7.3 Diversity Pressure

Optional:

* weak regularization encouraging experts to differ in output behavior
* only if experts collapse to near-identical predictions

### 7.4 Balanced Exposure

* ensure each expert sees sufficient regime-relevant data
* set minimum-sample thresholds before enabling an expert

---

## 8) Training Procedure

### 8.1 Stage 0: Baseline Analysis

Before training MoE:

* identify worst-performing strata from current best model
* confirm regime-specific error clustering

Only proceed if specialist structure is supported by evidence.

---

### 8.2 Stage 1: Shared Backbone Initialization

* start from best global checkpoint

### 8.3 Stage 2: Train Specialist Heads + Router

* keep backbone fixed or lightly finetuned initially
* train router and expert heads together
* use soft routing

### 8.4 Stage 3: Optional Full Finetuning

* unfreeze more layers only if stable
* monitor collapse and global regression carefully

---

## 9) Inference

At inference:

1. backbone computes shared features
2. each expert head predicts residual `r_k`
3. router predicts weights `π_k`
4. aggregate residuals to produce `r_mix`
5. output `z_hat = z_lr + r_mix`

This avoids checkpoint dispatch complexity in the first pilot.

---

## 10) Evaluation Plan

### 10.1 Required Comparisons

Compare against:

* best global baseline
* same architecture with a single residual head
* same mixture architecture with **uniform expert weights** (no real routing)

This isolates whether routing is actually helping.

---

### 10.2 Metrics

* elevation RMSE / MAE
* slope RMSE
* curvature / Laplacian RMSE

### 10.3 Stratum-Level Metrics (Primary)

Report first-class metrics for:

* worst baseline strata
* high-slope terrain
* hydrologic-sensitive areas
* urban / masked-adjacent regions
* high-uncertainty areas

Global average alone is not sufficient.

---

### 10.4 Router Diagnostics

Monitor:

* expert utilization frequency
* router entropy
* calibration: router confidence vs realized error
* spatial smoothness of expert weights

---

## 11) Pilot Success Criteria

Must satisfy ALL:

* materially improve worst-performing strata
* no meaningful regression in global elevation RMSE
* router uses multiple experts meaningfully
* experts show non-trivial behavioral separation
* no visible routing seams or instability

---

## 12) Risks

### Expert Collapse

* experts converge to same solution
* mitigated by routing regularization, curriculum, and biased exposure

### Weak Routing Features

* router learns shallow shortcuts or unstable assignments
* mitigated by restricting to robust inference-available inputs

### Data Fragmentation

* too many experts reduce sample efficiency
* mitigated by shared backbone and small `K`

### Operational Complexity

* mitigated in first pilot by single shared model with multiple heads

---

## 13) Position in Pipeline

This is a:

**regime-aware extension of deterministic residual modeling**

It is designed to solve statistical heterogeneity across terrain types, not scale separation or spatial-context limitations.

---

## 14) Summary

This plan reframes the problem as:

> Learn a small set of specialized residual behaviors and blend them smoothly using a terrain-aware router.

Success depends on proving that:

* terrain-specific error modes are real,
* routing is based on meaningful signals,
* experts actually specialize rather than duplicate one another.

