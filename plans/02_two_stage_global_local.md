# Plan: Two-Stage Global + Local Residual System (Rewritten)

## 1) Goal (Refined)

Develop a **two-stage residual system** that explicitly separates:

* **Stage A (Global):** low-frequency / large-scale bias correction
* **Stage B (Local):** high-frequency terrain detail refinement

This addresses optimization conflict in single-stage models, where one network must simultaneously learn:

* datum/bias corrections,
* regional terrain shaping,
* fine-scale structure (ridges, drainage, edges).

All predictions remain **residual-based**:

* `z_mid = z_lr + r_coarse`
* `z_hat = z_mid + r_detail`

---

## 2) Core Hypothesis

Separating scale-dependent error modes will:

* improve large-scale bias correction (Stage A),
* preserve capacity for fine terrain structure (Stage B),
* reduce over-smoothing seen in single-stage residual models.

This approach is most beneficial when:

* errors are multi-scale,
* supervision is noisy,
* deterministic models collapse toward averaged solutions.

---

## 3) Key Design Constraints (Non-Negotiable)

* Both stages predict **residuals only** (never absolute DEM).
* Residual caps applied to both heads.
* Full masking/weighting (`W`) reused from baseline.
* Uncertainty and mask channels included in both stages.
* Maintain evaluation parity with baseline metrics and splits.

---

## 4) Critical Design Safeguards

### 4.1 Stage B Must See Both `z_lr` and `z_mid`

Stage B input MUST include:

* original DEM: `z_lr`
* Stage A output: `z_mid`

This allows Stage B to:

* correct Stage A errors,
* fall back to original geometry when needed.

Without this, Stage A errors become unrecoverable.

---

### 4.2 Restrict Stage A Conditioning (Avoid Detail Leakage)

Stage A should focus on **low-frequency structure only**.

Guidelines:

* No AE embeddings in Stage A (preferred), OR
* Only weak/global conditioning (e.g., low-strength FiLM)

Stage B handles:

* AE-driven detail,
* semantic refinement.

---

### 4.3 Enforce Frequency Separation Explicitly

Do NOT rely on implicit learning.

Use one of:

**Option A (recommended):**

* Train Stage A on **downsampled targets**
* Upsample predictions to full resolution

**Option B:**

* Low-pass filter residual for Stage A target
* High-pass residual for Stage B target

Goal: make decomposition **identifiable**, not emergent.

---

## 5) Architecture

### 5.1 Stage A (Global Model)

* Operates at lower resolution or low-pass space
* Inputs:

  * `Z_lr`
  * `U_enc`
  * masks (`M_bld`, `M_wp`, `M_ws`)
* Output:

  * `r_coarse`

Design priorities:

* smoothness
* stability
* low-frequency accuracy

---

### 5.2 Stage B (Local Model)

* Operates at full resolution

* Inputs:

  * `z_lr`
  * `z_mid`
  * `U_enc`
  * masks
  * `E_ae`

* Output:

  * `r_detail`

Design priorities:

* slope fidelity
* curvature/detail recovery
* edge precision

---

## 6) Loss Design (Explicit Separation)

### 6.1 Stage A Loss

Focus on low-frequency structure:

* elevation loss (weighted)
* slope loss at **downsampled resolution**
* minimal or no high-frequency penalties

---

### 6.2 Stage B Loss

Focus on high-frequency detail:

* elevation residual correction
* strong slope loss (full resolution)
* curvature / Laplacian loss

Optional:

* emphasize residual relative to Stage A output

---

## 7) Training Procedure

### 7.1 Stage A Training

* Train to convergence
* Validate independently

### 7.2 Stage B Training

* Freeze Stage A
* Train Stage B using `z_mid`

### 7.3 Joint Finetuning (Optional)

Only if justified by metrics

Risk:

* may collapse separation

---

## 8) Inference Pipeline

1. Run Stage A → `r_coarse`
2. Compute `z_mid = z_lr + r_coarse`
3. Run Stage B → `r_detail`
4. Output `z_hat = z_mid + r_detail`

Provide fallback:

* Stage A-only output

---

## 9) Evaluation Plan

### 9.1 Compare

* Stage A only
* Stage A + B
* Best single-stage baseline

### 9.2 Metrics

* elevation RMSE / MAE
* slope RMSE
* curvature / Laplacian RMSE

### 9.3 Stratification (Critical)

* high-slope terrain
* near-water
* building-mask regions

### 9.4 Diagnostics

* visual ridge/valley sharpness
* artifact detection
* seam stability

---

## 10) Pilot Success Criteria

Must satisfy ALL:

* no regression in elevation RMSE
* improved slope RMSE in high-relief strata
* improved curvature/detail without speckle
* no increase in artifacts near masks

---

## 11) Risks

### Error Propagation

* Stage A errors may bias Stage B
* mitigated by including `z_lr` in Stage B input

### Collapse of Scale Separation

* Stage A may learn high-frequency detail
* mitigated by enforced frequency separation

### Complexity

* multi-stage training and inference
* checkpoint coordination required

---

## 12) Position in Pipeline

This is a:

**structured decomposition of deterministic residual learning**

It is intended to outperform single-stage models by:

* isolating bias correction,
* preserving capacity for local terrain structure.

---

## 13) Summary

This plan reframes the problem as:

> Explicit multi-scale residual decomposition, with enforced frequency separation and recoverable intermediate predictions.

Success depends on maintaining strict separation between global and local roles.

