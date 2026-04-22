# Plan: Variationally Regularized Residual Unrolling (Rewritten)

## 1) Goal (Refined)

Develop a **residual refinement model based on unrolled optimization**, combining:

* explicit objective terms (fidelity + regularization),
* iterative updates,
* lightweight learnable refinement blocks.

This introduces **structured inductive bias** for terrain plausibility while retaining learnable flexibility.

Final output remains:

* `z_hat = z_lr + r`

---

## 2) Core Hypothesis

Black-box residual models may:

* overfit noise,
* produce curvature artifacts,
* lack control over smoothness vs detail.

Unrolling an explicit objective can:

* stabilize corrections,
* enforce plausible terrain structure,
* improve robustness in weak/noisy supervision regions.

---

## 3) Key Design Constraints (Non-Negotiable)

* Optimize **residual correction**, not absolute DEM
* Residual clamping enforced per iteration or at output
* Use same masking (`W`), uncertainty, and trust conditioning
* Maintain identical evaluation metrics and splits
* Keep objective terms explicit and interpretable

---

## 4) First Pilot Objective (Strictly Defined)

Define residual-space energy:

E(r) =

* **Fidelity term:**

  * weighted error to target (`W * (z_lr + r - z_gt)`)

* **Baseline anchoring term:**

  * discourages unnecessary large corrections

* **Edge-aware smoothness term:**

  * anisotropic regularization (preserve slopes, suppress noise)

* **Curvature control term:**

  * penalize extreme curvature/spikes

This objective must be:

* differentiable
* used consistently across training and analysis

---

## 5) Unrolled Architecture

### 5.1 Iterative Update

Initialize:

* `r_0 = 0`

For k = 1…K:

* compute gradient of objective terms

* apply update step:

  * `r_k = r_{k-1} - α_k * grad(E)`

* apply learnable refinement block (proximal step)

---

### 5.2 Learnable Components

Each iteration includes:

* learnable step size `α_k` (bounded)
* small refinement/proximal network

Constraint:

* refinement block must be lightweight (not full model)

---

### 5.3 Iteration Count

* fixed small K (e.g., 3–5)
* tune K for quality vs cost

---

## 6) Training Procedure

### 6.1 End-to-End Training

* unroll K iterations
* supervise final output

### 6.2 Auxiliary Monitoring

Track per-iteration:

* objective value
* residual magnitude

Ensure:

* monotonic or stable improvement across iterations

---

## 7) Stability Controls (Required)

* gradient clipping
* bounded step sizes
* residual magnitude caps

Prevent:

* divergence
* oscillation

---

## 8) Baseline Comparisons (Critical)

Compare against:

* best feed-forward residual model
* parameter-matched iterative/refinement model (no explicit objective)

Purpose:

* isolate benefit of variational structure vs iteration alone

---

## 9) Evaluation Plan

### 9.1 Metrics

* elevation RMSE / MAE
* slope RMSE
* curvature / Laplacian RMSE

### 9.2 Plausibility Diagnostics (Primary)

* roughness spikes
* curvature outliers
* speckle artifacts

### 9.3 Stratified Evaluation

* high-slope terrain
* high-uncertainty regions
* AU generalization

---

## 10) Pilot Success Criteria

Must satisfy ALL:

* reduced curvature artifacts and speckle
* improved robustness in hard strata
* no regression in elevation RMSE
* stable iteration behavior (no divergence)

---

## 11) Risks

### Objective Mis-specification

* poorly chosen terms give no benefit

### Collapse to Generic Network

* proximal blocks dominate, ignoring explicit objective

### Optimization Instability

* sensitive to step sizes and scaling

### Compute Cost

* multiple iterations increase inference time

---

## 12) Position in Pipeline

This is a:

**structured, regularized alternative to black-box residual learning**

It targets plausibility and robustness rather than raw expressivity.

---

## 13) Summary

This plan reframes the problem as:

> Iteratively optimize a residual correction under explicit terrain-aware constraints.

Success depends on whether the defined objective meaningfully captures terrain structure better than unconstrained learning.

