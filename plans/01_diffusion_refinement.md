# Plan: Conditional Residual Diffusion for DEM Refinement (Rewritten)

## 1) Goal (Refined)

Develop a **conditional residual diffusion model** that refines a low-resolution DEM (GEDTM) into a higher-quality 10 m DTM by modeling structured, geometry-preserving corrections.

This is **not a replacement pipeline**, but a **stage-2 refinement model** targeting failure modes that deterministic models could not resolve:

* Over-smoothed ridges/valleys
* Missed fine drainage structure
* Locally ambiguous corrections under noisy supervision

Formally, model the conditional distribution over **residuals**:

`p(R | Z_lr, X_dem, X_ae)`

where:

* `R = Z_gt - Z_lr`
* Output: `Z_hat = Z_lr + R`

---

## 2) Core Hypothesis

Diffusion is justified **only if** remaining errors are:

* structurally ambiguous (multi-plausible geometry), or
* caused by regression averaging (loss of sharp terrain features)

Diffusion is **not expected to help** with:

* systematic bias
* datum mismatch
* poor masking / supervision noise

---

## 3) Model Design

### 3.1 Residual Diffusion (Required)

The model operates strictly in **residual space**:

* Forward process: add noise to residual `R`
* Reverse process: denoise to recover `R`

This anchors predictions to `Z_lr` and reduces hallucination risk.

---

### 3.2 Conditioning Contract (Explicit)

Condition the model on:

* `Z_lr10` (low-resolution DEM)
* `U_enc` (encoded GEDTM uncertainty)
* `M_bld10` (building mask)
* `M_wp10` (persistent water mask)
* `M_ws10` (shoreline mask)
* `E_ae10` (AlphaEarth embeddings)

Input structure:

* Noisy residual state `R_t`
* Conditioning stack (concatenated or via attention)

**Guidance dropout:**

* Drop AE embeddings only (not DEM/masks)

---

### 3.3 Architecture

* U-Net diffusion backbone
* Timestep embedding (sinusoidal + MLP)
* Conditioning via:

  * channel concatenation (baseline), or
  * cross-attention at mid-scales (optional upgrade)

Operate at patch scale (128×128 at 10 m)

---

### 3.4 Prediction Parameterization

Use one of:

* ε-prediction (noise)
* v-prediction (preferred for stability)

Evaluate both in pilot.

---

## 4) Loss Design (Terrain-Aware)

Total loss combines diffusion objective with geometry constraints:

### 4.1 Diffusion Loss

* Standard ε or v loss
* Masked with weight map `W`

### 4.2 Elevation Reconstruction

* SmoothL1 or L1 on reconstructed DEM
* Weighted by `W`

### 4.3 Slope Loss (Required)

* Finite-difference slope comparison
* Strong signal for terrain structure

### 4.4 Optional Regularizers

* Small Laplacian penalty (avoid speckle)
* Water smoothness constraint inside `M_wp10`

---

## 5) Masks, Weights, and Trust (Non-Negotiable)

Reuse the exact weighting logic from v1:

* buildings → weight 0
* persistent water → weight 0
* shoreline → downweighted
* uncertainty → smooth downweighting

These apply to:

* diffusion loss
* reconstruction loss
* slope loss

---

## 6) Training Procedure

### 6.1 Standard Diffusion Training

* Sample timestep `t`
* Add noise to residual
* Predict ε or v

### 6.2 Guidance Dropout

* Drop AE embeddings with probability schedule
* Keep DEM + masks always present

### 6.3 Data

* Same patches and sampling strategy as baseline
* No new patch generation required

---

## 7) Inference Modes

### 7.1 Deterministic Mode (Primary)

* Use deterministic sampler (e.g., DDIM-style)
* Fixed step count
* Produces stable, reproducible DEM

### 7.2 High-Quality Mode

* More steps
* Potential stochasticity
* Used for evaluation only

### 7.3 Optional Multi-Sample Mode

* Generate multiple samples per patch
* Measure structural uncertainty

Not required for production output

---

## 8) Sampler Budget (Part of Model Definition)

Define explicitly:

* Low-cost sampler (fast evaluation)
* High-quality sampler (best achievable)

All comparisons must report:

* quality vs steps
* wall-clock cost

---

## 9) Evaluation Plan (Stratified)

Compare against best deterministic baseline on:

### Core Metrics

* elevation RMSE / MAE
* slope RMSE
* curvature / Laplacian RMSE

### Stratified Metrics (Critical)

* high-slope terrain
* near-water regions
* building mask regions

### Additional

* visual inspection of ridges/valleys
* absence of hallucinations
* seam stability under tiling

### Optional

* sample variance across multiple draws

---

## 10) Pilot Definition

### Minimal Pilot Setup

* residual diffusion only
* full conditioning stack
* masked losses
* deterministic + high-quality sampler

### Success Criteria (Strict)

Must satisfy ALL:

* no regression in elevation RMSE
* improved slope RMSE in high-relief strata
* improved curvature fidelity (no speckle)
* no increase in artifacts near masks
* acceptable compute cost

---

## 11) Risks

### Compute Cost

* much higher than feed-forward
* mitigated via step budget and pilot scale

### Overfitting Noise

* diffusion can model label noise
* mitigated by masking + robust loss

### Hallucination Risk

* mitigated by residual formulation + conditioning

---

## 12) Position in Pipeline

This model is a:

**geometry-focused refinement stage**

It is only justified if it improves:

* terrain structure
* slope fidelity

without degrading:

* global elevation accuracy
* hydrologic plausibility

---

## 13) Summary

This plan reframes diffusion as:

> A conditional residual refinement model for recovering sharp, plausible terrain structure under noisy, masked supervision.

Not a general generative model, and not a replacement for deterministic baselines.

