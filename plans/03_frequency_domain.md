# Plan: Frequency-Domain / Multi-Band Residual Learning (Rewritten)

## 1) Goal (Refined)

Develop a **multi-band residual model** that decomposes terrain correction into explicit spatial-frequency components and learns each separately.

Target decomposition:

* **Low-frequency:** large-scale bias and smooth terrain trends
* **Mid-frequency:** landform structure (hillslopes, ridges, valleys)
* **High-frequency:** sharp edges, channels, fine detail

Final output:

* `R = Σ R_band`
* `Z_hat = Z_lr + R`

This is a **signal-domain decomposition**, not a multi-stage pipeline.

---

## 2) Core Hypothesis

Single-stage residual models underperform because:

* low- and high-frequency signals compete during optimization,
* models over-smooth high-frequency structure,
* or amplify noise when pushed to recover detail.

Explicit frequency separation allows:

* stable low-frequency correction,
* targeted improvement of structure and edges,
* controlled handling of noisy high-frequency components.

---

## 3) Fixed Decomposition (Required)

Use a **deterministic Laplacian pyramid** for first pilot.

Define:

* residual: `R = Z_gt - Z_lr`
* decompose into:

  * `R_low`
  * `R_mid`
  * `R_high`

Reconstruction must satisfy:

* `R ≈ R_low + R_mid + R_high`

The same decomposition and reconstruction operators MUST be used in:

* training
* validation
* inference

---

## 4) Key Design Constraints (Non-Negotiable)

* All predictions are **residual bands**, not absolute elevations.
* Residual clamping applied after recombination (and optionally per-band).
* Use the same masking/weighting (`W`) as baseline.
* Maintain uncertainty and trust conditioning.
* Maintain evaluation parity with baseline metrics.

---

## 5) Architecture

### 5.1 Shared Backbone

* Single encoder for DEM + conditioning inputs

### 5.2 Multi-Band Heads

* Separate prediction head per band:

  * `Head_low`
  * `Head_mid`
  * `Head_high`

Each head predicts its corresponding residual component.

---

## 6) Training Targets

Compute targets as:

* `R = Z_gt - Z_lr`
* `(R_low, R_mid, R_high) = LaplacianDecompose(R)`

Train each head against its band target.

---

## 7) Loss Design (Band-Specific)

### 7.1 Low-Frequency Loss

Focus on stability and bias correction:

* elevation loss (weighted)
* slope loss at coarse/downsampled resolution

---

### 7.2 Mid-Frequency Loss

Focus on terrain structure:

* elevation loss
* slope loss (moderate strength)

---

### 7.3 High-Frequency Loss (Critical)

Focus on edges and fine detail:

* strong slope loss
* curvature / Laplacian loss

With additional safeguards:

* stronger masking sensitivity
* optional lower residual cap
* regularization to prevent speckle/noise amplification

---

## 8) Reconstruction

At training and inference:

* `R_hat = R_low_hat + R_mid_hat + R_high_hat`
* `Z_hat = Z_lr + R_hat`

Apply residual clamping after recombination.

---

## 9) Training Procedure

### 9.1 Initialization

* initialize backbone from best global model (if available)

### 9.2 Joint Training

* train all bands simultaneously
* monitor per-band loss behavior

### 9.3 Optional Curriculum

If unstable:

* train low band first
* then enable mid band
* then high band

---

## 10) Stability Checks (Required)

### 10.1 Reconstruction Consistency

Verify:

* decomposition + recomposition matches original residual
* no systematic drift across bands

### 10.2 Band Energy Monitoring

Track:

* magnitude of each predicted band
* ensure high-frequency band does not dominate or explode

---

## 11) Evaluation Plan

### 11.1 Baselines

Compare against:

* best single-stage residual model
* same architecture without band separation

### 11.2 Metrics

* elevation RMSE / MAE
* slope RMSE
* curvature / Laplacian RMSE

### 11.3 Band-Level Metrics

* RMSE per band
* spectral energy error

### 11.4 Structure-Focused Evaluation

* high-slope terrain
* contour-adjacent regions
* ridge/valley sharpness

---

## 12) Pilot Success Criteria

Must satisfy ALL:

* improved slope/structure metrics
* improved contour fidelity
* no regression in elevation RMSE
* no visible speckle or noise artifacts
* stable band decomposition (no drift)

---

## 13) Risks

### Decomposition Mismatch

* imperfect reconstruction leads to artifacts
* mitigated by strict operator consistency

### High-Frequency Noise Amplification

* model overfits noisy labels
* mitigated by masking, regularization, and caps

### Misaligned Frequency vs Terrain Structure

* some terrain features span multiple bands
* may limit gains vs simpler approaches

---

## 14) Position in Pipeline

This is a:

**signal-domain decomposition of residual learning**

It targets frequency-based optimization conflicts, not spatial context or regime heterogeneity.

---

## 15) Summary

This plan reframes the problem as:

> Learn terrain corrections in separate frequency bands, then recombine them in a controlled, stable way.

Success depends on aligning frequency decomposition with real error structure and preventing high-frequency overfitting.

