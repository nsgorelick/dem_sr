# Plan: Strong Non-Neural Baselines (Rewritten)

## 1) Goal (Refined)

Establish **strong, interpretable non-neural baselines** to:

* quantify how much signal is recoverable from engineered features alone,
* test robustness under distribution shift,
* diagnose whether neural models are learning beyond tabular structure.

This is primarily a **diagnostic and benchmarking plan**, not a production model initiative.

All predictions are residual-based:

* `z_hat = z_lr + r_pred`

---

## 2) Core Hypothesis

If engineered geomorphology and conditioning features capture most predictive signal, then:

* tree-based models may approach neural performance,
* remaining errors may reflect data/pipeline limitations rather than model capacity.

If not, then neural models are justified.

---

## 3) Staged Plan (Required)

This plan is executed in **strict stages** to control complexity.

### Stage 1 (Required): Patch-Level Residual Baseline

* Predict **aggregate residual correction per patch**
* Use tabular features derived from inputs
* Model: **LightGBM or XGBoost only** (no MLP initially)

### Stage 2 (Optional): Per-Pixel Tabular Model

* Predict residual at sampled pixel locations
* Requires explicit neighborhood feature design
* Only proceed if Stage 1 shows strong signal

### Stage 3 (Optional): Hybrid Post-Corrector

* Use tabular model to correct residuals of neural outputs
* Treat as error-modeling layer, not standalone baseline

---

## 4) Key Design Constraints (Non-Negotiable)

* Predict residuals relative to `z_lr`, not absolute DEM
* Apply residual clipping to plausible terrain ranges
* Use masking and finite-value filtering during feature extraction
* Maintain strict train/val split integrity (no leakage)
* Evaluate with identical metrics and splits as neural models

---

## 5) Stage 1: Patch-Level Baseline (Primary)

### 5.1 Prediction Target

Predict a compact residual statistic per patch, such as:

* median residual bias
* mean residual
* optionally slope-related error summary

Target must be **well-defined and stable**.

---

### 5.2 Feature Families (Constrained)

All features must be computable at inference.

#### DEM Morphology

* mean / std of `z_lr`
* slope statistics (mean, p90)
* curvature summaries
* relief metrics

#### Uncertainty + Masks

* mean uncertainty
* fraction masked (building, water, shoreline)
* weighted valid fraction

#### AE Embeddings (Summarized)

* channel-wise mean / variance
* optional PCA-reduced features

#### Optional Context Features

* local neighborhood summaries (must avoid leakage)

---

### 5.3 Model

* LightGBM or XGBoost regression
* moderate hyperparameter tuning
* no neural models in this stage

---

## 6) Stage 2: Per-Pixel Tabular Model (Optional)

Only proceed if Stage 1 shows strong predictive signal.

### 6.1 Target

* residual per pixel (sampled subset)

### 6.2 Features

* pixel-local features
* small neighborhood summaries (fixed window)

### 6.3 Risks

* feature explosion
* insufficient spatial context
* heavy inference cost

---

## 7) Stage 3: Hybrid Post-Corrector (Optional)

Use tabular model to predict:

* residual error of neural output

Inputs:

* neural prediction
* original features
* masks and uncertainty

Goal:

* capture systematic residual errors missed by neural model

---

## 8) Training and Data Integrity

### 8.1 Leakage Prevention (Critical)

* ensure features do not aggregate across validation geography
* avoid using any target-derived signals at inference
* verify strict split isolation

### 8.2 Feature Serialization

* store features in tabular format (Parquet / NumPy)
* maintain reproducible feature pipeline

---

## 9) Evaluation Plan

### 9.1 Baselines

Compare against:

* `z_lr`
* best neural model

### 9.2 Metrics

* elevation RMSE / MAE
* slope RMSE

### 9.3 Diagnostic Interpretation (Primary)

Evaluate:

* gap between tabular and neural models
* performance under AU vs non-AU splits

### 9.4 Optional Analysis

* feature importance (gain-based)
* SHAP (secondary, not primary focus)

---

## 10) Pilot Success Criteria

Stage 1 success:

* clearly beats `z_lr`

Extended success:

* narrows gap to neural baseline on AU

Additional outcomes:

* identifies dominant predictive features
* reveals whether neural models add significant value

---

## 11) Risks

### Feature Engineering Complexity

* feature space can grow rapidly
* mitigated by strict feature families

### Leakage

* engineered aggregates may leak information
* mitigated by strict validation discipline

### Limited Expressivity

* tabular models cannot capture full spatial structure
* acceptable for baseline purpose

---

## 12) Position in Pipeline

This is a:

**diagnostic baseline and sanity check**

It evaluates whether the problem requires deep spatial models or can be largely solved with structured features.

---

## 13) Summary

This plan reframes the question as:

> How much of DEM correction can be explained by engineered geomorphology features alone?

Success provides insight into whether model complexity is justified or if gains are dominated by feature-level signal.

