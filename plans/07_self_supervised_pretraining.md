# Plan: Self-Supervised Pretraining + Supervised Finetune (Rewritten)

## 1) Goal (Refined)

Learn **strong, transferable representations** from large unlabeled DEM + AE data, then finetune for supervised DEM correction.

This plan targets **generalization and representation quality**, not changes to the supervised objective.

Downstream prediction remains:

* `z_hat = z_lr + r`

---

## 2) Core Hypothesis

If current models are limited by representation learning, then self-supervised pretraining will:

* improve AU (out-of-domain) performance,
* improve performance in hard strata (high slope, uncertainty),
* accelerate supervised convergence.

If not, gains will be minimal and bottlenecks lie elsewhere.

---

## 3) Key Design Constraints (Non-Negotiable)

* SSL affects **encoder only**; supervised head remains unchanged
* Residual formulation and clamping unchanged during finetune
* Same masking (`W`) and trust/uncertainty handling in finetune
* Strict train/val separation (no leakage from pretraining set into evaluation targets)
* Evaluation identical to baseline protocols

---

## 4) First Pilot Definition (Strict)

### Objective: Masked Reconstruction (MAE-style)

Use masked reconstruction over DEM + AE inputs.

* randomly mask input patches
* predict missing values from context

Rationale:

* preserves spatial structure
* aligns with downstream pixel-level prediction
* avoids learning harmful invariances

---

### 4.1 Encoder Design

* shared encoder for DEM + AE inputs
* same architecture family as downstream model (or compatible subset)

Joint pretraining is required in first pilot.

---

### 4.2 Masking Strategy

* random spatial masking (e.g., 50–75%)
* contiguous blocks preferred over independent pixels

---

## 5) Augmentation Policy (Critical)

Only use **terrain-safe augmentations**.

### Allowed

* 90° rotations
* horizontal / vertical flips
* small additive noise

### Forbidden

* scaling (breaks slope meaning)
* elastic deformation
* large misalignment between DEM and AE

---

## 6) Pretraining Data

### 6.1 Dataset Requirements

* large unlabeled corpus of DEM + AE
* include diverse geography (AU + non-AU)

### 6.2 Data Integrity

* deduplicate against supervised validation/test regions
* ensure no leakage of evaluation targets

---

## 7) Training Procedure

### 7.1 Pretraining

* train encoder on masked reconstruction objective
* track reconstruction loss and convergence

---

### 7.2 Representation Diagnostic (Required)

Before finetuning:

* freeze encoder
* train small linear or shallow head on supervised task

Purpose:

* test whether learned features are useful without adaptation

---

### 7.3 Finetuning Protocol (Ordered)

Run in sequence:

1. **Frozen encoder**
2. **Partial unfreeze (top layers)**
3. **Full finetune**

Compare all three.

---

### 7.4 Control Experiment (Critical)

Train baseline model with:

* same architecture
* longer training (no SSL)

Purpose:

* ensure gains are not due to extra compute alone

---

## 8) Evaluation Plan

### 8.1 Baselines

Compare:

* supervised from scratch
* SSL + finetune
* longer supervised training (control)

---

### 8.2 Metrics

* elevation RMSE / MAE
* slope RMSE
* curvature / Laplacian RMSE

---

### 8.3 Generalization Focus (Primary)

Evaluate specifically on:

* AU region
* high-slope terrain
* high-uncertainty regions

Global average alone is not sufficient.

---

## 9) Pilot Success Criteria

Must satisfy ALL:

* consistent AU improvement across at least two architectures
* improvement in hard strata (slope / uncertainty)
* improvement beyond longer supervised training baseline

---

## 10) Risks

### Objective Mismatch

* SSL task learns irrelevant invariances
* mitigated by using masked reconstruction

### Ineffective Representations

* encoder does not transfer well
* detected via representation diagnostic step

### Compute Overhead

* added pretraining cost
* must justify via measurable gains

---

## 11) Position in Pipeline

This is a:

**representation-learning enhancement**

It complements all other model improvements and is orthogonal to architecture changes.

---

## 12) Summary

This plan reframes the problem as:

> Learn general terrain representations from unlabeled data, then adapt them to supervised DEM correction.

Success depends on alignment between the pretraining objective and downstream geometry tasks, and on demonstrating real gains in out-of-domain generalization.

