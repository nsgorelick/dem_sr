# Plan: Large-Context Tile Training (Rewritten)

## 1) Goal (Refined)

Train models with **substantially larger spatial context** while computing loss on a **trusted center crop**, to address:

* context starvation from small patches,
* border artifacts,
* misinterpretation of terrain structure that depends on broader context.

This is a **training and inference geometry change**, not a new model family.

---

## 2) Core Hypothesis

Some terrain corrections require **context beyond 128×128 patches**, including:

* ridge/valley continuity,
* drainage structure,
* hillslope-scale geometry.

Providing larger context will:

* improve interior predictions,
* reduce seam artifacts,
* increase slope/structure consistency.

---

## 3) Fixed Pilot Geometry (Required)

Define explicitly:

* Input tile: **512×512**
* Supervised center: **256×256**
* Context halo: **128 px per side**

Loss is computed primarily on the **center region only**.

---

## 4) Key Design Constraints (Non-Negotiable)

* Residual prediction form unchanged: `z_hat = z_lr + r_tile`
* Residual clamping applied before stitching
* Same masking/weighting (`W`) used as baseline
* Same uncertainty + trust conditioning
* Same evaluation protocol and splits

---

## 5) Training Design

### 5.1 Tile Sampling

Replace patch sampling with tile sampling:

* sample 512×512 windows
* enforce **minimum valid coverage** in center region

Recommended constraints:

* `mean_W(center) >= threshold`
* `valid_frac(center) >= threshold`

Reject tiles with insufficient supervision.

---

### 5.2 Center-Crop Loss (Core Mechanism)

Compute loss as:

* full weight on center crop
* zero or reduced weight on outer halo

This ensures:

* model uses context for prediction,
* but is only supervised where labels are reliable.

---

### 5.3 Optional Edge Regularization

If seam artifacts appear:

* apply weak consistency loss near center boundary
* avoid over-constraining outer halo

---

## 6) Model

No architecture change required for pilot.

Use existing best-performing backbone.

---

## 7) Inference Pipeline

### 7.1 Sliding Window

* run model on overlapping tiles
* stride = center size (e.g., 256)

### 7.2 Blending

* blend overlaps using cosine or linear weights
* ensure residual clamping before blending

### 7.3 Output Assembly

* reconstruct full DEM from stitched tiles

---

## 8) Compute Strategy

Large tiles increase cost. Mitigate via:

* reduced batch size
* gradient accumulation
* mixed precision
* activation checkpointing

Maintain comparable training throughput where possible.

---

## 9) Evaluation Plan

### 9.1 Baselines

Compare against:

* original 128×128 model
* same architecture without center-crop loss (if feasible)

### 9.2 Metrics

* elevation RMSE / MAE
* slope RMSE
* curvature / Laplacian RMSE

### 9.3 Stratification

* high-slope terrain
* terrain continuity classes
* near-water regions

### 9.4 Seam Diagnostics (Critical)

* visual seam inspection
* quantitative seam error metrics

---

## 10) Pilot Success Criteria

Must satisfy ALL:

* improved slope/structure metrics
* reduced seam artifacts
* no regression in elevation RMSE
* acceptable runtime / throughput cost

---

## 11) Risks

### Supervision Dilution

* larger tiles may include low-quality regions
* mitigated via center coverage thresholds

### Compute Cost

* training becomes slower
* mitigated via optimization strategies

### Limited Gains

* if errors are purely local, improvement may be minimal

---

## 12) Scope Control (Important)

First pilot must be **context-only**:

* no multi-scale pyramids
* no architecture changes
* no new fusion mechanisms

Add complexity only after validating benefit.

---

## 13) Position in Pipeline

This is a:

**training geometry upgrade to improve spatial context usage**

It complements existing models without changing core architecture.

---

## 14) Summary

This plan reframes training as:

> Learn from large spatial context, but trust only the center.

Success depends on improving structure and seams without incurring excessive compute cost.

