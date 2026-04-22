
# Digital Terrain Model Superresolution – Product Plan

## 1. Objective

Develop a deep learning system to upscale 30 m Digital Terrain Models (DTMs) to 10 m resolution using lidar-derived DTMs as ground truth and auxiliary covariates (e.g., Google Alpha Earth). The system should preserve fine-scale terrain structure (ridges, valleys, breaks in slope) while maintaining global elevation accuracy.

---

## 2. Key Insight

Standard pixel-wise superresolution leads to oversmoothing and loss of geomorphological structure. Instead of replacing raster prediction with contour prediction, we will:

* Retain raster elevation as the primary output
* Introduce **geometry-aware supervision** (including contour-derived signals)
* Use multi-task learning to encode terrain structure

---

## 3. System Architecture

### 3.1 Inputs

* 30 m DTM (upsampled baseline input)
* High-resolution covariates (e.g., Alpha Earth layers)
* Optional: land cover, hydrology masks, etc.

### 3.2 Outputs (Multi-task)

Primary:

* 10 m DTM (continuous elevation raster)

Auxiliary heads:

* Slope (gradient magnitude)
* Gradient vectors (dx, dy)
* Curvature (optional second derivative)
* Contour representation:

  * Option A: Binary contour maps at selected intervals
  * Option B: Signed distance to nearest contour

---

## 4. Training Strategy

### 4.1 Residual Learning

* Upsample 30 m DTM using bicubic or spline interpolation
* Model predicts residual correction to reach 10 m target

### 4.2 Patch-Based Training

* Maintain patch-based training for efficiency
* Ensure patches are large enough to capture terrain context

---

## 5. Loss Functions

### 5.1 Primary Loss

* L1 or L2 loss on elevation

### 5.2 Geometry-Aware Losses

* Gradient loss: difference in dx/dy
* Slope loss: magnitude difference
* Curvature loss (optional): second derivative consistency

### 5.3 Contour-Aware Loss

* Compute contour levels from ground truth
* Train auxiliary head to predict contour maps or distance transforms
* Loss options:

  * Binary cross-entropy (for contour maps)
  * L1/L2 (for distance fields)

### 5.4 Multi-Scale Loss

* Apply losses at multiple resolutions to enforce global + local consistency

---

## 6. Evaluation Metrics

### 6.1 Standard Metrics

* RMSE (elevation)
* MAE (elevation)

### 6.2 Structural Metrics

* Slope RMSE
* Curvature error
* Gradient direction consistency

### 6.3 Contour Metrics

* Contour displacement error
* Topological consistency (splits/merges)

### 6.4 Hydrological Metrics (Optional)

* Flow direction agreement
* Drainage network similarity

---

## 7. Experiments Roadmap

### Phase 1: Baseline

* Pixel-wise model (current approach)
* Establish baseline metrics

### Phase 2: Residual + Gradient Loss

* Add residual learning
* Add gradient and slope losses

### Phase 3: Multi-task Learning

* Add auxiliary heads (slope, gradients)

### Phase 4: Contour-Aware Supervision

* Add contour prediction head
* Compare contour map vs distance transform approaches

### Phase 5: Advanced Geometry

* Add curvature losses
* Explore implicit surface representations (optional)

---

## 8. Risks & Mitigations

### Risk: Contour instability

* Use distance transforms instead of raw contour lines

### Risk: Overfitting to high-frequency noise

* Apply smoothing priors or regularization

### Risk: Poor flat-area performance

* Ensure losses include low-gradient regions

---

## 9. Future Extensions

* Implicit neural representations for terrain surfaces
* Vector contour generation as a post-processing step
* Integration with hydrological models
