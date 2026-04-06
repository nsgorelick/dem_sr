# DEM Super-Resolution Design Doc (v1)

**Owner:** gorelick  
**Last Modified**: Feb 26, 2026  
**Status:** Draft

---

## **1\) Objectives**

### **Summary**

This project aims to generate a wall-to-wall 10m bare-earth Digital Terrain Model (DTM) via super-resolution, using a supervised deep learning model. 

The model uses the global GEDTM30 (resampled to 10m) as the low-resolution input, with 10m AlphaEarth embeddings providing land-surface context and sparse LiDAR-derived DTMs (\~1–2m, downsampled to 10m) used as high-resolution supervision during training.

The first experiment will use a residual U-Net with a dual-encoder design to integrate the inputs using bounded global FiLM modulation to limit the risk of introducing building- or vegetation-like artifacts. Training focuses on residual correction (output \= GEDTM \+ learned adjustment), elevation and slope error losses, and masking or downweighting of buildings, persistent water, shorelines, and uncertain regions to account for data noise and differences in acquisition year.

### **Key constraints**

* No HR DSM/DTM pair; HR target is **LiDAR DTM only**.  
* LR input is always **GEDTM30** globally.  
* High-resolution guidance is **AlphaEarth 64-channel 10 m embeddings**.  
* Training HR coverage is sparse (LiDAR campaign footprints), and acquisition years span a wide range.

### **Success metrics (must-have)**

Evaluate on holdout regions (and year bins) using:

* **Elevation MAE / RMSE** (meters)  
* **Slope RMSE** (degrees or %), computed consistently  
* **Hydrology stability** proxies (optional but useful): flow direction disagreement, stream network overlap (later)  
* Stratified metrics:  
  * near-water buffer band  
  * urban/building mask region  
  * high-slope terrain

---

## **2\) Data**

### **Inputs (per 10 m grid)**

1. **GEDTM30** resampled to 10 m: `Z_lr10` (1 channel)  
2. **AlphaEarth embeddings**: `E_ae10` (64 channels)  
3. Additional training layers:  
   * **GEDTM uncertainty**: `U_lr10` (1 channel, int16 scaled by 100\)  
   * **Building mask**: `M_bld10` (1 channel at 10 m; derived from 30 m mask)  
   * **Water masks** from Global Surface Water and GLAD:  
     * `M_wp10`: **persistent water**.  
     * `M_ws10`: the **dynamic shoreline / water-variability zone** (3m double-sided buffer on M\_wp10 boundary.

### **Targets**

* `Z_gt10:` **High resolution LiDAR DTM** downsample to **10 m** 

---

## **3\) Preprocessing**

### **3.1 Consistent grids**

Everything is reprojected to the local UTM grid.

### **3.2 Handle vertical offsets (recommended)**

Because GEDTM30 and LiDAR may differ by vertical datum/bias, train the model to predict **residual corrections** and optionally remove a per-tile bias.

* Model predicts residual `R10`  
* Output \= `Z_lr10 + R10`  
* Loss computed against `Z_gt10`

**TODO: Detrend to remove per-tile mean bias (more robust)**

* Compute `b = median(Z_gt10 - Z_lr10)` over trusted pixels  
* Use `Z_lr10' = Z_lr10 + b` as “bias-corrected LR”  
* Train residual on top of `Z_lr10'`  
* At inference, you won’t have `b`; so only do this if you also learn a bias head or you’re confident bias is small. For v1, stick to Option A.

### **3.3 Masks and weights**

Create per-pixel loss weight `W` at 10 m:

`W = (1 - dilate(M_bld10))`  
  `* (1 - M_wp10)`  
  `* (1 - 0.8 * M_ws10)`  
  `* (1 - 0.5 * U_enc^2)`

Notes:

* buildings and persistent water → weight 0  
* shoreline → weight ≈ 0.2  
* uncertainty → smooth nonlinear downweighting  
* all factors combine multiplicatively

GEDTM uncertainty `U_lr10` is always used:

* Convert to float: `U = U_lr10 / 100`  
* Encode: `U_enc = clamp(log1p(U), 0, 1)`  
* Downweight: `W *= (1 - 0.5 * U_enc^2)`

Uncertainty (U\_enc)  is used in three roles:

1. **Loss weighting**: `W *= (1 - 0.5 * U_enc^2)`  
2. **Model input**: using `U_enc` (not raw U)  
3. **Sampling**: `mean_uncert = mean(U_enc)`

This ensures consistent interpretation across training.

### **3.4 Patch sampling**

**Patch size:** fixed **128×128** at 10 m.

**Per-patch stats (compute once):**

* `p90_slope`, `relief = p95(z)-p5(z)`  
* `resid_scale = p95(|Z_gt10 − Z_lr10|)`  
* `mean_uncert = mean(U_enc)`  
* `frac_building`, `frac_water`, `frac_shore`  
* `mean_W = mean(W)`, `valid_frac = mean(W > 0.05)`

**Hard filtering (reject):**

* `mean_W < 0.3`  
* `valid_frac < 0.7`  
* `frac_water > 0.5`  
* `relief < 0.5` or obvious corruption

**Primary stratification (by terrain):**  
Slope bins from `p90_slope`:

* A: 0–2°, B: 2–5°, C: 5–10°, D: 10–20°, E: \>20°

**Target mix (per epoch):** A 20%, B 25%, C 25%, D 20%, E 10%

**Difficulty score (within bin):**

* Normalize (`p5/p95` clip → \[0,1\]): slope `s`, residual `r`, uncertainty `u`, relief `h`  
* `edge_bonus e = 1` if `frac_shore > τ_shore` or `frac_building > τ_bld` else `0`  
* `score = 0.40*s + 0.30*r + 0.15*u + 0.10*h + 0.05*e`  
* Sampling weight: `w = 0.05 + sqrt(score)`

**Balancing:**

* Choose **slope bin → (zone, year\_bin) group (inverse-frequency) → patch (weighted by w)**

**Notes:**

* `mean_W`/`valid_frac` gate quality; they are not part of the score

### **3.4.1 Pseudocode (minimal)**

`function build_index(patches):`  
  `keep = []`  
  `for p in patches:`  
    `if p.mean_W < 0.3: continue`  
    `if p.valid_frac < 0.7: continue`  
    `if p.frac_water > 0.5: continue`  
    `if p.relief < 0.5 or p.is_corrupt: continue`  
    `p.bin = assign_slope_bin(p.p90_slope)`  
    `p.group = (p.zone, p.year_bin)`  
    `keep.append(p)`

  `# robust normalize over keep`  
  `norm = percentile_normalizers(keep, [p90_slope, resid_scale, mean_uncert, relief])`  
  `for p in keep:`  
    `s = norm.slope(p.p90_slope)`  
    `r = norm.resid(p.resid_scale)`  
    `u = norm.unc(p.mean_uncert)`  
    `h = norm.relief(p.relief)`  
    `e = 1 if (p.frac_shore > tau_shore or p.frac_building > tau_bld) else 0`  
    `p.score = 0.40*s + 0.30*r + 0.15*u + 0.10*h + 0.05*e`  
    `p.w = 0.05 + sqrt(p.score)`  
  `return keep`

`function sample_patch(index):`  
  `b = sample_slope_bin_with_quota()`  
  `Ib = filter(index, bin==b)`  
  `g = sample_group_inverse_frequency(Ib)        # (zone, year_bin)`  
  `Ibg = filter(Ib, group==g)`  
  `return weighted_random_choice(Ibg, weights=[p.w])`

---

---

## **4\) Model Architecture (v1): Dual Encoder \+ Global FiLM \+ Residual U-Net**

### **4.1 Why FiLM here**

FiLM lets AE embeddings **modulate** DEM features without letting the network “copy texture” from AE. Using **global FiLM** (per-channel parameters, not spatially varying) further reduces risk of hallucinating DSM-like detail.

### **4.2 High-level diagram**

* DEM branch (strong): encodes terrain base \+ masks/uncertainty  
* AE branch (light): encodes semantic context  
* FiLM modulation at mid-scales (1/2, 1/4, 1/8 resolution)  
* U-Net decoder reconstructs residual  
* Output \= upsampled GEDTM \+ residual

### **4.3 Inputs**

* DEM encoder input channels: 5  
  * `[Z_lr10, U_enc, M_bld10, M_wp10, M_ws10]`  
* AE encoder input channels: 64 (AE embeddings)

### **4.4 Channels (practical defaults)**

**DEM encoder channels per stage**

* S0 (1×): 32  
* S1 (1/2): 64  
* S2 (1/4): 128  
* S3 (1/8): 256  
* Bottleneck (1/16): 384 (optional; you can stop at 1/8 given 100×100)

**AE encoder channels**

* S0: 16  
* S1: 32  
* S2: 64  
* S3: 128

**Decoder channels**

* mirror DEM: 256 → 128 → 64 → 32

### **4.4.1 Core building blocks (explicit definitions)**

All convolutional blocks use **BatchNorm** and SiLU activation unless otherwise noted.

#### **ResBlock(C\_in, C\_out)**

Structure:

1. `Conv3×3(C_in → C_out, stride=1, padding=1)`  
2. `BatchNorm(C_out)`  
3. `SiLU`  
4. `Conv3×3(C_out → C_out, stride=1, padding=1)`  
5. `BatchNorm(C_out)`  
6. Residual connection:  
   * If `C_in != C_out`, use `Conv1×1(C_in → C_out)` on the skip path.  
7. `SiLU`

Pseudocode:

function ResBlock(x, C\_out):  
    y \= conv3x3(x, C\_out)  
    y \= batchnorm(y)  
    y \= silu(y)  
    y \= conv3x3(y, C\_out)  
    y \= batchnorm(y)

    if channels(x) \!= C\_out:  
        x \= conv1x1(x, C\_out)

    y \= y \+ x  
    y \= silu(y)  
    return y

#### **DemBlock\_s**

Each DEM stage uses **two ResBlocks**:

function DemBlock\_s(x, C\_out):  
    x \= ResBlock(x, C\_out)  
    x \= ResBlock(x, C\_out)  
    return x

#### **AeBlock\_s**

AE encoder uses lighter blocks (one ResBlock per stage):

function AeBlock\_s(x, C\_out):  
    x \= ResBlock(x, C\_out)  
    return x

#### **Down (spatial reduction)**

Use learnable strided convolution for downsampling:

function Down(x, C\_out):  
    x \= conv3x3(x, C\_out, stride=2, padding=1)  
    x \= batchnorm(x)  
    x \= silu(x)  
    return x

#### **Up (decoder upsampling)**

Use bilinear upsampling \+ Conv (stable for SR):

function Up(x, C\_out):  
    x \= bilinear\_upsample(x, scale=2)  
    x \= conv3x3(x, C\_out, stride=1, padding=1)  
    x \= batchnorm(x)  
    x \= silu(x)  
    return x

These definitions apply to both DEM and AE encoders unless otherwise specified.

### **4.5 FiLM modulation points**

Apply FiLM to DEM features at **S1, S2, S3** (not S0).

Let `F_dem[s]` be DEM features (B, C\_dem, H, W) at scale s.  
Let `F_ae[s]` be AE features (B, C\_ae, H, W).

Compute pooled vector:

* `v = GAP(F_ae[s])` → (B, C\_ae)

FiLM generator:

* `h = MLP(v)` → (B, 2\*C\_dem) outputs concatenated `[gamma, beta]`  
* reshape to `(B, C_dem, 1, 1)`

Apply bounded modulation (important):

* `gamma = tanh(gamma)` (bounds scale)  
* Use modulation strength `α_s` per scale:  
  * α1 \= 0.10, α2 \= 0.15, α3 \= 0.20 (start here)  
* Update:  
  * `F_dem[s] = (1 + α_s * gamma) * F_dem[s] + α_s * beta`

Optional trust gating (recommended if you have masks):

* Compute scalar trust `t` from masks/uncertainty pooled or per-pixel:  
  * simplest global: `t = 1 - mean(M_bld10 or water in patch)` clipped  
  * better: per-pixel trust map downsampled to scale s  
* Use:  
  * `F_dem[s] = (1 + α_s * t * gamma) * F_dem[s] + α_s * t * beta`

### **4.6 Residual head**

Decoder outputs residual `R10` (1 channel).  
Optionally clamp residual:

* `R10 = r_cap * tanh(R10 / r_cap)` with `r_cap=20m` initially

Final output:

* `Z_hat10 = Z_lr10 + R10`

### **4.7 Plan B if FiLM is insufficient**

If global FiLM (per-patch, per-channel modulation) is not expressive enough, switch to one of the following **drop-in fusion upgrades** while keeping the rest of the pipeline (residual learning, masks/weights, guidance dropout) unchanged.

#### **B1) Spatial gated late-fusion (no attention)**

**When to use:** FiLM helps but misses local transitions (shorelines, riparian corridors, forest/grass boundaries), or AE is under-utilized.

At scale `s` (recommended: S1–S3), let:

* `F_dem[s]` be DEM features (B, C\_d, H, W)  
* `F_ae[s]` be AE features (B, C\_a, H, W)  
* `T_s` be a trust stack at that scale: `[M_bld_dilated_s, M_wp_s, M_ws_s, U_s]`

Fusion (additive, gated):

* `A = Conv1x1(F_ae[s])` → (B, C\_d, H, W) (compress AE to match DEM channels)  
* `g = sigmoid(Conv1x1(concat(F_dem[s], A, T_s)))` → (B, C\_d, H, W)  
* `F_fused[s] = F_dem[s] + α_s * g ⊙ A`

Safety knobs:

* keep `α_s` small initially (e.g., 0.10/0.15/0.20)  
* include masks/uncertainty in `T_s` so `g` naturally suppresses AE influence in risky pixels

Pseudocode:

function fuse\_gated(F\_dem, F\_ae, T\_s, alpha):  
    A \= conv1x1\_ae\_to\_dem(F\_ae)                  \# (B,Cd,H,W)  
    g \= sigmoid(conv1x1\_gate(concat(F\_dem, A, T\_s)))  
    return F\_dem \+ alpha \* g \* A

#### **B2) Windowed cross-attention fusion (DEM queries, AE keys/values)**

**When to use:** you need stronger, context-dependent fusion (AE must help decide where to sharpen) and B1 still underperforms.

Compute cross-attention within local windows (8×8 or 16×16 depending on scale) at **coarser scales only** (typically S2 and/or S3).

* `Q = Wq(F_dem[s])`  
* `K = Wk(F_ae[s])`  
* `V = Wv(F_ae[s])`  
* `A = WindowedAttn(Q, K, V)`  
* `g = sigmoid(Conv1x1(concat(F_dem[s], A, T_s)))`  
* `F_fused[s] = F_dem[s] + α_s * g ⊙ Wo(A)`

Pseudocode:

function fuse\_xattn(F\_dem, F\_ae, T\_s, alpha, window):  
    Q \= proj\_q(F\_dem)  
    K \= proj\_k(F\_ae)  
    V \= proj\_v(F\_ae)  
    A \= windowed\_attention(Q, K, V, window=window)  
    g \= sigmoid(conv1x1\_gate(concat(F\_dem, A, T\_s)))  
    return F\_dem \+ alpha \* g \* proj\_out(A)

#### **B3) Mixture-of-Experts (MoE) residual heads**

**When to use:** systematic regime failures (e.g., coasts vs mountains vs flatlands) suggest multiple distinct “styles” are needed.

* Keep a shared DEM encoder (and optionally AE encoder).  
* Predict `K` residuals `R_k` with lightweight expert heads.  
* A gating network uses pooled AE \+ summary stats of masks/uncertainty to produce mixture weights `π`.  
* Output residual is `R = Σ_k π_k * R_k`.

Pseudocode:

function moe\_residual(F\_shared, E\_ae, stats):  
    pi \= softmax(mlp(concat(GAP(E\_ae), stats)))   \# (B,K)  
    R\_list \= \[head\_k(F\_shared) for k in 1..K\]     \# each (B,1,H,W)  
    R \= sum\_k pi\[:,k\] \* R\_list\[k\]  
    return R

---

**Recommended escalation path:** FiLM → B1 (gated) → B2 (cross-attn) → B3 (MoE).

---

## **5\) Losses**

### **5.1 Core elevation loss**

Use robust loss on elevation:

* `L_elev = mean(W * smoothL1(Z_hat10 - Z_gt10, delta=1.0))`  
  * L1 also fine; SmoothL1 helps with outliers

### **5.2 Slope loss (strongly recommended)**

Compute slope with finite differences on 10 m grid.

* `S(z) = sqrt((dz/dx)^2 + (dz/dy)^2)` using centered differences  
* `L_slope = mean(W_slope * |S(Z_hat10) - S(Z_gt10)|)`  
  Use same weights `W` or a slightly eroded `W` to avoid boundary artifacts.

Total:

* `L = L_elev + λ_slope * L_slope`  
  Start λ\_slope \= 0.5 and tune (0.2–1.0).

### **5.3 Optional regularizers (only if needed)**

* small Laplacian penalty outside ridges if you see speckle:  
  * `L_lap = mean(W * |∇² Z_hat10|)` with very small weight (1e-4 to 1e-3)  
* water smoothness inside persistent water:  
  * `mean(M_wp10 * |∇² Z_hat10|)` (small)

---

## **6\) Training Procedure**

### **6.1 Guidance dropout (important)**

To prevent AE dominance:

* With probability `p_guidance` per sample (start 0.3):  
  * replace AE embeddings with zeros (or shuffle embeddings across batch)  
    Schedule:  
* Epochs 0–N: p=0.4  
* Later: p=0.2

### **6.2 Data augmentation**

* Random rotations (0/90/180/270) and flips  
* Small random shifts (±1 px) between AE and DEM optionally to simulate co-reg noise (careful: do this rarely, e.g., 5–10% of samples)  
* Add small noise to GEDTM input (optional) to improve robustness

### **6.3 Optimization**

* AdamW  
* LR \~ 1e-4 (start)  
* cosine decay or step schedule  
* Train to convergence on held-out geo regions

### **6.4 Validation splits (non-negotiable)**

* Split by **geography** (entire regions withheld)  
* Also stratify by **year bins**  
  Report metrics per stratum.

---

## **7\) Inference**

### **7.1 Tiling**

Australia wall-to-wall:

* Use overlap tiling (e.g., 256×256 or 512×512 at 10 m) depending on memory  
* Blend overlaps with a cosine window to reduce seams

### **7.2 Inputs at inference**

You will have everywhere:

* GEDTM30  
* AE embeddings  
* water masks  
* building mask (if global)  
* uncertainty (if global)

Run model → residual → add to GEDTM30@10m.

### **7.3 Output products**

* `DTM10`: predicted 10 m DTM  
* Optional: `DTM10_residual`: residual map for diagnostics  
* Optional: `DTM10_confidence`: could be derived from GEDTM uncertainty \+ model epistemic (MC dropout) later

---

## **8\) Pseudocode**

### **8.1 Data loader (per patch)**

Assumption: **all rasters are already saved on disk in a common CRS, pixel grid, and resolution** (10 m for inputs/targets/masks). Training uses fixed `128×128` patches. The loader should only do window reads/crops, not reprojection.

\# On-disk assumption (per tile\_id), all in the SAME CRS/grid at 10 m:  
\#   \- Z\_lr10\[tile\_id\]      : GEDTM30 resampled to 10 m (float32)  
\#   \- E\_ae10\[tile\_id\]      : AlphaEarth embeddings at 10 m (float16/float32), 64 channels  
\#   \- Z\_gt10\[tile\_id\]      : LiDAR DTM downsampled to 10 m (float32)  
\#   \- U\_lr10\[tile\_id\]      : GEDTM uncertainty at 10 m (required)  
\#   \- M\_bld10\[tile\_id\]     : building mask at 10 m (required)  
\#   \- M\_wp10\[tile\_id\]      : persistent water mask at 10 m (required)  
\#   \- M\_ws10\[tile\_id\]      : dynamic shoreline / water-variability mask at 10 m (required)  
\# Optional (not required in v1): store derived weights W10\[tile\_id\] precomputed on disk.

\# Practical storage notes:  
\# \- Store rasters as COG GeoTIFF for single-band layers.  
\# \- Store AE embeddings as Zarr/NPY/COG with band-interleaving for fast window reads.  
\# \- Keep everything tile-indexed so random patch reads are O(1) metadata.

function load\_patch(tile\_id, crop\_xyhw):  
    \# crop\_xyhw \= (x0, y0, 128, 128\) in pixel coordinates of the 10 m tile grid  
    \# Fixed plan: w \= h \= 128\. Use reflect padding if crop extends beyond tile bounds.

    \# 1\) Read windowed crops (no reprojection)  
    Z\_lr \= read\_window("Z\_lr10", tile\_id, crop\_xyhw, pad="reflect")          \# (h,w)  
    Z\_gt \= read\_window("Z\_gt10", tile\_id, crop\_xyhw, pad="reflect")          \# (h,w)  
    E\_ae \= read\_window("E\_ae10", tile\_id, crop\_xyhw, pad="reflect")          \# (64,h,w)

    \# 2\) Read required auxiliary layers (already prepared at 10 m)  
    U\_lr  \= read\_window("U\_lr10",  tile\_id, crop\_xyhw, pad="reflect")         \# (h,w)  
    M\_bld \= read\_window("M\_bld10", tile\_id, crop\_xyhw, pad="reflect")         \# (h,w)  
    M\_wp  \= read\_window("M\_wp10",  tile\_id, crop\_xyhw, pad="reflect")         \# (h,w)  
    M\_ws  \= read\_window("M\_ws10",  tile\_id, crop\_xyhw, pad="reflect")         \# (h,w)

    \# Note: W\_y10 is used upstream to construct M\_wp10/M\_ws10 via union/intersection logic (preferred).

    \# 3\) Build per-pixel loss weight map W (compute on the fly for v1)  
    W \= ones(h,w)

    \# Dilate building mask to buffer label/co-registration uncertainty  
    M\_bld\_d \= dilate(M\_bld, radius\_px=2)      \# radius in 10 m pixels  
    W \*= (1 \- M\_bld\_d)

    \# Persistent open water: mask out entirely  
    W \*= (1 \- M\_wp)

    \# Shoreline band: downweight (don’t fully zero)  
    W \*= (1 \- 0.8 \* M\_ws)                     \# \=\> 0.2 weight in shore band

    \# Encode and downweight GEDTM uncertainty  
    U \= U\_lr / 100.0  
    U\_enc \= clamp(log1p(U), 0, 1\)  
    W \*= (1 \- 0.5 \* U\_enc \* U\_enc)

    \# Optional: mask invalid / nodata if present  
    \# valid \= isfinite(Z\_lr) & isfinite(Z\_gt)  
    \# W \*= valid

    \# 4\) Guidance dropout to prevent AE dominance  
    if random() \< p\_guidance:  
        E\_ae \= zeros\_like(E\_ae)

    \# 5\) Assemble model inputs  
    X\_dem \= stack(\[Z\_lr, U\_enc, M\_bld, M\_wp, M\_ws\])   \# (C\_dem,h,w)  
    X\_ae  \= E\_ae                                      \# (64,h,w)

    return X\_dem, X\_ae, Z\_lr, Z\_gt, W

### **8.2 Model forward (FiLM U-Net residual)**

function forward(X\_dem, X\_ae, Z\_lr10):  
    \# Encode DEM  
    Fd0 \= DemBlock0(X\_dem)               \# (32,H,W)  
    Fd1 \= Down(DemBlock1(Fd0))           \# (64,H/2,W/2)  
    Fd2 \= Down(DemBlock2(Fd1))           \# (128,H/4,W/4)  
    Fd3 \= Down(DemBlock3(Fd2))           \# (256,H/8,W/8)

    \# Encode AE (lighter)  
    Fa0 \= AeBlock0(X\_ae)                 \# (16,H,W)  
    Fa1 \= Down(AeBlock1(Fa0))            \# (32,H/2,W/2)  
    Fa2 \= Down(AeBlock2(Fa1))            \# (64,H/4,W/4)  
    Fa3 \= Down(AeBlock3(Fa2))            \# (128,H/8,W/8)

    \# FiLM at mid scales (global FiLM)  
    Fd1 \= apply\_film(Fd1, Fa1, alpha=0.10)  
    Fd2 \= apply\_film(Fd2, Fa2, alpha=0.15)  
    Fd3 \= apply\_film(Fd3, Fa3, alpha=0.20)

    \# Decode (U-Net)  
    U2 \= Up(Dec3(Fd3))                   \# \-\> (128,H/4,W/4)  
    U2 \= concat(U2, Fd2)  
    U2 \= Dec2(U2)

    U1 \= Up(U2)                          \# \-\> (64,H/2,W/2)  
    U1 \= concat(U1, Fd1)  
    U1 \= Dec1(U1)

    U0 \= Up(U1)                          \# \-\> (32,H,W)  
    U0 \= concat(U0, Fd0)  
    U0 \= Dec0(U0)

    R10 \= Conv1x1(U0)                    \# (1,H,W)  
    R10 \= r\_cap \* tanh(R10 / r\_cap)      \# optional clamp

    Z\_hat10 \= Z\_lr10 \+ R10  
    return Z\_hat10

### **8.3 FiLM implementation (global)**

function apply\_film(F\_dem, F\_ae, alpha):  
    v \= global\_avg\_pool(F\_ae)                \# (B, C\_ae)  
    h \= MLP(v)                               \# (B, 2\*C\_dem)  
    gamma, beta \= split(h)                   \# each (B, C\_dem)  
    gamma \= tanh(gamma)  
    gamma \= gamma.view(B,C\_dem,1,1)  
    beta  \= beta.view(B,C\_dem,1,1)  
    return (1 \+ alpha\*gamma) \* F\_dem \+ alpha\*beta

### **8.4 Loss computation**

function compute\_loss(Z\_hat10, Z\_gt10, W):  
    L\_elev \= mean( W \* smoothL1(Z\_hat10 \- Z\_gt10) )

    S\_hat \= slope(Z\_hat10)   \# finite diffs  
    S\_gt  \= slope(Z\_gt10)  
    L\_slope \= mean( W \* abs(S\_hat \- S\_gt) )

    return L\_elev \+ lambda\_slope \* L\_slope

---

## **9\) Ablations (planned experiments)**

You’ll want these to justify design decisions:

1. **No AE** (DEM-only)  
2. **Concat AE** (simple early fusion) vs **FiLM** vs **Plan B1 gated fusion**  
3. Global FiLM vs Spatial FiLM vs **Plan B2 cross-attention** (coarse scales only)  
4. Loss: elevation only vs elevation \+ slope  
5. Masking:  
   * none  
   * building only  
   * water only  
   * both \+ shoreline downweight  
6. Guidance dropout on/off

---

## **10\) Risk register and mitigations**

### **Risk: AE causes DSM-like hallucinations**

Mitigations:

* residual learning  
* global FiLM (not spatial)  
* guidance dropout  
* mask-based trust gating

### **Risk: time mismatch / real-world change corrupts supervision**

Mitigations:

* water masking/downweighting with year-window when possible  
* robust loss (SmoothL1)  
* clip outlier residuals for stability

### **Risk: inconsistent hydro-flattening conventions**

Mitigations:

* persistent water weight=0  
* shoreline band downweight  
* optional water smoothness regularizer

### **Risk: seam artifacts in wall-to-wall inference**

Mitigation:

* overlap tiling \+ blending window

---

## **11\) Implementation notes**

* Keep the model fully convolutional so it can run on arbitrarily large tiles.  
* Start with mixed precision training; AE channels increase memory footprint.  
* Log modulation stats:  
  * mean |alpha\*gamma| per scale  
  * performance with AE dropped at val time (sanity check)

---

## **12\) Roadmap to v2**

* Add spatial FiLM only if needed for small-scale terrain transitions.  
* Add a lightweight “acquisition-year embedding” conditioning if you detect year-specific bias.  
* Add a shallow auxiliary head to predict building/water as multitask supervision (optional).  
* Consider uncertainty estimation (MC dropout) to produce confidence outputs.

