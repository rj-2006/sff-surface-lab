# Mathematical Analysis of the SFF Pipeline

> A step-by-step breakdown of every algorithm used in the Shape-from-Focus 3D Surface Reconstruction pipeline — with both an intuitive explanation and the full mathematical detail for each stage.

---

## Table of Contents
1. [Input Validation & Drift Detection](#1-input-validation--drift-detection)
2. [ECC Image Registration](#2-ecc-image-registration)
3. [Sum-Modified-Laplacian Focus Measure](#3-sum-modified-laplacian-sml-focus-measure)
4. [Sub-Frame Gaussian Depth Interpolation](#4-sub-frame-gaussian-depth-interpolation)
5. [Confidence Scoring](#5-confidence-scoring)
6. [Joint Bilateral Filtering](#6-joint-bilateral-filtering)
7. [Split-Half Consistency Validation](#7-split-half-consistency-validation)
8. [Algorithm Reference Card](#8-algorithm-reference-card)

---

## The Core Idea

> A microscope at high magnification has an extremely shallow depth of field — only a thin slice of the scene is sharp at any moment. We exploit this as a depth sensor. By moving the focal plane through the sample in 1 µm steps and taking a photo at each step, we create a focus stack. For each pixel, we find the one frame where it is sharpest. That frame index × 1 µm = that pixel's physical depth. The pipeline then refines this to sub-micron precision using mathematics.

**Fundamental depth equation:**

```
Z(x, y) = argmax_k [ Φ(I_k(x, y)) ] × ΔZ
```

where `Φ` is a focus-measure operator, `I_k(x, y)` is pixel intensity at position `(x, y)` in frame `k`, and `ΔZ = 1.0 µm` is the physical Z-step.

---

## 1. Input Validation & Drift Detection

**Source file:** `src/drift_check.py`

### Intuitive Explanation

Imagine taking a video of a coin while slowly lifting a magnifying glass. If someone nudges the camera sideways between shots, the coin will appear to move. Before measuring depth, we check whether the camera moved and by how much. We run three independent tests to verify the SFF assumption holds — that *only* the Z-plane changes between frames, with no XY movement.

### Mathematical Detail

#### 1a. Global Focus Energy

For each frame `k`, we compute the **mean Laplacian energy**:

```
E_k = (1 / H·W) × Σ(x,y) [ ∇²I_k(x,y) ]²
```

The Laplacian `∇²I = ∂²I/∂x² + ∂²I/∂y²` is a differential operator measuring local intensity curvature. Sharp textures → large curvatures. Blurred images → near-zero curvature. Squaring ensures sign-independence.

We expect `E_k` to form a single smooth unimodal peak. Multiple peaks indicate vibration, sample layering, or stage anomalies.

#### 1b. Harris Corner Detection for Drift Tracking

The **Harris Corner Response** is computed as:

```
R(x,y) = det(M) − k · tr(M)²
```

where `M` is the **structure tensor** — a 2×2 matrix of local image gradients:

```
M = Σ_{(u,v) ∈ W} | I_x²    I_x·I_y |
                   | I_x·I_y  I_y²   |
```

`R > 0` indicates a corner. These corners are tracked across frames using **Normalised Cross-Correlation (NCC)**:

```
NCC(u,v) = Σ(x,y) [T(x,y) − T̄][I(x+u, y+v) − Ī]
           ─────────────────────────────────────────
                    √(Σ T² · Σ I²)
```

Maximum NCC gives the new position of each feature. Drift is the Euclidean displacement from its reference position. Threshold: **1.0 px** — exceeding this triggers registration.

**Results on our datasets:**
| Dataset | Max Drift | Action Taken |
|---------|-----------|--------------|
| MAIN-SET1 | 4.24 px | Registration required |
| leaf2 | 14.14 px | Registration required |

### Why This Matters

If drift is ignored, SFF assigns the wrong depth to the wrong physical point, creating shear artifacts — the surface appears to slope when it should be flat.

---

## 2. ECC Image Registration

**Source file:** `src/registration.py`

### Intuitive Explanation

Think of alignment as a sliding puzzle. You know what the final picture should look like (the sharpest frame). You slide each other frame left/right/up/down until it overlaps as perfectly as possible with the reference. ECC is a mathematically precise way to do that sliding — and it works even when the image is blurry.

### Mathematical Detail

We find warp parameters `p` (translation `[t_x, t_y]`) that maximise the **Enhanced Correlation Coefficient**:

```
ρ(p) = ⟨ Î_ref ,  W(I; p)̂ ⟩
       ─────────────────────────
       ‖ Î_ref ‖ · ‖ Ŵ(I; p) ‖
```

where `Î` denotes the mean-subtracted image `(I − Ī)`. The normalisation makes this **robust to illumination changes** — critical since frame brightness varies as the focal plane moves.

Evangelidis & Psarakis (2008) derived a closed-form update from a Taylor expansion of the warp:

```
Δp = (Jᵀ J)⁻¹ Jᵀ Î_ref
```

where `J` is the **image Jacobian** — gradient of `W(I; p)` with respect to `p`. We iterate 50–100 times until convergence (Δp < 10⁻³).

### Why ECC over SIFT / ORB Feature Matching

| Property | ECC | SIFT / ORB |
|---|---|---|
| Works on blurred frames | ✓ | ✗ |
| Pixel-level precision | ✓ | Approximate |
| Fails with no keypoints | No | Yes (most frames are defocused) |
| Robust to illumination change | ✓ | Partial |

Feature detectors need textured, in-focus regions. Most frames in a focus stack are severely blurred. ECC operates directly on the entire pixel array and degrades gracefully.

---

## 3. Sum-Modified-Laplacian (SML) Focus Measure

**Source file:** `src/focus_measure.py`

### Intuitive Explanation

You look at the same spot through a camera and slowly change focus. At one specific distance, that spot is perfectly sharp — maximum contrast, maximum detail. We measure "how sharp" each pixel is at each depth. The depth where it's sharpest is its real depth. But we need a careful mathematical measure of sharpness that doesn't get fooled by diagonal edges.

### Mathematical Detail

#### The Modified Laplacian

We discretely approximate the absolute second directional derivatives:

```
ML(x,y) = | I(x+s, y) − 2I(x,y) + I(x−s, y) |
         + | I(x, y+s) − 2I(x,y) + I(x, y−s) |
```

Parameters: step size `s = 1 px`.

#### Why Absolute Values? The Cancellation Problem

The **standard Laplacian** is:
```
∇²I = ∂²I/∂x² + ∂²I/∂y²   ← addition, not absolute value
```

On a 45° diagonal edge, `∂²I/∂x²` and `∂²I/∂y²` are equal and opposite — they cancel to zero. A genuinely sharp diagonal edge registers as zero sharpness. This is a fundamental flaw.

The **Modified Laplacian** takes the absolute value of each term before summing:
```
ML = |∂²I/∂x²| + |∂²I/∂y²|   ← absolute value of each separately
```

No cancellation is possible. Every orientation of edge contributes positively.

#### Spatial Windowing — The SML Sum

```
SML(x,y) = Σ_{(m,n) ∈ W_{5×5}}  ML(m,n),   where ML(m,n) ≥ T
```

Parameters: `5×5` window, threshold `T = 0.0`.

**Why windowing:** Focus is a local property — a single isolated pixel has no texture, so it cannot be measured as sharp or blurry by itself. The window accumulates sharpness evidence from neighbours.

**Why 5×5:** Nayar & Nakagawa (1994) recommend 3–7. A 3×3 window is over-sensitive to noise. A 9×9 window blurs depth boundaries spatially. 5×5 is the empirical optimum.

This produces the **focus volume** `F ∈ ℝ^{N × H × W}` — the mathematical foundation of the depth map.

---

## 4. Sub-Frame Gaussian Depth Interpolation

**Source file:** `src/depth_estimation.py`

### Intuitive Explanation

Your focus dial has tick marks every 1 µm. The sharpest point might actually be *between* two tick marks. Instead of just reading the nearest tick, we look at how sharp the image is at the ticks on either side and draw a smooth curve through those values. The very top of that curve tells us the precise location — even between ticks.

### Mathematical Detail

#### Argmax Baseline

```
Z_argmax(x,y) = argmax_k  F_k(x,y)  ×  ΔZ
```

Depth is quantised to integer multiples of ΔZ = 1 µm. This is the *maximum possible resolution* from hardware alone.

#### The Physics: Why Gaussian?

A microscope's **Point Spread Function (PSF)** is approximately Gaussian. The focus measure response of a pixel as a function of depth follows:

```
F(z) ≈ A · exp( −(z − z₀)² / 2σ² )
```

where `z₀` is the true depth and `σ` relates to the depth of field.

#### Log-Space Linearisation

Taking the natural logarithm of both sides:

```
ln F(z) = ln A − (z − z₀)² / 2σ²
        = az² + bz + c             ← a parabola in log-space
```

where:
- `a = −1 / (2σ²) < 0`  (must be negative — downward-opening parabola)
- `b = z₀ / σ²`
- `c = ln A − z₀² / 2σ²`

#### Least Squares Polynomial Fit

For each pixel, we take frame `k*` with peak focus and its ±2 neighbours — **5 data points**. We fit a 2nd-degree polynomial in log-space:

```
min_{a,b,c}  Σᵢ₌₁⁵  ( ln F_{kᵢ} − a·kᵢ² − b·kᵢ − c )²
```

Solved via the **Normal Equations**: `(XᵀX)θ = Xᵀy`, implemented as `numpy.polyfit`.

#### Analytic Sub-Frame Peak

The vertex of the fitted parabola gives the true depth:

```
z_peak = (−b / 2a) × ΔZ,    valid only if a < 0
```

If `a ≥ 0`, the parabola opens upward — no real focus peak — and we fall back to argmax.

#### Fit Quality: R²

```
R² = 1 − Σᵢ(ln F_{kᵢ} − ŷᵢ)² / Σᵢ(ln F_{kᵢ} − ln F̄)²
```

Stored per-pixel and passed to confidence scoring.

**Result on MAIN-SET1:**
- Mean sub-frame shift: **0.51 µm**
- Max sub-frame shift: **2.50 µm**

These pixels gained precision that no hardware upgrade could provide.

---

## 5. Confidence Scoring

**Source file:** `src/confidence.py`

### Intuitive Explanation

Not all depth measurements are equally believable. If you're measuring depth on a completely smooth, featureless patch of metal, there's nothing to focus on — any depth reading is just noise. The confidence score flags which pixels we actually trust, so garbage pixels don't silently contaminate the 3D model.

### Mathematical Detail

Three independent signals are combined multiplicatively:

```
C(x,y) = C_prom(x,y) × C_R²(x,y) × C_edge(x,y)
```

#### Signal 1: Peak Prominence

Derived from the parabola's curvature coefficient `a`:

```
C_prom(x,y) = clip( (|a(x,y)| − |a|_min) / (|a|_99th − |a|_min),  0,  1 )
```

A flat, textureless surface has `a ≈ 0` — no real focus peak exists.

#### Signal 2: Fit Quality

```
C_R²(x,y) = max(0,  R²(x,y))
```

High R² = data closely follows the Gaussian model = genuine optical focus.

#### Signal 3: Edge / Boundary Penalty

```
C_edge(x,y) = 0  if k*(x,y) ∈ {0, N−1}
             = 1  otherwise
```

Pixels whose focus peak is at the first or last frame have their true in-focus depth outside the scanned Z-range — the measurement is clipped, not measured.

#### Masking

```
If C(x,y) < 0.3  →  Z(x,y) = NaN
```

**Why multiplicative, not additive:**  
If any single factor is zero (zero R², zero prominence, or an edge pixel), the whole confidence collapses to zero. We do not want a pixel with a perfect parabolic fit but zero prominence to be considered reliable — all three physical conditions must be satisfied simultaneously.

**Result on MAIN-SET1:** 30.1% of pixels rejected (92,545 / 307,200) — flat background regions and low-texture areas of the particle surface.

---

## 6. Joint Bilateral Filtering

**Source file:** `src/smoothing.py`

### Intuitive Explanation

Imagine smoothing a bumpy road with a steamroller. A regular steamroller flattens everything equally — including real speed bumps (the features we care about). The Joint Bilateral Filter is a smart steamroller that checks an aerial photo of the road before rolling. It only flattens where the photo shows flat ground, and stops at any real feature it sees in the photo.

### Mathematical Detail

Standard **Gaussian blur** (what we explicitly avoid):

```
I_smooth(x,y) = Σ_{(u,v) ∈ W}  G_s(u,v) · I(x+u, y+v)
```

where `G_s` is a purely spatial Gaussian kernel — it smooths everything, including cavity walls.

**Joint Bilateral Filter** adds a range kernel guided by the all-in-focus image `I_guide`:

```
D_smooth(x,y) = (1 / W_p) × Σ_{(u,v) ∈ W}
    G_s(u,v)
  · G_r( I_guide(x+u, y+v) − I_guide(x,y) )
  · D(x+u, y+v)
```

where:

```
G_s(u,v)  = exp( −(u² + v²) / 2σ_s² )      spatial Gaussian,  diameter d = 9 px
G_r(ΔI)   = exp( −ΔI² / 2σ_r² )            range Gaussian,    σ_r = 25
W_p       = Σ G_s · G_r                      normalisation
```

**Key insight:**  
When `ΔI` is large — i.e. the guide image shows a sharp intensity edge at the rim of a cavity — `G_r → 0`. The neighbour on the other side of the edge contributes almost nothing to the smoothed depth value. The filter **stops at the cavity wall**.

**Why use the all-in-focus image as the guide:**  
It contains every real structural edge of the sample at full resolution. Depth discontinuities at cavity walls correspond exactly to intensity edges in the all-in-focus image. The filter stops at the correct physical boundary.

---

## 7. Split-Half Consistency Validation

**Source file:** `src/validation.py`

### Intuitive Explanation

Ask two groups of students to solve the same problem using different halves of the data. If both groups arrive at the same answer independently, the method works. If they disagree wildly, something is broken. We split our frame stack into odd and even frames, reconstruct depth independently from each, and measure how closely they agree.

### Mathematical Detail

#### Stack Splitting

```
S_odd  = { I₁, I₃, I₅, … }     (N/2 frames, effective Z-step = 2ΔZ)
S_even = { I₀, I₂, I₄, … }     (N/2 frames, effective Z-step = 2ΔZ)
```

Each sub-stack goes through the full `SML → Gaussian interpolation → confidence masking → bilateral smoothing` pipeline independently.

#### Offset Correction

The even sub-stack starts at frame 0, the odd at frame 1. Since each frame step = 1 µm:

```
Z_even_corrected = Z_even + ΔZ
```

#### Combined Reliability Mask

A pixel is included in statistics only if it is reliable in **both** sub-stacks:

```
M_both(x,y) = M_odd(x,y) AND M_even(x,y)
```

This was the **bug fixed in the latest commit** (`b450ec6`) — previously the validation function skipped confidence masking, letting unreliable pixels silently inflate or deflate the MAE.

#### Agreement Metrics

```
MAE  = (1 / |M_both|) × Σ_{(x,y) ∈ M_both}  | Z_odd(x,y) − Z_even_corrected(x,y) |

RMSE = √[ (1 / |M_both|) × Σ ( Z_odd − Z_even_corrected )² ]

r    = Σ(Z_odd − Z̄_odd)(Z_even − Z̄_even)
       ────────────────────────────────────────────────────
       √[ Σ(Z_odd − Z̄_odd)² · Σ(Z_even − Z̄_even)² ]
```

**Classification:**

| MAE Range | Classification |
|-----------|---------------|
| < 2 µm | ✓ Excellent |
| 2 – 5 µm | Acceptable |
| ≥ 5 µm | Poor — check focus curves |

**MAIN-SET1 result: MAE = 1.162 µm → Excellent**

#### Why This Proves the Algorithm Works

Each sub-stack has half the frames and double the Z-step. The Gaussian interpolation is working harder. The fact that two independently-derived depth maps agree to 1.162 µm — without any shared data — proves genuine depth signal is being recovered. A pure-noise reconstruction would give MAE on the order of the Z-step itself (≫ 2 µm).

---

## 8. Algorithm Reference Card

| Step | Algorithm | Mathematical Core | Why This One |
|------|-----------|------------------|--------------|
| Drift Detection | Harris + NCC Template Matching | Structure tensor eigenvalues; normalised cross-correlation | Robust without keypoints; works on partially defocused frames |
| Registration | ECC (Evangelidis & Psarakis, 2008) | Iterative gradient descent on normalised cross-correlation | Works on blurred frames where feature detectors fail |
| Focus Measure | Sum-Modified-Laplacian (SML) | Absolute-value finite-difference 2nd derivatives, windowed sum | No edge-orientation cancellation vs. standard Laplacian |
| Depth Estimation | Log-Gaussian Parabolic Fit | Least-squares polynomial fit; vertex at z = −b/2a | Sub-micron precision beyond hardware motor step |
| Confidence Scoring | Multiplicative R² × Prominence × Edge Penalty | Normalised parabola curvature + goodness-of-fit + boundary check | Any zero factor kills the pixel — all conditions must hold |
| Smoothing | Joint Bilateral Filter (Tomasi & Manduchi, 1998) | Spatial × range Gaussian kernels guided by all-in-focus image | Preserves cavity walls; naive Gaussian blur destroys them |
| Validation | Split-Half MAE | Independent reconstructions from disjoint frame sets | Internal ground truth without a profilometer |

---

## References

1. **Nayar, S.K. & Nakagawa, Y. (1994).** Shape from Focus. *IEEE TPAMI*, 16(8), 824–831.
2. **Pertuz, S., Puig, D., & Garcia, M.A. (2013).** Analysis of focus measure operators for shape-from-focus. *Pattern Recognition*, 46(5), 1415–1432.
3. **Evangelidis, G.D. & Psarakis, E.Z. (2008).** Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization. *IEEE TPAMI*, 30(10), 1858–1865.
4. **Tomasi, C. & Manduchi, R. (1998).** Bilateral Filtering for Gray and Color Images. *ICCV*, 839–846.
5. **He, K., Sun, J., & Tang, X. (2013).** Guided Image Filtering. *IEEE TPAMI*, 35(6), 1397–1409.
