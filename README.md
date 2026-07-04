# sff-surface-lab

**Shape-from-Focus 3D Surface Reconstruction for Milling-Induced Surface Topology**

> Research internship project — Chatter prediction in milling operations  
> Supervisor: Dr. R.K. Mittal, IIT Guwahati

---

## Overview

During milling, chatter (unwanted tool vibration) leaves characteristic microscopic deformations on the workpiece — cavities, grooves, and irregular surface topology. Accurately measuring these features in 3D is key to predicting and diagnosing chatter.

This pipeline converts **focus-stacked microscope images** into metrically calibrated 3D surface models using **Shape-from-Focus (SFF)** — a classical computer vision technique that exploits the shallow depth-of-field of microscope optics to recover precise depth information.

---

## How It Works

A focus stack is a sequence of images captured at fixed XY position while the focal plane is stepped through the sample at a known interval (1 µm/frame). Because only the surface point that coincides with the focal plane is sharp, each pixel is brightest/sharpest in exactly one frame — and that frame index gives us the depth.

**Core equation:**
```
Z(x, y) = argmax_k [ Φ(I_k(x, y)) ] × ΔZ
```
where `Φ` is the focus measure operator and `ΔZ = 1 µm` is the Z-step.

### Key Algorithms

| Stage | Technique | Why |
|-------|-----------|-----|
| Frame alignment | ECC registration (Evangelidis & Psarakis, 2008) | Robust to blur; corrects mechanical XY stage drift |
| Focus measure | Sum-Modified-Laplacian (Nayar & Nakagawa, 1994) | Avoids sign-cancellation of the standard Laplacian |
| Sub-frame depth | Log-Gaussian parabolic interpolation | Achieves sub-micron precision beyond the motor step limit |
| Smoothing | Joint Bilateral Filter | Smooths noise while preserving sharp cavity walls |
| Uncertainty | R² + peak prominence confidence score | Masks unreliable textureless regions |
| Validation | Split-half consistency test | Proves reconstruction stability without ground truth |

---

## Results

Tested on two focus stacks of milled particles:

| Dataset | Frames | Z Range | Valid Pixels | Split-Half MAE |
|---------|--------|---------|--------------|----------------|
| MAIN-SET1 | 81 | 80 µm | 89.6% | **1.162 µm** |
| leaf2 | 101 | 100 µm | — | — |

The **1.162 µm MAE** on MAIN-SET1 confirms sub-micron precision is achieved through Gaussian interpolation — well below the 1 µm physical motor step.

---

## Project Structure

```
sff-surface-lab/
├── src/
│   ├── config.py            # All parameters in one place
│   ├── loader.py            # Image stack ingestion
│   ├── drift_check.py       # Input validation (3 independent checks)
│   ├── registration.py      # ECC frame alignment
│   ├── focus_measure.py     # SML / Tenengrad / Laplacian
│   ├── depth_estimation.py  # Sub-frame Gaussian interpolation
│   ├── smoothing.py         # Joint Bilateral / Guided filtering
│   ├── confidence.py        # Per-pixel uncertainty scoring
│   ├── validation.py        # Split-half & ground-truth testing
│   ├── calibration.py       # XY pixel-size calibration
│   └── visualization.py     # Depth maps, cross-sections, 3D HTML
├── scripts/
│   ├── 00_extract_data.py         # Unzip raw image archives
│   ├── 01_validate_stack.py       # Run drift & quality checks
│   ├── 02_register_frames.py      # ECC alignment (if needed)
│   ├── 03_run_pipeline.py         # Full reconstruction
│   ├── 04_compare_focus_measures.py  # Benchmark SML vs others
│   └── 05_validate_results.py     # Split-half / ground truth
├── data/                    # Raw image stacks (not in git — see data/README.md)
├── outputs/                 # Generated depth maps and 3D models
├── SFF_Reconstruction_Book.md  # 25-page technical deep-dive
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place image stacks in data/ (see data/README.md)

# 3. Validate input — check for drift
python scripts/01_validate_stack.py

# 4. Register frames (only if drift detected in step 3)
python scripts/02_register_frames.py

# 5. Run full reconstruction
python scripts/03_run_pipeline.py

# 6. Validate results (split-half consistency)
python scripts/05_validate_results.py --mode split-half
```

### Options

```bash
# Choose focus measure
python scripts/03_run_pipeline.py --method sml|laplacian|tenengrad

# Use guided filter instead of bilateral
python scripts/03_run_pipeline.py --smoothing guided

# Process a specific dataset only
python scripts/03_run_pipeline.py --dataset MAIN-SET1
```

---

## Configuration

All tunable parameters live in [`src/config.py`](src/config.py). Key defaults:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `Z_STEP_UM` | `1.0` µm | Physical stage step per frame |
| `XY_UM_PER_PIXEL` | `None` | Set after microscope calibration |
| `sml_window_size` | `5` | 5×5 pooling window for SML |
| `interp_half_width` | `2` | ±2 frames → 5-point Gaussian fit |
| `bilateral_d` | `9` | Bilateral filter diameter (pixels) |
| `confidence_threshold` | `0.3` | Below this, pixel depth is masked |

---

## Outputs

```
outputs/
├── diagnostics/<dataset>/
│   ├── global_focus_energy.png   # Z-energy curve (unimodality check)
│   ├── local_focus_energy.png    # Per-tile energy (spatial uniformity)
│   └── drift_tracking.png        # Feature displacement over frames
├── depth_maps/<dataset>/
│   ├── depth_raw.npy             # Raw depth map (float32, µm)
│   ├── depth_smoothed.npy        # Bilaterally smoothed depth map
│   ├── confidence.npy            # Per-pixel confidence [0, 1]
│   ├── *_all_in_focus.png        # Extended depth-of-field composite
│   ├── *_depth_map.png           # Colormapped depth heatmap
│   ├── *_confidence.png          # Confidence overlay
│   ├── *_cross_sections.png      # Depth profiles (X and Y slices)
│   └── *_dashboard.png           # Multi-panel diagnostic summary
└── 3d_models/
    └── *_3d_surface.html         # Interactive 3D surface (Plotly, open in browser)
```

---

## References

1. Nayar, S.K. & Nakagawa, Y. (1994). *Shape from Focus.* IEEE TPAMI, 16(8), 824–831.
2. Pertuz, S., Puig, D., & Garcia, M.A. (2013). *Analysis of focus measure operators for shape-from-focus.* Pattern Recognition, 46(5), 1415–1432.
3. Evangelidis, G.D. & Psarakis, E.Z. (2008). *Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization.* IEEE TPAMI, 30(10), 1858–1865.
4. He, K., Sun, J., & Tang, X. (2013). *Guided Image Filtering.* IEEE TPAMI, 35(6), 1397–1409.

---

## License

Academic/research use. Please cite the references above if this methodology informs published work.
