"""
depth_estimation.py — Depth map extraction from focus volumes.

Step 4 of the pipeline: Convert the focus volume FM(x, y, z) into a
depth map D(x, y) by finding the frame index where each pixel is sharpest,
then refining with sub-frame Gaussian interpolation.

The naive approach (argmax) gives integer frame indices → depth resolution
locked to the focus step size (1 µm). Gaussian interpolation recovers
sub-step precision from the shape of the focus-measure curve.

REF: Nayar & Nakagawa, "Shape from Focus", IEEE TPAMI 16(8), 1994, §4.1
     They model the focus-measure curve as Gaussian and fit in log-space.
"""

import numpy as np
from tqdm import tqdm

from .config import DEPTH_CFG, Z_STEP_UM


def estimate_depth_argmax(
    focus_volume: np.ndarray,
    z_step_um: float = None,
) -> np.ndarray:
    """
    Estimate depth map using simple argmax (no interpolation).

    This is the baseline — integer frame resolution only.
    Kept as a reference to show how much sub-frame interpolation improves.

    Parameters
    ----------
    focus_volume : np.ndarray, shape (N, H, W)

    Returns
    -------
    depth_map : np.ndarray, shape (H, W), float32
        Depth in microns.
    """
    if z_step_um is None:
        z_step_um = Z_STEP_UM

    frame_indices = np.argmax(focus_volume, axis=0)
    depth_map = frame_indices.astype(np.float32) * z_step_um

    return depth_map


def estimate_depth_gaussian(
    focus_volume: np.ndarray,
    z_step_um: float = None,
    half_width: int = None,
    min_focus: float = None,
    require_concave: bool = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate depth with sub-frame Gaussian interpolation.

    For each pixel:
    1. Find k* = argmax(FM[:, y, x]) — coarse peak
    2. Extract FM values in a window [k*-w, k*+w]
    3. Fit ln(FM) = a·k² + b·k + c (parabola in log-space = Gaussian)
    4. Sub-frame peak: k_sub = -b / (2a)
    5. Depth: z = k_sub × z_step_um

    WHY Gaussian (not parabola on raw FM):
    The defocus PSF is approximately Gaussian, so the focus-measure response
    as a function of defocus distance is also approximately Gaussian.
    Fitting in log-space linearizes this model.

    ALTERNATIVE: Parabola fit on raw FM values. This works too (and is what
    many implementations do), but it's less physically motivated and tends
    to give slightly wider peaks → less precise interpolation. We implement
    both under the hood and use Gaussian by default.

    Parameters
    ----------
    focus_volume : np.ndarray, shape (N, H, W), float64

    Returns
    -------
    depth_map : np.ndarray, shape (H, W), float32
        Sub-frame depth in microns. NaN where estimation failed.
    r_squared : np.ndarray, shape (H, W), float32
        Goodness-of-fit (R²) for the Gaussian fit. Used for confidence.
    fit_valid : np.ndarray, shape (H, W), bool
        True where the Gaussian fit produced a valid (concave) peak.
    """
    if z_step_um is None:
        z_step_um = Z_STEP_UM
    if half_width is None:
        half_width = DEPTH_CFG.interp_half_width
    if min_focus is None:
        min_focus = DEPTH_CFG.min_focus_measure
    if require_concave is None:
        require_concave = DEPTH_CFG.require_concave_fit

    n_frames, h, w = focus_volume.shape
    depth_map = np.full((h, w), np.nan, dtype=np.float32)
    r_squared = np.zeros((h, w), dtype=np.float32)
    fit_valid = np.zeros((h, w), dtype=bool)

    coarse_peaks = np.argmax(focus_volume, axis=0)  # (H, W)
    peak_values = np.max(focus_volume, axis=0)       # (H, W)


    total_pixels = h * w
    processed = 0

    print(f"Estimating depth with Gaussian interpolation (±{half_width} frames)...")

    for y in tqdm(range(h), desc="Depth estimation", unit="row"):
        for x in range(w):
            k_star = coarse_peaks[y, x]
            fm_peak = peak_values[y, x]

            if fm_peak < min_focus:
                continue

            k_lo = max(0, k_star - half_width)
            k_hi = min(n_frames - 1, k_star + half_width)

            if k_hi - k_lo < 2:
                depth_map[y, x] = k_star * z_step_um
                fit_valid[y, x] = False
                continue

            k_vals = np.arange(k_lo, k_hi + 1, dtype=np.float64)
            fm_vals = focus_volume[k_lo:k_hi + 1, y, x]

            if np.any(fm_vals <= 0):
                depth_map[y, x] = k_star * z_step_um
                fit_valid[y, x] = False
                continue

            log_fm = np.log(fm_vals)

            try:
                coeffs = np.polyfit(k_vals, log_fm, 2)
                a, b, c = coeffs

                if require_concave and a >= 0:
                    depth_map[y, x] = k_star * z_step_um
                    fit_valid[y, x] = False
                    continue

                k_sub = -b / (2 * a)

                if abs(k_sub - k_star) > half_width + 0.5:
                    depth_map[y, x] = k_star * z_step_um
                    fit_valid[y, x] = False
                    continue

                k_sub = np.clip(k_sub, 0, n_frames - 1)

                depth_map[y, x] = k_sub * z_step_um
                fit_valid[y, x] = True

                log_fm_pred = np.polyval(coeffs, k_vals)
                ss_res = np.sum((log_fm - log_fm_pred) ** 2)
                ss_tot = np.sum((log_fm - np.mean(log_fm)) ** 2)
                r_squared[y, x] = 1.0 - ss_res / (ss_tot + 1e-10)

            except (np.linalg.LinAlgError, ValueError):
                depth_map[y, x] = k_star * z_step_um
                fit_valid[y, x] = False

    valid_mask = ~np.isnan(depth_map)
    n_valid = np.sum(valid_mask)
    n_fitted = np.sum(fit_valid)
    print(f"  Valid pixels: {n_valid}/{total_pixels} ({100*n_valid/total_pixels:.1f}%)")
    print(f"  Gaussian-fitted: {n_fitted}/{total_pixels} ({100*n_fitted/total_pixels:.1f}%)")
    if n_valid > 0:
        print(f"  Depth range: {np.nanmin(depth_map):.2f} to {np.nanmax(depth_map):.2f} µm")
        print(f"  Mean R²: {np.mean(r_squared[fit_valid]):.4f}" if n_fitted > 0 else "")

    return depth_map, r_squared, fit_valid


def build_all_in_focus(stack: np.ndarray, focus_volume: np.ndarray) -> np.ndarray:
    """
    Build an all-in-focus (extended depth of field) composite image.

    For each pixel, select the intensity from the frame where the focus
    measure is highest. This gives a sharp image of the entire surface.

    WHY: Used as the guide image for edge-aware smoothing (Step 5),
    and as a standalone output for visualization.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W), float32
    focus_volume : np.ndarray, shape (N, H, W), float64

    Returns
    -------
    composite : np.ndarray, shape (H, W), float32
    """
    best_frames = np.argmax(focus_volume, axis=0)  # (H, W)
    h, w = best_frames.shape
    composite = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            composite[y, x] = stack[best_frames[y, x], y, x]

    return composite
