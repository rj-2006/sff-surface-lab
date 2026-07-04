"""
focus_measure.py — Focus measure operators for Shape-from-Focus.

Step 3 of the pipeline: Compute per-pixel focus quality across all frames
to build a "focus volume" — a 3D array FM(x, y, z) where z is the frame index.
The frame with the highest FM for each pixel is where that point is in focus.

Three operators implemented:
1. Sum-Modified-Laplacian (SML) — the literature-standard, our primary method
2. Windowed Laplacian-squared energy — our old baseline for comparison
3. Tenengrad (Sobel gradient energy) — additional reference

REF: Nayar & Nakagawa, "Shape from Focus", IEEE TPAMI 16(8), 1994
REF: Pertuz et al., "Analysis of focus measure operators for SFF", 2013
"""

import cv2
import numpy as np
from tqdm import tqdm

from .config import FOCUS_CFG


def sum_modified_laplacian(
    stack: np.ndarray,
    window_size: int = None,
    step_size: int = None,
    threshold: float = None,
) -> np.ndarray:
    """
    Compute Sum-Modified-Laplacian (SML) focus measure for each frame.

    The SML operator (Nayar & Nakagawa 1994):
    1. Modified Laplacian:
       ML(x,y) = |I(x+s,y) - 2·I(x,y) + I(x-s,y)|
               + |I(x,y+s) - 2·I(x,y) + I(x,y-s)|
       where s = step_size

    2. Sum within window, keeping only values above threshold:
       SML(x,y) = Σ_{window} ML(u,v)  for ML(u,v) ≥ T

    WHY SML over standard Laplacian:
    The standard Laplacian ∇²I = ∂²I/∂x² + ∂²I/∂y² can cancel out at edges
    oriented at 45° (positive curvature in x cancels negative in y). The
    Modified Laplacian uses absolute values of each partial derivative
    separately, preventing this cancellation.

    WHY this is NOT a simple convolution:
    The threshold T and the absolute values make this a non-linear operator.
    We must compute ML pixel-by-pixel, then threshold, then sum in a window.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W), float32 [0, 1]
    window_size : int, default 5
    step_size : int, default 1
    threshold : float, default 0.0

    Returns
    -------
    focus_volume : np.ndarray, shape (N, H, W), float64
        SML focus measure for each pixel in each frame.
    """
    if window_size is None:
        window_size = FOCUS_CFG.sml_window_size
    if step_size is None:
        step_size = FOCUS_CFG.sml_step_size
    if threshold is None:
        threshold = FOCUS_CFG.sml_threshold

    n_frames, h, w = stack.shape
    focus_volume = np.zeros((n_frames, h, w), dtype=np.float64)

    kernel_x = np.zeros((1, 2 * step_size + 1), dtype=np.float64)
    kernel_x[0, 0] = 1.0
    kernel_x[0, step_size] = -2.0
    kernel_x[0, 2 * step_size] = 1.0

    kernel_y = kernel_x.T  # Same kernel, transposed for Y direction

    half_w = window_size // 2

    for i in tqdm(range(n_frames), desc="Computing SML", unit="frame"):
        frame = stack[i].astype(np.float64)

        lap_x = cv2.filter2D(frame, cv2.CV_64F, kernel_x)
        lap_y = cv2.filter2D(frame, cv2.CV_64F, kernel_y)

        ml = np.abs(lap_x) + np.abs(lap_y)

        if threshold > 0:
            ml[ml < threshold] = 0.0

        sml = cv2.boxFilter(
            ml, ddepth=cv2.CV_64F,
            ksize=(window_size, window_size),
            normalize=False,
        )

        focus_volume[i] = sml

    return focus_volume


def laplacian_energy(
    stack: np.ndarray,
    ksize: int = None,
    window_size: int = None,
) -> np.ndarray:
    """
    Compute windowed Laplacian-squared energy focus measure.

    FM(x,y) = Σ_{window} (∇²I(u,v))²

    This was our original (baseline) focus measure. Included for side-by-side
    comparison with SML to document the upgrade.

    WHY it's inferior to SML:
    The standard Laplacian can have sign cancellation at certain edge
    orientations. Squaring recovers magnitude but amplifies noise more than
    the absolute-value approach in SML.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W), float32 [0, 1]

    Returns
    -------
    focus_volume : np.ndarray, shape (N, H, W), float64
    """
    if ksize is None:
        ksize = FOCUS_CFG.laplacian_ksize
    if window_size is None:
        window_size = FOCUS_CFG.laplacian_window_size

    n_frames, h, w = stack.shape
    focus_volume = np.zeros((n_frames, h, w), dtype=np.float64)

    for i in tqdm(range(n_frames), desc="Computing Laplacian²", unit="frame"):
        frame = stack[i].astype(np.float64)

        lap = cv2.Laplacian(frame, cv2.CV_64F, ksize=ksize)

        lap_sq = lap ** 2

        fm = cv2.boxFilter(
            lap_sq, ddepth=cv2.CV_64F,
            ksize=(window_size, window_size),
            normalize=False,
        )
        focus_volume[i] = fm

    return focus_volume


def tenengrad(
    stack: np.ndarray,
    ksize: int = None,
    window_size: int = None,
) -> np.ndarray:
    """
    Compute Tenengrad (Sobel gradient energy) focus measure.

    FM(x,y) = Σ_{window} (Gx² + Gy²)  where Gx, Gy = Sobel gradients

    WHY include Tenengrad:
    Pertuz et al. (2013) found Tenengrad competitive with SML and sometimes
    more stable. Having three methods lets us test robustness: if all three
    agree on the depth map, we can be more confident.

    ALTERNATIVE: Variance of Laplacian, Gray-Level Variance — these are
    simpler but generally less accurate (Pertuz et al. 2013, Table III).

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W), float32 [0, 1]

    Returns
    -------
    focus_volume : np.ndarray, shape (N, H, W), float64
    """
    if ksize is None:
        ksize = FOCUS_CFG.tenengrad_ksize
    if window_size is None:
        window_size = FOCUS_CFG.tenengrad_window_size

    n_frames, h, w = stack.shape
    focus_volume = np.zeros((n_frames, h, w), dtype=np.float64)

    for i in tqdm(range(n_frames), desc="Computing Tenengrad", unit="frame"):
        frame = stack[i].astype(np.float64)

        gx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=ksize)

        grad_energy = gx ** 2 + gy ** 2

        fm = cv2.boxFilter(
            grad_energy, ddepth=cv2.CV_64F,
            ksize=(window_size, window_size),
            normalize=False,
        )
        focus_volume[i] = fm

    return focus_volume


def compute_all_focus_measures(stack: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute all three focus measures on the same stack.

    Returns
    -------
    measures : dict mapping name → focus_volume (N, H, W)
    """
    return {
        "SML": sum_modified_laplacian(stack),
        "Laplacian²": laplacian_energy(stack),
        "Tenengrad": tenengrad(stack),
    }
