"""
smoothing.py — Edge-aware depth map smoothing.

Step 5 of the pipeline: Smooth the raw depth map to reduce noise while
preserving sharp depth transitions at cavity edges.

WHY edge-aware over simple Gaussian/median:
A cavity in the milled surface has a sharp depth drop at its edge — exactly
the feature we need to measure. Gaussian blur smears this edge outward,
making the cavity appear wider and shallower than it really is. Bilateral
and guided filters use intensity similarity (from the all-in-focus composite)
to keep smoothing within regions of similar appearance, leaving edges intact.

Two methods:
1. Joint bilateral filter — uses the composite image as a guide
2. Guided filter — faster alternative, avoids gradient reversal artifacts

REF: Tomasi & Manduchi, "Bilateral filtering", ICCV 1998
REF: He et al., "Guided Image Filtering", IEEE TPAMI 35(6), 2013
"""

import cv2
import numpy as np

from .config import SMOOTH_CFG


def smooth_bilateral(
    depth_map: np.ndarray,
    guide_image: np.ndarray,
    d: int = None,
    sigma_color: float = None,
    sigma_space: float = None,
) -> np.ndarray:
    """
    Edge-aware smoothing using joint bilateral filter.

    Parameters
    ----------
    depth_map : np.ndarray, shape (H, W), float32
        Raw depth map (may contain NaN for invalid pixels).
    guide_image : np.ndarray, shape (H, W), float32
        All-in-focus composite — provides edge information.

    Returns
    -------
    smoothed : np.ndarray, shape (H, W), float32
    """
    if d is None:
        d = SMOOTH_CFG.bilateral_d
    if sigma_color is None:
        sigma_color = SMOOTH_CFG.bilateral_sigma_color
    if sigma_space is None:
        sigma_space = SMOOTH_CFG.bilateral_sigma_space

    nan_mask = np.isnan(depth_map)
    depth_filled = depth_map.copy()
    if np.any(nan_mask):
        from scipy.ndimage import median_filter
        median_depth = median_filter(
            np.nan_to_num(depth_map, nan=np.nanmedian(depth_map)),
            size=5
        )
        depth_filled[nan_mask] = median_depth[nan_mask]

    guide_float32 = guide_image.astype(np.float32)

    smoothed = cv2.ximgproc.jointBilateralFilter(
        joint=guide_float32,
        src=depth_filled,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    smoothed[nan_mask] = np.nan

    return smoothed


def smooth_guided(
    depth_map: np.ndarray,
    guide_image: np.ndarray,
    radius: int = None,
    eps: float = None,
) -> np.ndarray:
    """
    Edge-aware smoothing using guided filter.

    ALTERNATIVE to bilateral. Advantages:
    - O(N) complexity vs O(N·d²) for bilateral
    - No gradient reversal artifacts
    - Analytically exact (not iterative)

    Disadvantage:
    - Slightly less sharp edge preservation in some cases

    Parameters
    ----------
    depth_map : np.ndarray, shape (H, W), float32
    guide_image : np.ndarray, shape (H, W), float32

    Returns
    -------
    smoothed : np.ndarray, shape (H, W), float32
    """
    if radius is None:
        radius = SMOOTH_CFG.guided_radius
    if eps is None:
        eps = SMOOTH_CFG.guided_eps

    nan_mask = np.isnan(depth_map)
    depth_filled = depth_map.copy()
    if np.any(nan_mask):
        from scipy.ndimage import median_filter
        median_depth = median_filter(
            np.nan_to_num(depth_map, nan=np.nanmedian(depth_map)),
            size=5
        )
        depth_filled[nan_mask] = median_depth[nan_mask]

    smoothed = cv2.ximgproc.guidedFilter(
        guide=guide_image,
        src=depth_filled,
        radius=radius,
        eps=eps,
    )

    smoothed[nan_mask] = np.nan
    return smoothed


def smooth_depth_map(
    depth_map: np.ndarray,
    guide_image: np.ndarray,
    method: str = None,
) -> np.ndarray:
    """
    Smooth a depth map using the configured method.

    Parameters
    ----------
    depth_map : np.ndarray, shape (H, W), float32
    guide_image : np.ndarray, shape (H, W), float32
    method : str, "bilateral" or "guided"

    Returns
    -------
    smoothed : np.ndarray, shape (H, W), float32
    """
    if method is None:
        method = SMOOTH_CFG.method

    if method == "bilateral":
        return smooth_bilateral(depth_map, guide_image)
    elif method == "guided":
        return smooth_guided(depth_map, guide_image)
    else:
        raise ValueError(f"Unknown smoothing method: {method}. Use 'bilateral' or 'guided'.")


def compare_smoothing_methods(
    depth_map: np.ndarray,
    guide_image: np.ndarray,
) -> dict:
    """
    Run both smoothing methods and return comparison metrics.

    Returns
    -------
    results : dict with keys:
        - bilateral: smoothed depth map
        - guided: smoothed depth map
        - difference: absolute difference between the two
        - max_diff: maximum difference in µm
        - mean_diff: mean difference in µm
    """
    bilateral = smooth_bilateral(depth_map, guide_image)
    guided = smooth_guided(depth_map, guide_image)

    diff = np.abs(bilateral - guided)
    valid = ~np.isnan(diff)

    return {
        "bilateral": bilateral,
        "guided": guided,
        "difference": diff,
        "max_diff": float(np.nanmax(diff)) if np.any(valid) else 0.0,
        "mean_diff": float(np.nanmean(diff)) if np.any(valid) else 0.0,
    }
