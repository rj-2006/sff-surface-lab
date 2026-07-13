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

--- PATCH (2026-07-08) ---
FIX: The previous NaN-fill used a single global scalar (np.nanmedian of the
WHOLE depth map) to plug every invalid/low-confidence pixel before smoothing.
That creates an artificial step/cliff at every mask boundary — most visibly
around the image border, where registration + vignetting produce a ring of
invalid pixels. Because the bilateral filter is edge-aware by design, it
"sees" that fabricated step as a real edge and preserves it instead of
smoothing over it. Result: spikes ringing the border and inside any interior
low-confidence patch.

FIX: replaced the global-median fill with cv2.inpaint (Navier-Stokes),
which propagates depth values inward from the boundary of the invalid
region using its actual local neighbors — no fabricated discontinuity.

REF: Tomasi & Manduchi, "Bilateral filtering", ICCV 1998
REF: He et al., "Guided Image Filtering", IEEE TPAMI 35(6), 2013
REF: Bertalmio et al., "Navier-Stokes, Fluid Dynamics, and Image and Video
     Inpainting", CVPR 2001 (cv2.INPAINT_NS)
"""

import cv2
import numpy as np

from .config import SMOOTH_CFG


def _inpaint_invalid(depth_map: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    """
    Fill invalid (NaN / low-confidence) pixels via local inpainting.

    Replaces the old approach of filling with a single global median value,
    which fabricates a hard edge at the mask boundary. Inpainting instead
    propagates values inward from the valid boundary pixels, so the filled
    region blends with its actual local neighborhood.

    Parameters
    ----------
    depth_map : np.ndarray, shape (H, W), float32
        May contain NaN. Values outside NaN are assumed valid.
    invalid_mask : np.ndarray, shape (H, W), bool
        True where the pixel should be treated as invalid and inpainted,
        e.g. NaN pixels OR pixels the confidence mask rejected.

    Returns
    -------
    filled : np.ndarray, shape (H, W), float32
        Depth map with invalid regions filled by inpainting. No NaNs.
    """
    if not np.any(invalid_mask):
        return np.nan_to_num(depth_map, nan=0.0).astype(np.float32)

    # cv2.inpaint requires a finite source image — seed NaNs with the
    # nearest valid value via a cheap distance-transform fill first, so
    # inpainting has real numbers to propagate from at the mask edges.
    valid_mask = ~invalid_mask
    if not np.any(valid_mask):
        # Nothing valid at all — degenerate case, nothing sensible to return.
        return np.nan_to_num(depth_map, nan=0.0).astype(np.float32)

    from scipy.ndimage import distance_transform_edt
    seed = depth_map.copy()
    seed[invalid_mask] = 0.0
    seed = np.nan_to_num(seed, nan=0.0).astype(np.float32)

    _, (iy, ix) = distance_transform_edt(
        invalid_mask, return_distances=True, return_indices=True
    )
    nearest_fill = seed[iy, ix]
    seed[invalid_mask] = nearest_fill[invalid_mask]

    mask_u8 = invalid_mask.astype(np.uint8) * 255
    inpainted = cv2.inpaint(seed, mask_u8, inpaintRadius=5, flags=cv2.INPAINT_NS)

    return inpainted.astype(np.float32)


def smooth_bilateral(
    depth_map: np.ndarray,
    guide_image: np.ndarray,
    confidence_mask: np.ndarray = None,
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
    confidence_mask : np.ndarray, shape (H, W), bool, optional
        True = reliable pixel (from confidence.compute_confidence). Pixels
        where this is False are treated as invalid and inpainted, same as
        NaN pixels. Pass this in — without it, low-confidence pixels that
        aren't already NaN will still leak spikes through.

    Returns
    -------
    smoothed : np.ndarray, shape (H, W), float32
        NaN preserved only where BOTH raw depth was NaN and no confidence
        mask was supplied (backward-compat); otherwise fully filled.
    """
    if d is None:
        d = SMOOTH_CFG.bilateral_d
    if sigma_color is None:
        sigma_color = SMOOTH_CFG.bilateral_sigma_color
    if sigma_space is None:
        sigma_space = SMOOTH_CFG.bilateral_sigma_space

    nan_mask = np.isnan(depth_map)
    if confidence_mask is not None:
        invalid_mask = nan_mask | (~confidence_mask)
    else:
        invalid_mask = nan_mask

    depth_filled = _inpaint_invalid(depth_map, invalid_mask)

    guide_float32 = guide_image.astype(np.float32)

    smoothed = cv2.ximgproc.jointBilateralFilter(
        joint=guide_float32,
        src=depth_filled,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    # Only re-mask pixels with NO information at all (raw NaN and, if given,
    # rejected by confidence too) — everything else now has a real,
    # locally-consistent value rather than a fabricated one.
    smoothed[invalid_mask] = np.nan

    return smoothed


def smooth_guided(
    depth_map: np.ndarray,
    guide_image: np.ndarray,
    confidence_mask: np.ndarray = None,
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
    confidence_mask : np.ndarray, shape (H, W), bool, optional
        True = reliable pixel. See smooth_bilateral docstring.

    Returns
    -------
    smoothed : np.ndarray, shape (H, W), float32
    """
    if radius is None:
        radius = SMOOTH_CFG.guided_radius
    if eps is None:
        eps = SMOOTH_CFG.guided_eps

    nan_mask = np.isnan(depth_map)
    if confidence_mask is not None:
        invalid_mask = nan_mask | (~confidence_mask)
    else:
        invalid_mask = nan_mask

    depth_filled = _inpaint_invalid(depth_map, invalid_mask)

    smoothed = cv2.ximgproc.guidedFilter(
        guide=guide_image,
        src=depth_filled,
        radius=radius,
        eps=eps,
    )

    smoothed[invalid_mask] = np.nan
    return smoothed


def smooth_depth_map(
    depth_map: np.ndarray,
    guide_image: np.ndarray,
    confidence_mask: np.ndarray = None,
    method: str = None,
) -> np.ndarray:
    """
    Smooth a depth map using the configured method.

    Parameters
    ----------
    depth_map : np.ndarray, shape (H, W), float32
    guide_image : np.ndarray, shape (H, W), float32
    confidence_mask : np.ndarray, shape (H, W), bool, optional
        True = reliable pixel, from confidence.compute_confidence(). Strongly
        recommended — without it, low-confidence-but-non-NaN pixels are the
        source of the spike artifacts.
    method : str, "bilateral" or "guided"

    Returns
    -------
    smoothed : np.ndarray, shape (H, W), float32
    """
    if method is None:
        method = SMOOTH_CFG.method

    if method == "bilateral":
        return smooth_bilateral(depth_map, guide_image, confidence_mask=confidence_mask)
    elif method == "guided":
        return smooth_guided(depth_map, guide_image, confidence_mask=confidence_mask)
    else:
        raise ValueError(f"Unknown smoothing method: {method}. Use 'bilateral' or 'guided'.")


def compare_smoothing_methods(
    depth_map: np.ndarray,
    guide_image: np.ndarray,
    confidence_mask: np.ndarray = None,
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
    bilateral = smooth_bilateral(depth_map, guide_image, confidence_mask=confidence_mask)
    guided = smooth_guided(depth_map, guide_image, confidence_mask=confidence_mask)

    diff = np.abs(bilateral - guided)
    valid = ~np.isnan(diff)

    return {
        "bilateral": bilateral,
        "guided": guided,
        "difference": diff,
        "max_diff": float(np.nanmax(diff)) if np.any(valid) else 0.0,
        "mean_diff": float(np.nanmean(diff)) if np.any(valid) else 0.0,
    }
