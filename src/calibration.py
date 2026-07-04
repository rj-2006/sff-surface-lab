"""
calibration.py — Lateral (X/Y) calibration for physical measurements.

Step 8 of the pipeline: Without knowing microns-per-pixel in X/Y, we can
only report depth, not cavity width, area, aspect ratio, or volume.

Two modes:
1. Manual: user provides µm/pixel from the microscope spec sheet
2. Grid: image a calibration grid of known spacing, auto-detect intersections

REF: Standard practice in quantitative microscopy.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import XY_UM_PER_PIXEL


def set_lateral_calibration(um_per_pixel: float) -> None:
    """
    Set the lateral pixel size globally.

    Parameters
    ----------
    um_per_pixel : float
        Microns per pixel in X and Y (assumes square pixels).
    """
    import src.config as cfg
    cfg.XY_UM_PER_PIXEL = um_per_pixel
    print(f"Lateral calibration set: {um_per_pixel:.4f} µm/pixel")


def calibrate_from_grid(
    grid_image: np.ndarray,
    known_spacing_um: float,
    pattern_size: tuple[int, int] = (7, 7),
) -> Optional[float]:
    """
    Calibrate lateral pixel size from a calibration grid image.

    Parameters
    ----------
    grid_image : np.ndarray, shape (H, W), uint8
        Image of a calibration grid (checkerboard pattern).
    known_spacing_um : float
        Known distance between grid intersections in microns.
    pattern_size : tuple (rows, cols)
        Number of inner corners in the checkerboard.

    Returns
    -------
    um_per_pixel : float or None
        Calibrated pixel size, or None if detection failed.
    """
    ret, corners = cv2.findChessboardCorners(grid_image, pattern_size)

    if not ret:
        print("⚠ Could not detect checkerboard corners.")
        print("  Try adjusting pattern_size or improving image contrast.")
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(grid_image, corners, (5, 5), (-1, -1), criteria)

    corners = corners.reshape(-1, 2)
    rows, cols = pattern_size

    pixel_distances = []
    for r in range(rows):
        for c in range(cols - 1):
            idx1 = r * cols + c
            idx2 = r * cols + c + 1
            d = np.linalg.norm(corners[idx1] - corners[idx2])
            pixel_distances.append(d)
    for r in range(rows - 1):
        for c in range(cols):
            idx1 = r * cols + c
            idx2 = (r + 1) * cols + c
            d = np.linalg.norm(corners[idx1] - corners[idx2])
            pixel_distances.append(d)

    mean_pixel_dist = np.mean(pixel_distances)
    std_pixel_dist = np.std(pixel_distances)

    um_per_pixel = known_spacing_um / mean_pixel_dist

    print(f"Grid calibration results:")
    print(f"  Detected {len(corners)} corners")
    print(f"  Mean grid spacing: {mean_pixel_dist:.2f} ± {std_pixel_dist:.2f} pixels")
    print(f"  Known spacing: {known_spacing_um:.2f} µm")
    print(f"  → Pixel size: {um_per_pixel:.4f} µm/pixel")

    set_lateral_calibration(um_per_pixel)
    return um_per_pixel


def measure_cavity(
    depth_map: np.ndarray,
    mask: np.ndarray,
    um_per_pixel: float = None,
    z_step_um: float = 1.0,
) -> dict:
    """
    Measure physical dimensions of a cavity (masked region in depth map).

    Parameters
    ----------
    depth_map : np.ndarray, shape (H, W), float32
        Depth map in µm.
    mask : np.ndarray, shape (H, W), bool
        True for pixels that are part of the cavity.
    um_per_pixel : float, optional
        Lateral pixel size. If None, reports in pixels.

    Returns
    -------
    measurements : dict
        - depth_max_um: maximum cavity depth
        - depth_mean_um: mean cavity depth
        - area_px: area in pixels
        - area_um2: area in µm² (if calibrated)
        - width_px, height_px: bounding box dimensions in pixels
        - width_um, height_um: bounding box in µm (if calibrated)
        - volume_um3: estimated volume (if calibrated)
        - aspect_ratio: width/depth
    """
    if um_per_pixel is None:
        um_per_pixel = XY_UM_PER_PIXEL

    valid = mask & ~np.isnan(depth_map)

    if not np.any(valid):
        return {"error": "No valid pixels in mask"}

    cavity_depth = depth_map[valid]
    surface_level = np.percentile(depth_map[~mask & ~np.isnan(depth_map)], 50)
    relative_depth = surface_level - cavity_depth  # positive = deeper

    measurements = {
        "depth_max_um": float(np.max(relative_depth)),
        "depth_mean_um": float(np.mean(relative_depth)),
        "depth_std_um": float(np.std(relative_depth)),
        "area_px": int(np.sum(valid)),
    }

    ys, xs = np.where(valid)
    measurements["width_px"] = int(xs.max() - xs.min() + 1)
    measurements["height_px"] = int(ys.max() - ys.min() + 1)

    if um_per_pixel is not None:
        px_area_um2 = um_per_pixel ** 2
        measurements["area_um2"] = measurements["area_px"] * px_area_um2
        measurements["width_um"] = measurements["width_px"] * um_per_pixel
        measurements["height_um"] = measurements["height_px"] * um_per_pixel

        measurements["volume_um3"] = float(np.sum(relative_depth) * px_area_um2)

        measurements["aspect_ratio"] = measurements["width_um"] / measurements["depth_max_um"]
    else:
        measurements["note"] = "Lateral calibration not set — width/area/volume unavailable"

    return measurements
