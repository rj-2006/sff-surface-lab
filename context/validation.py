"""
validation.py — Ground-truth validation and internal consistency testing.

Step 7 of the pipeline: Verify that depth estimates are correct.

Two modes:
1. Ground-truth comparison (when external data is available):
   Compare our depth map against profilometer data or a known-height standard.

2. Internal consistency (always runnable):
   Split the stack into odd/even frames, reconstruct independently from each,
   and check if the two depth maps agree. This tests reproducibility, not
   absolute accuracy — but it's the best we can do without external reference.

--- PATCH (2026-07-08) ---
FIX: split_half_validation() previously discarded r_squared/fit_valid from
estimate_depth_gaussian() with `_, _` and never called compute_confidence().
That meant every depth estimate — including low-prominence, poor-fit, and
scan-range-edge-clipped pixels that confidence scoring exists specifically
to catch — flowed straight into smoothing and the final heightfield. Because
smooth_bilateral() is edge-aware, it preserved those bad estimates as if
they were real geometry, producing spikes (border ring + interior patches
in flat/textureless regions).

FIX: now calls compute_confidence() on both odd and even reconstructions,
applies BOTH masks (a pixel must be reliable in both sub-stacks) before
smoothing, and reports the pre/post spike-affected pixel count so this is
auditable in the report rather than silent.

REF: Standard practice in metrology — repeatability is a prerequisite for accuracy.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import DIAGNOSTICS_DIR, Z_STEP_UM
from .focus_measure import sum_modified_laplacian
from .depth_estimation import estimate_depth_gaussian
from .confidence import compute_confidence
from .smoothing import smooth_depth_map


def split_half_validation(
    stack: np.ndarray,
    guide_image: np.ndarray,
    dataset_name: str = "dataset",
    save_dir: Optional[Path] = None,
    apply_confidence_mask: bool = True,
    apply_smoothing: bool = True,
) -> dict:
    """
    Internal consistency test: reconstruct from odd and even frames independently.

    If the pipeline is working correctly, both sub-stacks should produce
    depth maps that agree within ~1-2 µm (since the sub-stacks have 2x step size).

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
    guide_image : np.ndarray, shape (H, W)
    dataset_name : str
    save_dir : Path, optional
    apply_confidence_mask : bool
        If True (default), reject low-confidence pixels from BOTH sub-stack
        reconstructions before computing agreement statistics or smoothing.
        Set False only to reproduce the old (buggy) unmasked behavior for
        comparison.
    apply_smoothing : bool
        If True (default), also run edge-aware smoothing on each masked
        depth map using compute_confidence's mask, so the returned depth
        maps match what actually feeds the final heightfield.

    Returns
    -------
    results : dict with keys:
        - depth_odd, depth_even: masked+smoothed depth maps from each sub-stack
        - confidence_odd, confidence_even: confidence maps
        - mask_odd, mask_even: reliable-pixel boolean masks
        - n_masked_odd, n_masked_even: count of pixels rejected by confidence
        - difference: pixel-wise absolute difference (masked pixels excluded)
        - mae, rmse, max_error: summary statistics (µm), computed only over
          pixels reliable in BOTH sub-stacks
        - correlation: Pearson correlation between the two maps
    """
    if save_dir is None:
        save_dir = DIAGNOSTICS_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    n_frames = stack.shape[0]

    odd_indices = np.arange(0, n_frames, 2)
    even_indices = np.arange(1, n_frames, 2)

    stack_odd = stack[odd_indices]
    stack_even = stack[even_indices]

    print(f"\nSplit-half validation:")
    print(f"  Odd frames: {len(odd_indices)} (step size = {2*Z_STEP_UM:.1f} µm)")
    print(f"  Even frames: {len(even_indices)} (step size = {2*Z_STEP_UM:.1f} µm)")

    print("\n  Reconstructing from odd frames...")
    fv_odd = sum_modified_laplacian(stack_odd)
    depth_odd, r2_odd, valid_odd = estimate_depth_gaussian(fv_odd, z_step_um=2 * Z_STEP_UM)

    print("\n  Reconstructing from even frames...")
    fv_even = sum_modified_laplacian(stack_even)
    depth_even, r2_even, valid_even = estimate_depth_gaussian(fv_even, z_step_um=2 * Z_STEP_UM)

    depth_even_corrected = depth_even + Z_STEP_UM

    conf_odd, mask_odd = compute_confidence(fv_odd, r2_odd, valid_odd)
    conf_even, mask_even = compute_confidence(fv_even, r2_even, valid_even)

    n_masked_odd = int(np.sum(~mask_odd))
    n_masked_even = int(np.sum(~mask_even))
    print(f"\n  Confidence masking:")
    print(f"    Odd sub-stack:  {n_masked_odd} pixels rejected "
          f"({100 * n_masked_odd / mask_odd.size:.1f}%)")
    print(f"    Even sub-stack: {n_masked_even} pixels rejected "
          f"({100 * n_masked_even / mask_even.size:.1f}%)")

    if apply_confidence_mask:
        depth_odd = depth_odd.copy()
        depth_even_corrected = depth_even_corrected.copy()
        depth_odd[~mask_odd] = np.nan
        depth_even_corrected[~mask_even] = np.nan

    if apply_smoothing:
        depth_odd = smooth_depth_map(
            depth_odd, guide_image,
            confidence_mask=mask_odd if apply_confidence_mask else None,
        )
        depth_even_corrected = smooth_depth_map(
            depth_even_corrected, guide_image,
            confidence_mask=mask_even if apply_confidence_mask else None,
        )

    # Agreement stats only over pixels reliable in BOTH reconstructions —
    # otherwise a pixel confidently wrong in one sub-stack could still be
    # averaged against a masked-out NaN neighbor and hide the disagreement.
    both_reliable = mask_odd & mask_even if apply_confidence_mask else np.ones_like(mask_odd)
    valid = both_reliable & ~(np.isnan(depth_odd) | np.isnan(depth_even_corrected))
    diff = np.abs(depth_odd - depth_even_corrected)

    if np.sum(valid) > 0:
        mae = float(np.nanmean(diff[valid]))
        rmse = float(np.sqrt(np.nanmean(diff[valid] ** 2)))
        max_error = float(np.nanmax(diff[valid]))

        from scipy.stats import pearsonr
        corr, _ = pearsonr(depth_odd[valid], depth_even_corrected[valid])
    else:
        mae = rmse = max_error = float('nan')
        corr = 0.0

    print(f"\n  Agreement statistics (confidence-masked, n={np.sum(valid)}):")
    print(f"    MAE:  {mae:.3f} µm")
    print(f"    RMSE: {rmse:.3f} µm")
    print(f"    Max:  {max_error:.3f} µm")
    print(f"    Correlation: {corr:.6f}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    im = ax.imshow(depth_odd, cmap='viridis')
    ax.set_title('Depth (odd frames, masked+smoothed)')
    plt.colorbar(im, ax=ax, label='µm')

    ax = axes[1]
    im = ax.imshow(depth_even_corrected, cmap='viridis')
    ax.set_title('Depth (even frames, masked+smoothed)')
    plt.colorbar(im, ax=ax, label='µm')

    ax = axes[2]
    diff_display = np.where(valid, diff, np.nan)
    vmax = np.nanpercentile(diff_display, 99) if np.any(valid) else 1.0
    im = ax.imshow(diff_display, cmap='hot', vmax=vmax)
    ax.set_title(f'|Difference| (MAE={mae:.2f} µm)')
    plt.colorbar(im, ax=ax, label='µm')

    fig.suptitle(f'{dataset_name} — Split-Half Consistency', fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_dir / "split_half_validation.png", dpi=150)
    plt.close(fig)

    return {
        "depth_odd": depth_odd,
        "depth_even": depth_even_corrected,
        "confidence_odd": conf_odd,
        "confidence_even": conf_even,
        "mask_odd": mask_odd,
        "mask_even": mask_even,
        "n_masked_odd": n_masked_odd,
        "n_masked_even": n_masked_even,
        "difference": diff,
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "correlation": corr,
    }


def compare_with_ground_truth(
    depth_map: np.ndarray,
    ground_truth: np.ndarray,
    pixel_size_um: float = 1.0,
    dataset_name: str = "dataset",
    save_dir: Optional[Path] = None,
) -> dict:
    """
    Compare our depth map against external ground-truth data.

    Parameters
    ----------
    depth_map : np.ndarray, shape (H, W)
        Our estimated depth map in µm.
    ground_truth : np.ndarray, shape (H', W')
        Reference depth data (e.g., profilometer). May need alignment.
    pixel_size_um : float
        Lateral pixel size for spatial alignment.

    Returns
    -------
    results : dict with error statistics and aligned maps.
    """
    if save_dir is None:
        save_dir = DIAGNOSTICS_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if ground_truth.shape != depth_map.shape:
        ground_truth = cv2.resize(
            ground_truth,
            (depth_map.shape[1], depth_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    valid = ~(np.isnan(depth_map) | np.isnan(ground_truth))
    diff = depth_map - ground_truth

    if np.sum(valid) > 0:
        mae = float(np.nanmean(np.abs(diff[valid])))
        rmse = float(np.sqrt(np.nanmean(diff[valid] ** 2)))
        max_error = float(np.nanmax(np.abs(diff[valid])))
        bias = float(np.nanmean(diff[valid]))

        from scipy.stats import pearsonr
        corr, _ = pearsonr(depth_map[valid], ground_truth[valid])
    else:
        mae = rmse = max_error = bias = float('nan')
        corr = 0.0

    print(f"\nGround-truth comparison ({dataset_name}):")
    print(f"  MAE:  {mae:.3f} µm")
    print(f"  RMSE: {rmse:.3f} µm")
    print(f"  Max:  {max_error:.3f} µm")
    print(f"  Bias: {bias:.3f} µm")
    print(f"  Correlation: {corr:.6f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "bias": bias,
        "correlation": corr,
        "difference": diff,
    }
