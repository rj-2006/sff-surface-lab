"""
drift_check.py — Input validation and drift detection for focus stacks.

Step 1 of the pipeline: Before trusting any depth computation, verify that
the input data satisfies the core assumption of shape-from-focus — that ONLY
the focal plane depth changes between frames, with no XY drift.

Three independent checks:
1. Global focus-energy curve: should be smooth and unimodal
2. Local (tiled) focus-energy curves: each tile should have a clean peak
3. Feature-point tracking: high-contrast points should stay fixed in XY

REF: This is not from a specific paper — it's a methodological safeguard.
The bug we caught earlier (truncated stack) would have been caught by check 1.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from .config import DRIFT_CFG, DIAGNOSTICS_DIR


def compute_global_focus_energy(stack: np.ndarray) -> np.ndarray:
    """
    Compute mean Laplacian energy for each frame in the stack.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W), float32 [0, 1]

    Returns
    -------
    energy : np.ndarray, shape (N,)
        Mean |∇²I|² for each frame.

    WHY: Laplacian energy is a quick, robust global sharpness metric.
    The frame with highest energy is the "globally sharpest" — where most
    of the surface is near the focal plane.
    """
    n_frames = stack.shape[0]
    energy = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        frame_uint8 = (stack[i] * 255).astype(np.uint8)
        lap = cv2.Laplacian(frame_uint8, cv2.CV_64F)
        energy[i] = np.mean(lap ** 2)

    return energy


def compute_local_focus_energy(
    stack: np.ndarray,
    grid: tuple[int, int] = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Compute focus energy for each tile in a grid, across all frames.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
    grid : tuple (rows, cols), default from config

    Returns
    -------
    tile_energy : np.ndarray, shape (n_tiles, N)
        Focus energy per tile per frame.
    tile_bounds : list of (y0, y1, x0, x1)
        Pixel bounds of each tile.
    """
    if grid is None:
        grid = DRIFT_CFG.tile_grid

    n_frames, h, w = stack.shape
    rows, cols = grid
    n_tiles = rows * cols

    tile_h = h // rows
    tile_w = w // cols

    tile_energy = np.zeros((n_tiles, n_frames), dtype=np.float64)
    tile_bounds = []

    idx = 0
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            tile_bounds.append((y0, y1, x0, x1))

            for i in range(n_frames):
                tile = stack[i, y0:y1, x0:x1]
                tile_uint8 = (tile * 255).astype(np.uint8)
                lap = cv2.Laplacian(tile_uint8, cv2.CV_64F)
                tile_energy[idx, i] = np.mean(lap ** 2)

            idx += 1

    return tile_energy, tile_bounds


def check_unimodality(energy: np.ndarray, label: str = "global") -> dict:
    """
    Check if a focus-energy curve has a single clean peak.

    Returns
    -------
    result : dict
        - is_unimodal: bool
        - peak_frame: int (index of highest peak)
        - peak_value: float
        - n_peaks: int
        - secondary_peaks: list of frame indices
        - warnings: list of str
    """
    e_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-10)

    peaks, properties = find_peaks(e_norm, prominence=0.1)

    peak_frame = int(np.argmax(energy))
    peak_value = float(energy[peak_frame])

    warnings = []

    if peak_frame == 0 or peak_frame == len(energy) - 1:
        warnings.append(
            f"Peak at frame {peak_frame} (edge of stack) — "
            "the focus sweep may be truncated!"
        )

    secondary = [p for p in peaks if p != peak_frame]

    result = {
        "label": label,
        "is_unimodal": len(peaks) <= 1,
        "peak_frame": peak_frame,
        "peak_value": peak_value,
        "n_peaks": len(peaks),
        "secondary_peaks": secondary,
        "warnings": warnings,
    }

    if not result["is_unimodal"]:
        warnings.append(
            f"Found {len(peaks)} peaks at frames {list(peaks)} — "
            "expected one smooth peak. Possible causes: vibration, "
            "drift, or an object at multiple depth levels."
        )

    return result


def track_features(
    stack: np.ndarray,
    n_features: int = None,
    template_size: int = None,
    search_margin: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Track high-contrast feature points across all frames.

    Detects features in the sharpest frame, then uses template matching
    to find them in every other frame.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)

    Returns
    -------
    positions : np.ndarray, shape (n_features, N, 2)
        (x, y) pixel position of each feature in each frame.
    displacements : np.ndarray, shape (n_features, N)
        Euclidean displacement from the reference position.

    WHY template matching over optical flow:
    Optical flow assumes small displacements and smooth motion — fine for video,
    but focus stacks have dramatic appearance changes (blur/sharp) that confuse
    flow. Template matching of small high-contrast patches is more robust here.
    """
    if n_features is None:
        n_features = DRIFT_CFG.n_features
    if template_size is None:
        template_size = DRIFT_CFG.template_size
    if search_margin is None:
        search_margin = DRIFT_CFG.search_margin

    n_frames, h, w = stack.shape
    half = template_size // 2

    global_energy = compute_global_focus_energy(stack)
    ref_idx = int(np.argmax(global_energy))
    ref_frame = (stack[ref_idx] * 255).astype(np.uint8)

    harris = cv2.cornerHarris(ref_frame, blockSize=5, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)  # NMS approximation

    margin = half + search_margin
    harris[:margin, :] = 0
    harris[-margin:, :] = 0
    harris[:, :margin] = 0
    harris[:, -margin:] = 0

    flat_indices = np.argsort(harris.ravel())[::-1][:n_features * 3]
    candidates_y, candidates_x = np.unravel_index(flat_indices, harris.shape)

    selected_x, selected_y = [], []
    for x, y in zip(candidates_x, candidates_y):
        if len(selected_x) >= n_features:
            break
        too_close = False
        for sx, sy in zip(selected_x, selected_y):
            if abs(x - sx) < 20 and abs(y - sy) < 20:
                too_close = True
                break
        if not too_close:
            selected_x.append(int(x))
            selected_y.append(int(y))

    actual_n = len(selected_x)
    positions = np.zeros((actual_n, n_frames, 2), dtype=np.float64)
    displacements = np.zeros((actual_n, n_frames), dtype=np.float64)

    for fi in range(actual_n):
        ref_x, ref_y = selected_x[fi], selected_y[fi]
        positions[fi, ref_idx] = [ref_x, ref_y]

        template = ref_frame[
            ref_y - half: ref_y + half + 1,
            ref_x - half: ref_x + half + 1,
        ]

        for frame_i in range(n_frames):
            if frame_i == ref_idx:
                continue

            frame = (stack[frame_i] * 255).astype(np.uint8)

            sy0 = max(0, ref_y - half - search_margin)
            sy1 = min(h, ref_y + half + 1 + search_margin)
            sx0 = max(0, ref_x - half - search_margin)
            sx1 = min(w, ref_x + half + 1 + search_margin)

            search_region = frame[sy0:sy1, sx0:sx1]

            if search_region.shape[0] < template.shape[0] or \
               search_region.shape[1] < template.shape[1]:
                positions[fi, frame_i] = [ref_x, ref_y]
                continue

            result = cv2.matchTemplate(
                search_region, template, cv2.TM_CCOEFF_NORMED
            )
            _, _, _, max_loc = cv2.minMaxLoc(result)

            found_x = sx0 + max_loc[0] + half
            found_y = sy0 + max_loc[1] + half
            positions[fi, frame_i] = [found_x, found_y]

            dx = found_x - ref_x
            dy = found_y - ref_y
            displacements[fi, frame_i] = np.sqrt(dx ** 2 + dy ** 2)

    return positions, displacements


def run_drift_check(
    stack: np.ndarray,
    dataset_name: str,
    save_dir: Optional[Path] = None,
) -> dict:
    """
    Run all drift checks on a focus stack.

    Returns
    -------
    result : dict with keys:
        - global_energy: np.ndarray
        - global_check: dict (unimodality check)
        - tile_energy: np.ndarray
        - tile_checks: list of dicts
        - positions: np.ndarray (feature positions)
        - displacements: np.ndarray
        - max_drift_px: float
        - drift_detected: bool
        - all_warnings: list of str
    """
    if save_dir is None:
        save_dir = DIAGNOSTICS_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Drift Check — {dataset_name}")
    print(f"{'='*60}")

    all_warnings = []

    print("\n1. Global focus-energy curve...")
    global_energy = compute_global_focus_energy(stack)
    global_check = check_unimodality(global_energy, "global")
    all_warnings.extend(global_check["warnings"])

    print(f"   Peak at frame {global_check['peak_frame']} "
          f"(energy={global_check['peak_value']:.2f})")
    print(f"   Unimodal: {'✓' if global_check['is_unimodal'] else '✗'}")
    if global_check["secondary_peaks"]:
        print(f"   Secondary peaks at frames: {global_check['secondary_peaks']}")

    print("\n2. Local focus-energy curves...")
    tile_energy, tile_bounds = compute_local_focus_energy(stack)
    tile_checks = []
    for t in range(tile_energy.shape[0]):
        tc = check_unimodality(tile_energy[t], f"tile_{t}")
        tile_checks.append(tc)
        all_warnings.extend(tc["warnings"])

    n_unimodal = sum(1 for tc in tile_checks if tc["is_unimodal"])
    print(f"   {n_unimodal}/{len(tile_checks)} tiles have unimodal peaks")

    peak_frames = [tc["peak_frame"] for tc in tile_checks]
    print(f"   Peak frame range: {min(peak_frames)} to {max(peak_frames)} "
          f"(span={max(peak_frames)-min(peak_frames)} frames)")

    print("\n3. Feature-point drift tracking...")
    positions, displacements = track_features(stack)
    max_drift = float(np.max(displacements))
    mean_drift = float(np.mean(np.max(displacements, axis=1)))

    drift_detected = max_drift > DRIFT_CFG.max_drift_px
    print(f"   Tracked {positions.shape[0]} features across {stack.shape[0]} frames")
    print(f"   Max drift: {max_drift:.3f} px")
    print(f"   Mean max drift per feature: {mean_drift:.3f} px")
    print(f"   Drift detected (>{DRIFT_CFG.max_drift_px} px): "
          f"{'YES ⚠' if drift_detected else 'NO ✓'}")

    if drift_detected:
        all_warnings.append(
            f"XY drift detected: max={max_drift:.2f} px. "
            f"Frame registration (Step 2) is recommended."
        )

    _plot_diagnostics(
        global_energy, tile_energy, tile_bounds, tile_checks,
        positions, displacements, stack.shape, dataset_name, save_dir
    )

    print(f"\n{'─'*40}")
    if all_warnings:
        print("⚠ WARNINGS:")
        for w in all_warnings:
            print(f"  • {w}")
    else:
        print("✓ All checks passed — stack looks clean.")
    print(f"Diagnostic plots saved to: {save_dir}")

    return {
        "global_energy": global_energy,
        "global_check": global_check,
        "tile_energy": tile_energy,
        "tile_checks": tile_checks,
        "positions": positions,
        "displacements": displacements,
        "max_drift_px": max_drift,
        "drift_detected": drift_detected,
        "all_warnings": all_warnings,
    }


def _plot_diagnostics(
    global_energy, tile_energy, tile_bounds, tile_checks,
    positions, displacements, stack_shape, dataset_name, save_dir,
):
    """Generate and save all diagnostic plots."""

    n_frames = len(global_energy)
    frames = np.arange(n_frames)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, global_energy, 'b-', linewidth=1.5)
    peak_idx = np.argmax(global_energy)
    ax.axvline(peak_idx, color='r', linestyle='--', alpha=0.7,
               label=f'Peak: frame {peak_idx}')
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Mean Laplacian energy')
    ax.set_title(f'{dataset_name} — Global Focus Energy vs. Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "global_focus_energy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    n_tiles = tile_energy.shape[0]
    cmap = plt.cm.tab20(np.linspace(0, 1, n_tiles))
    for t in range(n_tiles):
        te = tile_energy[t]
        te_norm = (te - te.min()) / (te.max() - te.min() + 1e-10)
        ax.plot(frames, te_norm, color=cmap[t], alpha=0.6, linewidth=0.8,
                label=f'Tile {t} (peak@{tile_checks[t]["peak_frame"]})')
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Normalized focus energy')
    ax.set_title(f'{dataset_name} — Local Focus Energy (per tile)')
    ax.legend(fontsize=6, ncol=4, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "local_focus_energy.png", dpi=150)
    plt.close(fig)

    if positions.shape[0] > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        for fi in range(positions.shape[0]):
            ax.plot(frames, displacements[fi], alpha=0.5, linewidth=0.8)
        ax.axhline(DRIFT_CFG.max_drift_px, color='r', linestyle='--',
                    alpha=0.7, label=f'Threshold: {DRIFT_CFG.max_drift_px} px')
        ax.set_xlabel('Frame index')
        ax.set_ylabel('Displacement (pixels)')
        ax.set_title(f'{dataset_name} — Feature Drift')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _, h, w = stack_shape
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect('equal')
        for fi in range(positions.shape[0]):
            ref_x, ref_y = positions[fi, np.argmax(global_energy)]
            ax.plot(ref_x, ref_y, 'r+', markersize=8)
        ax.set_title(f'{dataset_name} — Feature Locations')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        fig.tight_layout()
        fig.savefig(save_dir / "drift_tracking.png", dpi=150)
        plt.close(fig)
