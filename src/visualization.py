"""
visualization.py — 3D surface rendering and diagnostic visualizations.

All visualization outputs for the pipeline:
- Interactive 3D surface plot (Plotly → HTML)
- Depth map heatmaps (Matplotlib)
- All-in-focus composite
- Cross-section line profiles
- Confidence overlay
- Multi-panel diagnostic dashboard
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .config import Z_STEP_UM, XY_UM_PER_PIXEL, DEPTH_MAP_DIR, MODEL_DIR, BORDER_MARGIN_PX


# --- PATCH (2026-07-16) ---
# BORDER REMOVAL FOR THE 3D MODEL ONLY.
#
# WHY: smoothing.py's confidence-mask fix already prevents unreliable pixels
# from leaking through as *unfiltered* spikes — but it still inpaint-fills
# every unreliable pixel (including the border ring caused by vignetting /
# registration crop / edge-of-field defocus) with a locally-plausible height
# so the 2D depth map stays continuous. That fill is correct for the 2D map,
# but wrong for the 3D model: the border isn't the subject, and its filled
# heights (real or fabricated) stretch the z-axis range, which compresses
# the color range across the actual subject (everything reads as one shade
# of blue).
#
# FIX: plot_3d_surface now optionally takes the confidence-reliable `mask`
# and, using it, finds the unreliable region that is connected to the image
# edge (flood-fill / connected-component labeling from the four borders).
# That connected region is treated as "border" and set to NaN in the 3D
# z-array only — Plotly renders NaN as a gap, so the border simply isn't
# part of the mesh. Any unreliable pixels NOT connected to the edge (e.g.
# a genuine occlusion/hole inside the subject) are left to the normal
# inpaint fill, same as before, so real interior geometry isn't punched
# full of holes.
#
# This only touches plot_3d_surface. plot_depth_map, plot_confidence_overlay,
# plot_cross_sections, and plot_diagnostic_dashboard are unchanged and still
# show the full frame, border included, as intended.
def _find_border_region(invalid_mask: np.ndarray, margin_px: int = 0) -> np.ndarray:
    """
    Identify the subset of `invalid_mask` that forms a ring connected to
    the image border, as opposed to isolated invalid patches inside the
    subject (e.g. an occluded cavity).

    Parameters
    ----------
    invalid_mask : np.ndarray, shape (H, W), bool
        True where a pixel is NaN or confidence-rejected.
    margin_px : int
        Optional additional fixed-width frame (in pixels, at the
        resolution of invalid_mask) to force-include as border regardless
        of connectivity. 0 = automatic detection only.

    Returns
    -------
    border_mask : np.ndarray, shape (H, W), bool
        True where the pixel should be excluded from the 3D model as
        border.
    """
    h, w = invalid_mask.shape
    border_mask = np.zeros((h, w), dtype=bool)

    if np.any(invalid_mask):
        from scipy.ndimage import label
        # 8-connectivity so a diagonally-touching ring still merges into
        # one component instead of fragmenting at corners.
        structure = np.ones((3, 3), dtype=int)
        labeled, num = label(invalid_mask, structure=structure)
        if num > 0:
            edge_labels = set(labeled[0, :].tolist())
            edge_labels |= set(labeled[-1, :].tolist())
            edge_labels |= set(labeled[:, 0].tolist())
            edge_labels |= set(labeled[:, -1].tolist())
            edge_labels.discard(0)
            if edge_labels:
                border_mask = np.isin(labeled, list(edge_labels))

    if margin_px > 0:
        border_mask[:margin_px, :] = True
        border_mask[-margin_px:, :] = True
        border_mask[:, :margin_px] = True
        border_mask[:, -margin_px:] = True

    return border_mask


def plot_3d_surface(
    depth_map: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    dataset_name: str = "surface",
    save_dir: Optional[Path] = None,
    subsample: int = 2,
    mask: Optional[np.ndarray] = None,
    registration_mask: Optional[np.ndarray] = None,
    border_margin_px: Optional[int] = None,
) -> Path:
    """
    Create an interactive 3D surface plot using Plotly, with a toggle
    between two views of the same reconstruction:

    - "Smoothed (Cavity Filled)": border removed, interior low-confidence
      pixels (e.g. cavity floor/walls) inpaint-filled for a continuous
      mesh. Default view.
    - "Extra Smooth": a second, wider-parameter bilateral pass on top of
      the above, aimed at small residual spikes that survive the main
      smoothing.py pass.

    Color bounds for both traces are percentile-based (2nd/98th) rather
    than raw min/max, so a handful of outlier pixels can't stretch the
    colorscale thin across the rest of the surface — this is what keeps
    small genuine variation (e.g. chatter waviness) visually distinct
    instead of reading as a near-uniform shade.

    Parameters
    ----------
    depth_map : np.ndarray, shape (H, W), float32, in µm
    confidence : np.ndarray, shape (H, W), optional
        Confidence map for opacity control.
    subsample : int
        Downsample factor for performance (2 = every other pixel).
    mask : np.ndarray, shape (H, W), bool, optional
        Confidence-reliable mask (True = reliable), from
        confidence.compute_confidence(). Marks genuine interior
        ambiguity (e.g. cavity walls / occlusion). In the "Smoothed"
        trace these are inpaint-filled; in the "Raw" trace they're left
        as gaps.
    registration_mask : np.ndarray, shape (H, W), bool, optional
        Full-stack-coverage mask (True = real data in every layer), from
        registration.compute_valid_coverage_mask(). False marks the
        registration-warp-boundary artifact — a hard fill introduced
        wherever a frame's drift correction pushed real content outside
        the original field of view. Excluded (NaN) from BOTH traces,
        since it's never real data in either view. Only pass this for
        datasets that actually went through registration
        (02_register_frames.py) — no drift, no warp, nothing to exclude.
    border_margin_px : int, optional
        Extra fixed-width frame (pixels, full resolution before subsample)
        to force-exclude as border regardless of what the masks say.
        Defaults to config.BORDER_MARGIN_PX (0 = off). Acts as a safety
        net on top of the two masks above, not a replacement for them.
    """
    import plotly.graph_objects as go

    if save_dir is None:
        save_dir = MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    if border_margin_px is None:
        border_margin_px = BORDER_MARGIN_PX

    dm = depth_map[::subsample, ::subsample]
    h, w = dm.shape

    nan_mask = np.isnan(dm)

    # Registration-invalid pixels alone seed border detection (see PATCH
    # 2026-07-18b above _find_border_region) — confidence-invalid pixels
    # never count as border, even if adjacent, so a real interior feature
    # (like a cavity) can never get swallowed into the border exclusion.
    registration_invalid = np.zeros((h, w), dtype=bool)
    if registration_mask is not None:
        reg_mask_sub = registration_mask[::subsample, ::subsample]
        registration_invalid = ~reg_mask_sub

    margin_sub = max(0, border_margin_px // subsample) if border_margin_px else 0
    if np.any(registration_invalid) or margin_sub > 0:
        border_mask = _find_border_region(registration_invalid, margin_px=margin_sub)
    else:
        border_mask = np.zeros((h, w), dtype=bool)

    confidence_invalid = np.zeros((h, w), dtype=bool)
    if mask is not None:
        mask_sub = mask[::subsample, ::subsample]
        confidence_invalid = ~mask_sub

    interior_invalid = (confidence_invalid | nan_mask) & ~border_mask

    # --- "Smoothed (Cavity Filled)" variant ---
    dm_smoothed = dm.copy()
    if np.any(interior_invalid):
        from .smoothing import _inpaint_invalid
        dm_smoothed = _inpaint_invalid(dm_smoothed, interior_invalid)
    if np.any(border_mask):
        dm_smoothed[border_mask] = np.nan

    # --- "Extra Smooth" variant ---
    # A second, wider-parameter bilateral pass over the already cavity-
    # filled result, aimed at the small residual spikes that survive the
    # main smoothing.py pass (genuine but small — not the border/cavity
    # artifacts, which are already handled separately above). Self-guided
    # (no composite image needed here) since this is a display-polish
    # pass, not a measurement step.
    dm_extra_smooth = dm_smoothed.copy()
    if np.any(border_mask):
        from .smoothing import _inpaint_invalid
        # Temporarily fill the border so the filter has no NaN to choke
        # on — re-excluded right after, same as the main pass.
        dm_extra_smooth = _inpaint_invalid(dm_extra_smooth, border_mask)
    dm_extra_smooth = cv2.bilateralFilter(
        dm_extra_smooth.astype(np.float32), d=15, sigmaColor=50.0, sigmaSpace=25.0
    )
    if np.any(border_mask):
        dm_extra_smooth[border_mask] = np.nan

    if XY_UM_PER_PIXEL is not None:
        x = np.arange(w) * subsample * XY_UM_PER_PIXEL
        y = np.arange(h) * subsample * XY_UM_PER_PIXEL
        x_label = "X (µm)"
        y_label = "Y (µm)"
    else:
        x = np.arange(w) * subsample
        y = np.arange(h) * subsample
        x_label = "X (pixels)"
        y_label = "Y (pixels)"

    def _zrange_percentile(arr, low_pct=2.0, high_pct=98.0):
        """
        Percentile-based color bounds instead of raw min/max. A handful
        of remaining outlier pixels can otherwise stretch the colorscale
        thin across the rest of the surface, washing out genuine small
        variation (like chatter waviness) into a near-uniform shade.
        Clipping to the 2nd/98th percentile keeps those outliers on the
        surface (z/height is untouched — this only affects color), just
        clamped to the colorscale's extreme color instead of dominating
        its range.
        """
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return 0.0, 1.0
        lo = float(np.percentile(valid, low_pct))
        hi = float(np.percentile(valid, high_pct))
        if hi <= lo:
            lo, hi = float(valid.min()), float(valid.max())
        return lo, hi

    smoothed_zmin, smoothed_zmax = _zrange_percentile(dm_smoothed)
    extra_zmin, extra_zmax = _zrange_percentile(dm_extra_smooth)

    traces = [
        go.Surface(
            x=x, y=y, z=dm_smoothed,
            colorscale='Turbo',
            cmin=smoothed_zmin,
            cmax=smoothed_zmax,
            name='Smoothed (Cavity Filled)',
            showscale=True,
            colorbar=dict(
                title=dict(text='Depth (µm)', side='right')
            ),
            visible=True,
        ),
        go.Surface(
            x=x, y=y, z=dm_extra_smooth,
            colorscale='Turbo',
            cmin=extra_zmin,
            cmax=extra_zmax,
            name='Extra Smooth',
            showscale=True,
            colorbar=dict(
                title=dict(text='Depth (µm)', side='right')
            ),
            visible=False,
        ),
    ]

    fig = go.Figure(data=traces)

    layout_update = dict(
        title=dict(text=f'{dataset_name} — 3D Surface Reconstruction', font_size=16),
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title='Depth (µm)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=h/w, z=0.3),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        width=1000,
        height=700,
    )

    layout_update['updatemenus'] = [
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"visible": [True, False]}],
                    label="Smoothed (Cavity Filled)",
                    method="update"
                ),
                dict(
                    args=[{"visible": [False, True]}],
                    label="Extra Smooth",
                    method="update"
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.9,
            xanchor="right",
            y=1.15,
            yanchor="top"
        )
    ]


    fig.update_layout(**layout_update)

    html_path = save_dir / f"{dataset_name}_3d_surface.html"
    fig.write_html(str(html_path))
    print(f"  3D surface saved: {html_path}")

    return html_path


def plot_depth_map(
    depth_map: np.ndarray,
    dataset_name: str = "depth",
    save_dir: Optional[Path] = None,
    title: str = None,
) -> Path:
    """
    Plot depth map as a heatmap with colorbar in µm.
    """
    if save_dir is None:
        save_dir = DEPTH_MAP_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = f'{dataset_name} — Depth Map'

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        depth_map, cmap='viridis',
        vmin=np.nanpercentile(depth_map, 1),
        vmax=np.nanpercentile(depth_map, 99),
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Depth (µm)', fontsize=12)
    ax.set_title(title, fontsize=14)

    if XY_UM_PER_PIXEL is not None:
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
    else:
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

    fig.tight_layout()
    png_path = save_dir / f"{dataset_name}_depth_map.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"  Depth map saved: {png_path}")
    return png_path


def plot_confidence_overlay(
    depth_map: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    dataset_name: str = "confidence",
    save_dir: Optional[Path] = None,
) -> Path:
    """
    Plot depth map with low-confidence regions highlighted.
    """
    if save_dir is None:
        save_dir = DEPTH_MAP_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    im = ax.imshow(depth_map, cmap='viridis',
                   vmin=np.nanpercentile(depth_map, 1),
                   vmax=np.nanpercentile(depth_map, 99))
    plt.colorbar(im, ax=ax, shrink=0.7, label='µm')
    ax.set_title('Depth Map')

    ax = axes[1]
    im = ax.imshow(confidence, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.7, label='Confidence')
    ax.set_title('Confidence Map')

    ax = axes[2]
    masked_depth = depth_map.copy()
    masked_depth[~mask] = np.nan
    im = ax.imshow(masked_depth, cmap='viridis',
                   vmin=np.nanpercentile(depth_map, 1),
                   vmax=np.nanpercentile(depth_map, 99))
    plt.colorbar(im, ax=ax, shrink=0.7, label='µm')
    ax.set_title(f'Reliable Depth ({np.sum(mask)/mask.size*100:.1f}%)')

    fig.suptitle(f'{dataset_name} — Confidence Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()

    png_path = save_dir / f"{dataset_name}_confidence.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"  Confidence overlay saved: {png_path}")
    return png_path


def plot_cross_sections(
    depth_map: np.ndarray,
    rows: list[int] = None,
    cols: list[int] = None,
    dataset_name: str = "cross_section",
    save_dir: Optional[Path] = None,
) -> Path:
    """
    Plot depth cross-sections along specified rows and columns.
    """
    if save_dir is None:
        save_dir = DEPTH_MAP_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    h, w = depth_map.shape

    if rows is None:
        rows = [h // 4, h // 2, 3 * h // 4]
    if cols is None:
        cols = [w // 4, w // 2, 3 * w // 4]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    x_axis = np.arange(w)
    if XY_UM_PER_PIXEL is not None:
        x_axis = x_axis * XY_UM_PER_PIXEL
    for row in rows:
        ax.plot(x_axis, depth_map[row, :], label=f'Row {row}', linewidth=1)
    ax.set_xlabel('X (µm)' if XY_UM_PER_PIXEL else 'X (pixels)')
    ax.set_ylabel('Depth (µm)')
    ax.set_title('Horizontal Cross-Sections')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    y_axis = np.arange(h)
    if XY_UM_PER_PIXEL is not None:
        y_axis = y_axis * XY_UM_PER_PIXEL
    for col in cols:
        ax.plot(y_axis, depth_map[:, col], label=f'Col {col}', linewidth=1)
    ax.set_xlabel('Y (µm)' if XY_UM_PER_PIXEL else 'Y (pixels)')
    ax.set_ylabel('Depth (µm)')
    ax.set_title('Vertical Cross-Sections')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset_name} — Depth Cross-Sections', fontsize=14, fontweight='bold')
    fig.tight_layout()

    png_path = save_dir / f"{dataset_name}_cross_sections.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Cross-sections saved: {png_path}")
    return png_path


def save_composite(
    composite: np.ndarray,
    dataset_name: str = "composite",
    save_dir: Optional[Path] = None,
) -> Path:
    """Save the all-in-focus composite image."""
    if save_dir is None:
        save_dir = DEPTH_MAP_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    png_path = save_dir / f"{dataset_name}_all_in_focus.png"
    cv2.imwrite(str(png_path), (composite * 255).astype(np.uint8))
    print(f"  All-in-focus composite saved: {png_path}")
    return png_path


def plot_diagnostic_dashboard(
    depth_map: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    composite: np.ndarray,
    global_energy: np.ndarray,
    dataset_name: str = "dashboard",
    save_dir: Optional[Path] = None,
) -> Path:
    """
    Multi-panel diagnostic dashboard combining key visualizations.
    """
    if save_dir is None:
        save_dir = DEPTH_MAP_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(composite, cmap='gray')
    ax.set_title('All-in-Focus Composite')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(depth_map, cmap='viridis',
                   vmin=np.nanpercentile(depth_map, 1),
                   vmax=np.nanpercentile(depth_map, 99))
    plt.colorbar(im, ax=ax, shrink=0.7, label='µm')
    ax.set_title('Depth Map')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(confidence, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.7, label='Confidence')
    ax.set_title('Confidence Map')
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(global_energy, 'b-', linewidth=1.5)
    ax.axvline(np.argmax(global_energy), color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Focus energy')
    ax.set_title('Global Focus Energy')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    valid_depths = depth_map[mask & ~np.isnan(depth_map)]
    if len(valid_depths) > 0:
        ax.hist(valid_depths, bins=50, color='steelblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Depth (µm)')
    ax.set_ylabel('Pixel count')
    ax.set_title('Depth Distribution (reliable pixels)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    h, w = depth_map.shape
    mid_row = h // 2
    x = np.arange(w)
    ax.plot(x, depth_map[mid_row, :], 'b-', linewidth=1, label='Depth')
    ax.fill_between(x, depth_map[mid_row, :],
                    alpha=0.3, color='steelblue')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Depth (µm)')
    ax.set_title(f'Cross-Section (row {mid_row})')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset_name} — Diagnostic Dashboard',
                 fontsize=16, fontweight='bold')

    png_path = save_dir / f"{dataset_name}_dashboard.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Dashboard saved: {png_path}")
    return png_path