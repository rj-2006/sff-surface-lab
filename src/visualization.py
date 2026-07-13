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

from .config import Z_STEP_UM, XY_UM_PER_PIXEL, DEPTH_MAP_DIR, MODEL_DIR


def plot_3d_surface(
    depth_map: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    dataset_name: str = "surface",
    save_dir: Optional[Path] = None,
    subsample: int = 2,
    raw_depth_map: Optional[np.ndarray] = None,
) -> Path:
    """
    Create an interactive 3D surface plot using Plotly.

    Parameters
    ----------
    depth_map : np.ndarray, shape (H, W), float32, in µm
    confidence : np.ndarray, shape (H, W), optional
        Confidence map for opacity control.
    subsample : int
        Downsample factor for performance (2 = every other pixel).
    raw_depth_map : np.ndarray, shape (H, W), optional
        Raw/unsmoothed depth map to allow toggle comparison.
    """
    import plotly.graph_objects as go

    if save_dir is None:
        save_dir = MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    dm = depth_map[::subsample, ::subsample]
    h, w = dm.shape

    dm_filled = dm.copy()
    if np.any(np.isnan(dm_filled)):
        # Use local inpainting instead of global median to avoid cliff artifacts
        from .smoothing import _inpaint_invalid
        nan_mask = np.isnan(dm_filled)
        dm_filled = _inpaint_invalid(dm_filled, nan_mask)

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

    traces = []
    
    # 1. Smoothed/Fixed surface (Visible by default)
    traces.append(
        go.Surface(
            x=x, y=y, z=dm_filled,
            colorscale='Viridis',
            name='Smoothed (Fixed)',
            showscale=True,
            colorbar=dict(
                title=dict(text='Depth (µm)', side='right')
            ),
            visible=True
        )
    )

    # 2. Original surface before fix (Hidden by default, uses old smoothing without confidence mask)
    has_raw = raw_depth_map is not None
    if has_raw:
        raw_dm = raw_depth_map[::subsample, ::subsample]
        raw_filled = raw_dm.copy()
        if np.any(np.isnan(raw_filled)):
            raw_filled = np.nan_to_num(raw_filled, nan=np.nanmedian(raw_dm))
        
        traces.append(
            go.Surface(
                x=x, y=y, z=raw_filled,
                colorscale='Viridis',
                name='Original (Before Fix)',
                showscale=True,
                colorbar=dict(
                    title=dict(text='Depth (µm)', side='right')
                ),
                visible=False
            )
        )

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

    if has_raw:
        layout_update['updatemenus'] = [
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True, False]}],
                        label="Smoothed (Fixed)",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True]}],
                        label="Original (Before Fix)",
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
