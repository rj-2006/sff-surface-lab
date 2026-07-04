"""
04_compare_focus_measures.py — Side-by-side comparison of SML, Laplacian², Tenengrad.

Usage:
    python scripts/04_compare_focus_measures.py [--dataset MAIN-SET1|leaf2]

Runs all three focus measures on the same stack, produces depth maps from each,
and generates a comparison figure documenting which method works best for our data.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, DIAGNOSTICS_DIR
from src.loader import load_stack
from src.focus_measure import compute_all_focus_measures
from src.depth_estimation import estimate_depth_gaussian, estimate_depth_argmax


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare focus measure operators")
    parser.add_argument("--dataset", type=str, default="MAIN-SET1")
    args = parser.parse_args()

    ds_path = DATA_DIR / args.dataset
    aligned_path = DATA_DIR / f"{args.dataset}_aligned"
    if aligned_path.is_dir():
        ds_path = aligned_path
        print(f"Using aligned stack: {aligned_path}")

    stack, metadata = load_stack(ds_path)

    print("\nComputing all focus measures (this takes a while)...")
    t0 = time.time()
    measures = compute_all_focus_measures(stack)
    print(f"Done in {time.time()-t0:.1f}s")

    results = {}
    for name, fv in measures.items():
        print(f"\nDepth estimation ({name})...")
        depth_gauss, r_squared, fit_valid = estimate_depth_gaussian(fv)
        depth_argmax = estimate_depth_argmax(fv)

        results[name] = {
            "focus_volume": fv,
            "depth_gaussian": depth_gauss,
            "depth_argmax": depth_argmax,
            "r_squared": r_squared,
            "fit_valid": fit_valid,
        }

    save_dir = DIAGNOSTICS_DIR / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)

    method_names = list(results.keys())
    n_methods = len(method_names)

    fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 10))

    for i, name in enumerate(method_names):
        r = results[name]

        ax = axes[0, i]
        dm = r["depth_gaussian"]
        im = ax.imshow(dm, cmap='viridis',
                       vmin=np.nanpercentile(dm, 1),
                       vmax=np.nanpercentile(dm, 99))
        plt.colorbar(im, ax=ax, shrink=0.7, label='µm')
        ax.set_title(f'{name}\n(Gaussian interp)')
        ax.axis('off')

        ax = axes[1, i]
        dm = r["depth_argmax"]
        im = ax.imshow(dm, cmap='viridis',
                       vmin=np.nanpercentile(dm, 1),
                       vmax=np.nanpercentile(dm, 99))
        plt.colorbar(im, ax=ax, shrink=0.7, label='µm')
        ax.set_title(f'{name}\n(Argmax)')
        ax.axis('off')

    fig.suptitle(f'{args.dataset} — Focus Measure Comparison', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_dir / "focus_measure_comparison_depth.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, n_methods - 1, figsize=(6*(n_methods-1), 5))
    if n_methods - 1 == 1:
        axes = [axes]

    ref_name = method_names[0]  # SML as reference
    ref_depth = results[ref_name]["depth_gaussian"]

    for i, name in enumerate(method_names[1:]):
        other_depth = results[name]["depth_gaussian"]
        diff = np.abs(ref_depth - other_depth)

        ax = axes[i]
        im = ax.imshow(diff, cmap='hot', vmax=np.nanpercentile(diff, 99))
        plt.colorbar(im, ax=ax, shrink=0.7, label='µm')

        valid = ~(np.isnan(ref_depth) | np.isnan(other_depth))
        mae = np.nanmean(diff[valid]) if np.any(valid) else 0
        ax.set_title(f'|{ref_name} - {name}|\nMAE = {mae:.3f} µm')
        ax.axis('off')

    fig.suptitle(f'{args.dataset} — Method Differences', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_dir / "focus_measure_comparison_diff.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for name in method_names:
        r2 = results[name]["r_squared"]
        valid = results[name]["fit_valid"]
        if np.any(valid):
            ax.hist(r2[valid].ravel(), bins=50, alpha=0.5, label=name, density=True)
    ax.set_xlabel('R² (Gaussian fit quality)')
    ax.set_ylabel('Density')
    ax.set_title('Fit Quality Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name in method_names:
        dm = results[name]["depth_gaussian"]
        valid = ~np.isnan(dm)
        if np.any(valid):
            ax.hist(dm[valid].ravel(), bins=50, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Depth (µm)')
    ax.set_ylabel('Density')
    ax.set_title('Depth Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{args.dataset} — Method Statistics', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_dir / "focus_measure_comparison_stats.png", dpi=150)
    plt.close(fig)

    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY — {args.dataset}")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Valid %':>10} {'Mean R²':>10} {'Depth Range':>15}")
    print(f"{'-'*55}")

    for name in method_names:
        r = results[name]
        dm = r["depth_gaussian"]
        valid = ~np.isnan(dm)
        pct = 100 * np.sum(valid) / dm.size
        r2_mean = np.mean(r["r_squared"][r["fit_valid"]]) if np.any(r["fit_valid"]) else 0
        d_range = f"{np.nanmin(dm):.1f}–{np.nanmax(dm):.1f}" if np.any(valid) else "N/A"
        print(f"{name:<15} {pct:>9.1f}% {r2_mean:>10.4f} {d_range:>15}")

    print(f"\nComparison plots saved to {save_dir}")


if __name__ == "__main__":
    main()
