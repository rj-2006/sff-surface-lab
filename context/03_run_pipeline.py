"""
03_run_pipeline.py — Full SFF reconstruction pipeline (Steps 3-6).

Usage:
    python scripts/03_run_pipeline.py [--dataset MAIN-SET1|leaf2|all]
                                       [--method sml|laplacian|tenengrad]
                                       [--smoothing bilateral|guided]
                                       [--no-smooth]
                                       [--save-intermediate]

Runs the complete pipeline:
1. Load stack (uses aligned frames if available)
2. Compute focus measure (SML by default)
3. Estimate depth with sub-frame Gaussian interpolation
4. Compute confidence scores
5. Apply edge-aware smoothing
6. Generate all outputs: depth map, 3D model, diagnostics
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, DEPTH_MAP_DIR, DIAGNOSTICS_DIR
from src.loader import load_stack
from src.drift_check import compute_global_focus_energy
from src.focus_measure import sum_modified_laplacian, laplacian_energy, tenengrad
from src.depth_estimation import estimate_depth_gaussian, estimate_depth_argmax, build_all_in_focus
from src.confidence import compute_confidence
from src.smoothing import smooth_depth_map
from src.visualization import (
    plot_3d_surface, plot_depth_map, plot_confidence_overlay,
    plot_cross_sections, save_composite, plot_diagnostic_dashboard,
)


FOCUS_METHODS = {
    "sml": sum_modified_laplacian,
    "laplacian": laplacian_energy,
    "tenengrad": tenengrad,
}


def run_pipeline(dataset_path: Path, args) -> dict:
    """Run the full pipeline on one dataset."""
    name = dataset_path.name.replace("_aligned", "")
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"SFF Pipeline — {name}")
    print(f"{'='*60}")

    print("\n[1/6] Loading stack...")
    stack, metadata = load_stack(dataset_path)
    n_frames, h, w = stack.shape

    print(f"\n[2/6] Computing focus measure ({args.method.upper()})...")
    focus_fn = FOCUS_METHODS[args.method]
    focus_volume = focus_fn(stack)

    print("\n[3/6] Building all-in-focus composite...")
    composite = build_all_in_focus(stack, focus_volume)

    print("\n[4/6] Estimating depth (Gaussian interpolation)...")
    depth_map, r_squared, fit_valid = estimate_depth_gaussian(focus_volume)

    depth_argmax = estimate_depth_argmax(focus_volume)

    print("\n[5/6] Computing confidence scores...")
    confidence, mask = compute_confidence(focus_volume, r_squared, fit_valid)

    n_masked = int(np.sum(~mask))
    print(f"  Confidence-rejected pixels: {n_masked}/{mask.size} "
          f"({100 * n_masked / mask.size:.1f}%)")

    if not args.no_smooth:
        print(f"\n[6/6] Smoothing depth map ({args.smoothing})...")
        # PATCH (2026-07-08): previously `mask` was computed but never passed
        # here, so low-confidence pixels (flat focus curves, poor Gaussian
        # fits, scan-range-edge peaks) flowed straight into the edge-aware
        # smoother, which preserved them as fake "edges" instead of removing
        # them -> spikes around the border and in low-texture interior
        # regions. Passing confidence_mask makes smooth_depth_map inpaint
        # those pixels from their reliable local neighbors before smoothing.
        depth_smoothed = smooth_depth_map(
            depth_map, composite, confidence_mask=mask, method=args.smoothing
        )
    else:
        print("\n[6/6] Skipping smoothing (--no-smooth)")
        depth_smoothed = depth_map.copy()
        depth_smoothed[~mask] = np.nan

    print(f"\nSaving outputs...")
    output_dir = DEPTH_MAP_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = DIAGNOSTICS_DIR / name
    diag_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "depth_raw.npy", depth_map)
    np.save(output_dir / "depth_smoothed.npy", depth_smoothed)
    np.save(output_dir / "depth_argmax.npy", depth_argmax)
    np.save(output_dir / "confidence.npy", confidence)
    np.save(output_dir / "mask.npy", mask)
    print(f"  Arrays saved to {output_dir}")

    if args.save_intermediate:
        np.save(output_dir / "focus_volume.npy", focus_volume)
        np.save(output_dir / "r_squared.npy", r_squared)
        print("  Intermediate arrays saved (focus_volume, r_squared)")

    print("\nGenerating visualizations...")
    global_energy = compute_global_focus_energy(stack)

    save_composite(composite, name, output_dir)
    plot_depth_map(depth_smoothed, name, output_dir, f"{name} — Smoothed Depth Map")
    plot_depth_map(depth_map, f"{name}_raw", output_dir, f"{name} — Raw Depth Map")
    plot_confidence_overlay(depth_smoothed, confidence, mask, name, output_dir)
    plot_cross_sections(depth_smoothed, dataset_name=name, save_dir=output_dir)
    plot_3d_surface(depth_smoothed, confidence, name)
    plot_diagnostic_dashboard(
        depth_smoothed, confidence, mask, composite,
        global_energy, name, output_dir,
    )

    interp_diff = np.abs(depth_map - depth_argmax)
    valid = ~np.isnan(interp_diff)
    print(f"\nInterpolation improvement over argmax:")
    print(f"  Mean shift: {np.nanmean(interp_diff):.4f} µm")
    print(f"  Max shift:  {np.nanmax(interp_diff):.4f} µm")

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline complete in {elapsed:.1f}s")

    return {
        "depth_map": depth_smoothed,
        "depth_raw": depth_map,
        "confidence": confidence,
        "mask": mask,
        "composite": composite,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Run SFF reconstruction pipeline")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset name or 'all'")
    parser.add_argument("--method", type=str, default="sml",
                        choices=list(FOCUS_METHODS.keys()),
                        help="Focus measure method (default: sml)")
    parser.add_argument("--smoothing", type=str, default="bilateral",
                        choices=["bilateral", "guided"],
                        help="Smoothing method (default: bilateral)")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Skip depth smoothing")
    parser.add_argument("--save-intermediate", action="store_true",
                        help="Save focus volume and R² arrays")
    args = parser.parse_args()

    if args.dataset == "all":
        raw_dirs = sorted([d for d in DATA_DIR.iterdir()
                          if d.is_dir() and not d.name.endswith("_aligned")])
        datasets = []
        for d in raw_dirs:
            aligned = DATA_DIR / f"{d.name}_aligned"
            if aligned.is_dir():
                print(f"Using aligned stack for {d.name}")
                datasets.append(aligned)
            else:
                datasets.append(d)
    else:
        aligned = DATA_DIR / f"{args.dataset}_aligned"
        if aligned.is_dir():
            print(f"Using aligned stack for {args.dataset}")
            datasets = [aligned]
        else:
            datasets = [DATA_DIR / args.dataset]

    if not datasets:
        print(f"No datasets found in {DATA_DIR}")
        sys.exit(1)

    for ds in datasets:
        if not ds.is_dir():
            print(f"⚠ Not found: {ds}")
            continue
        run_pipeline(ds, args)


if __name__ == "__main__":
    main()
