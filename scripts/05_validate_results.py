"""
05_validate_results.py — Run validation tests on reconstruction results.

Usage:
    python scripts/05_validate_results.py [--dataset MAIN-SET1|leaf2]
                                           [--mode split-half|ground-truth]
                                           [--ground-truth-file PATH]
"""

import sys
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, DIAGNOSTICS_DIR
from src.loader import load_stack
from src.depth_estimation import build_all_in_focus
from src.focus_measure import sum_modified_laplacian
from src.validation import split_half_validation, compare_with_ground_truth


def main():
    parser = argparse.ArgumentParser(description="Validate reconstruction results")
    parser.add_argument("--dataset", type=str, default="MAIN-SET1")
    parser.add_argument("--mode", type=str, default="split-half",
                        choices=["split-half", "ground-truth"])
    parser.add_argument("--ground-truth-file", type=str, default=None,
                        help="Path to ground-truth depth map (.npy)")
    args = parser.parse_args()

    ds_path = DATA_DIR / args.dataset
    aligned_path = DATA_DIR / f"{args.dataset}_aligned"
    if aligned_path.is_dir():
        ds_path = aligned_path

    stack, metadata = load_stack(ds_path)

    if args.mode == "split-half":
        fv = sum_modified_laplacian(stack)
        composite = build_all_in_focus(stack, fv)

        results = split_half_validation(
            stack, composite, dataset_name=args.dataset
        )

        print(f"\n{'─'*40}")
        if results["mae"] < 2.0:
            print(f"✓ Excellent consistency: MAE = {results['mae']:.3f} µm (< 2 µm)")
        elif results["mae"] < 5.0:
            print(f"~ Acceptable consistency: MAE = {results['mae']:.3f} µm (< 5 µm)")
        else:
            print(f"⚠ Poor consistency: MAE = {results['mae']:.3f} µm (≥ 5 µm)")
            print("  This suggests the pipeline may have issues. Check focus curves.")

    elif args.mode == "ground-truth":
        if args.ground_truth_file is None:
            print("ERROR: --ground-truth-file required for ground-truth mode")
            sys.exit(1)

        gt_path = Path(args.ground_truth_file)
        if not gt_path.exists():
            print(f"ERROR: Ground truth file not found: {gt_path}")
            sys.exit(1)

        from src.config import DEPTH_MAP_DIR
        depth_path = DEPTH_MAP_DIR / args.dataset / "depth_smoothed.npy"
        if not depth_path.exists():
            print(f"ERROR: Run 03_run_pipeline.py first to generate depth map")
            sys.exit(1)

        depth_map = np.load(depth_path)
        ground_truth = np.load(gt_path)

        results = compare_with_ground_truth(
            depth_map, ground_truth, dataset_name=args.dataset
        )


if __name__ == "__main__":
    main()
